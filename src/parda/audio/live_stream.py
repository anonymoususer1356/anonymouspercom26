#!/usr/bin/env python3
# Consume a continuous PCM16 microphone stream with resident deployment models.

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import apply_agc, resample_audio
from lseend import (
    OnlineLSEEND,
    activity_matrix,
    build_segments,
    load_metadata,
)
from live_pipeline import (
    ActivityRouter,
    OverlapWorker,
    StreamingASR,
    TrackManager,
    lock_external_thread_pools,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_MODELS = PROJECT_ROOT / "models" / "deployment"
RAW_MODELS = PROJECT_ROOT / "models" / "raw"


# Replace a status file atomically so readers never observe partial JSON.
def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# Read up to one processing block, retaining an incomplete final block at EOF.
def read_pcm_block(stream, bytes_per_block: int) -> bytes:
    chunks = []
    remaining = bytes_per_block
    while remaining:
        data = stream.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


# Convert little-endian PCM16 to the 16 kHz floating-point model input.
def decode_block(data: bytes, input_sample_rate: int, use_agc: bool) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("PCM16 input ended in the middle of a sample")
    pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    audio = resample_audio(pcm, input_sample_rate, 16000)
    return apply_agc(audio, 16000) if use_agc else audio


# Save the source artifacts shared by file and microphone deployments.
def save_results(args, mixture, metadata, router, tracks, asr, origin, events) -> None:
    sf.write(args.output / "normalised_input.wav", mixture, 16000, subtype="FLOAT")
    activity = router.activity()
    slots = (
        [index for index in range(activity.shape[1]) if activity[:, index].any()]
        if activity.size
        else []
    )
    write_json(
        args.output / "diarization.json",
        build_segments(activity, slots, float(metadata["frame_hz"])),
    )
    write_json(args.output / "overlap_assignments.json", router.overlaps)

    detailed = list(events)
    for event in router.timing_events + tracks.timing_events:
        item = dict(event)
        item["start_seconds"] = round(item.pop("started") - origin, 6)
        item["end_seconds"] = round(item.pop("finished") - origin, 6)
        detailed.append(item)
    detailed.extend(asr.detailed_events)
    write_json(
        args.output / "detailed_timing.json",
        {"origin": "source_worker_ready", "events": sorted(detailed, key=lambda item: item["start_seconds"])},
    )
    write_json(
        args.output / "source_manifest.json",
        {
            "mode": "live_pcm16",
            "input_sample_rate": args.input_sample_rate,
            "model_sample_rate": 16000,
            "audio_seconds": round(len(mixture) / 16000, 6),
            "chunk_seconds": args.chunk_seconds,
            "agc": not args.no_agc,
            "prewarmed_speaker_slots": args.prewarm_speakers,
            "models": {
                "lseend": str(args.lseend_model),
                "convtasnet_2": str(args.convtasnet_2_onnx),
                "convtasnet_3": str(args.convtasnet_3_onnx),
                "speaker_encoder": str(args.speaker_encoder_model),
                "nemotron_encoder": str(args.nemotron_encoder),
            },
            "threads": {
                "lseend": args.lseend_threads,
                "separation": args.separation_threads,
                "track_speaker_encoder": args.speaker_encoder_threads,
                "overlap_speaker_encoder": args.overlap_speaker_encoder_threads,
                "asr": args.asr_threads,
            },
        },
    )


# Keep every source-stage model resident while microphone blocks arrive on stdin.
def run(args: argparse.Namespace) -> None:
    lock_external_thread_pools()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for speaker-track writing")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tracks").mkdir(exist_ok=True)
    status_path = args.output / "source_status.json"
    write_json(status_path, {"phase": "loading_models", "received_seconds": 0.0, "processed_chunks": 0})

    metadata = load_metadata(args.lseend_metadata)
    tracks = TrackManager(
        args.output,
        None,
        args.speaker_encoder_model,
        args.speaker_encoder_threads,
        args.profile_update_seconds,
    )
    overlap = OverlapWorker(args)
    asr = StreamingASR(args, args.output)
    lseend = OnlineLSEEND(
        args.lseend_model,
        metadata,
        args.lseend_threads,
        onnx_optimization="all",
        memory_pattern=True,
        allow_spinning=False,
    )
    if args.warmup_models:
        lseend.warmup()
    tracks.prewarm(args.prewarm_speakers)
    overlap.wait_until_ready()
    asr.prewarm(args.prewarm_speakers)

    origin = time.perf_counter()
    write_json(args.output / "source_clock.json", {
        "started_monotonic": origin,
        "started_unix": time.time(),
    })
    asr.set_benchmark_origin(origin)
    mixture = np.zeros(0, dtype=np.float32)
    router = ActivityRouter(
        mixture,
        float(metadata["frame_hz"]),
        args,
        tracks,
        overlap,
        asr,
    )
    timing_events = []
    bytes_per_block = round(args.chunk_seconds * args.input_sample_rate) * 2
    chunk_index = 0
    finished = False

    try:
        write_json(status_path, {"phase": "ready", "received_seconds": 0.0, "processed_chunks": 0})
        print("source worker ready", flush=True)
        while True:
            data = read_pcm_block(sys.stdin.buffer, bytes_per_block)
            if not data:
                break
            capture_finished = time.perf_counter()
            audio = decode_block(data, args.input_sample_rate, not args.no_agc)
            router.append_audio(audio)
            mixture = router.mixture
            available_seconds = len(mixture) / 16000

            lseend_started = time.perf_counter()
            diarisation = lseend.accept(resample_audio(audio, 16000, 8000))
            lseend_finished = time.perf_counter()
            router_started = time.perf_counter()
            router.add(diarisation, available_seconds)
            router.process_ready(final=False)
            tracks.drain_profiles()
            router_finished = time.perf_counter()

            source_start = max(0.0, available_seconds - len(audio) / 16000)
            timing_events.extend([
                {
                    "stage": "source_capture",
                    "start_seconds": round(source_start, 6),
                    "end_seconds": round(available_seconds, 6),
                    "chunk": chunk_index,
                },
                {
                    "stage": "lseend",
                    "start_seconds": round(lseend_started - origin, 6),
                    "end_seconds": round(lseend_finished - origin, 6),
                    "chunk": chunk_index,
                },
                {
                    "stage": "router",
                    "start_seconds": round(router_started - origin, 6),
                    "end_seconds": round(router_finished - origin, 6),
                    "chunk": chunk_index,
                },
            ])
            write_json(status_path, {
                "phase": "streaming",
                "received_seconds": round(available_seconds, 3),
                "processed_chunks": chunk_index + 1,
                "last_chunk_processing_seconds": round(router_finished - capture_finished, 6),
                "resident_speaker_slots": len(tracks.processes),
                "overlap_regions": len(router.overlaps),
            })
            chunk_index += 1

        final_events = lseend.finish()
        router.add(final_events, len(mixture) / 16000)
        router.process_ready(final=True)
        overlap.finish()
        asr.finish()
        tracks.finish(total_samples=len(mixture))
        finished = True
        save_results(args, mixture, metadata, router, tracks, asr, origin, timing_events)
        write_json(status_path, {
            "phase": "complete",
            "received_seconds": round(len(mixture) / 16000, 3),
            "processed_chunks": chunk_index,
            "resident_speaker_slots": len(tracks.processes),
            "written_speaker_tracks": len(list((args.output / "tracks").glob("SPEAKER_*.mkv"))),
            "overlap_regions": len(router.overlaps),
        })
        print(f"source worker wrote {args.output}", flush=True)
    except Exception as error:
        write_json(status_path, {
            "phase": "failed",
            "received_seconds": round(len(mixture) / 16000, 3),
            "processed_chunks": chunk_index,
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        if not finished:
            try:
                overlap.finish()
            except Exception:
                pass
            try:
                asr.finish()
            except Exception:
                pass
            try:
                tracks.finish(total_samples=len(mixture))
            except Exception:
                pass


# Parse the raw-stream protocol and the same deployment settings as live_pipeline.py.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resident source pipeline reading PCM16 mono from stdin")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-sample-rate", type=int, required=True)
    parser.add_argument("--no-agc", action="store_true")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--activity-threshold", type=float, default=0.4)
    parser.add_argument("--overlap-collar-seconds", type=float, default=0.0)
    parser.add_argument("--maximum-overlap-seconds", type=float, default=3.0)
    parser.add_argument("--always-separate-speakers", type=int, choices=(1, 2, 3))
    parser.add_argument("--minimum-similarity", type=float, default=0.2)
    parser.add_argument("--profile-update-seconds", type=float, default=1.0)
    parser.add_argument("--turn-hangover-seconds", type=float, default=1.2)
    parser.add_argument("--turn-fallback-silence-seconds", type=float, default=2.4)
    parser.add_argument("--turn-max-utterance-seconds", type=float, default=30.0)
    parser.add_argument("--prewarm-speakers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--warmup-models", action="store_true",
                        help="Execute a disposable LS-EEND step before the capture clock.")
    parser.add_argument("--lseend-threads", type=int, default=1)
    parser.add_argument("--separation-threads", type=int, default=2)
    parser.add_argument("--speaker-encoder-threads", "--campplus-threads", dest="speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--overlap-speaker-encoder-threads", "--overlap-cam-threads", dest="overlap_speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--asr-threads", type=int, default=3)
    parser.add_argument("--asr-batch-ready", action="store_true")
    parser.add_argument("--speaker-encoder-batch-overlap", "--cam-batch-overlap", dest="speaker_encoder_batch_overlap", action="store_true")
    parser.add_argument("--lseend-model", type=Path, default=DEPLOYMENT_MODELS / "lseend" / "ls_eend_dih3_step_ort_optimized.onnx")
    parser.add_argument("--lseend-metadata", type=Path, default=DEPLOYMENT_MODELS / "lseend" / "ls_eend_dih3_step.json")
    parser.add_argument("--convtasnet-2", type=Path, default=RAW_MODELS / "convtasnet_2" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-3", type=Path, default=RAW_MODELS / "convtasnet_3" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-2-onnx", type=Path, default=DEPLOYMENT_MODELS / "convtasnet_2" / "convtasnet_2_sepnoisy_ort_optimized.onnx")
    parser.add_argument("--convtasnet-3-onnx", type=Path, default=DEPLOYMENT_MODELS / "convtasnet_3" / "convtasnet_3_sepnoisy_ort_optimized.onnx")
    parser.add_argument("--speaker-encoder-model", "--campplus-model", dest="speaker_encoder_model", type=Path,
                        default=DEPLOYMENT_MODELS / "speaker_encoder_b1" / "redimnet_b1_5s_ort_enable_all.onnx")
    parser.add_argument("--nemotron-encoder", type=Path, default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "encoder.int8.onnx")
    parser.add_argument("--nemotron-decoder", type=Path, default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "decoder.int8.onnx")
    parser.add_argument("--nemotron-joiner", type=Path, default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "joiner.int8.onnx")
    parser.add_argument("--nemotron-tokens", type=Path, default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "tokens.txt")
    args = parser.parse_args()

    if args.input_sample_rate < 8000 or args.chunk_seconds <= 0:
        parser.error("input sample rate and chunk seconds must be positive")
    required = (
        args.lseend_model,
        args.lseend_metadata,
        args.convtasnet_2,
        args.convtasnet_3,
        args.convtasnet_2_onnx,
        args.convtasnet_3_onnx,
        args.speaker_encoder_model,
        args.nemotron_encoder,
        args.nemotron_decoder,
        args.nemotron_joiner,
        args.nemotron_tokens,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing required models:\n" + "\n".join(missing))
    return args


if __name__ == "__main__":
    run(parse_args())
