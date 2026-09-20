#!/usr/bin/env python3
# Run the deployable live audio path: LS-EEND -> conditional separation -> ReDimNet-B1 -> Moonshine.
#
# Input is a file for repeatable testing, but it is released in fixed five-second
# chunks. The routing decisions use only audio that has already been released.

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import apply_agc, load_audio, resample_audio
from lseend import (
    OnlineLSEEND,
    activity_matrix,
    build_segments,
    load_metadata,
)
from separation import ConvTasNetSeparator, SandglassetSeparator, sample_masks
from speaker_embeddings import ReDimNetEncoder, assign_channels, normalize_embedding


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_MODELS = PROJECT_ROOT / "models" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "audio"
ONNX_CACHE = PROJECT_ROOT / "outputs" / "model_cache"
STOP = object()
sys.path.insert(0, str(PROJECT_ROOT / "src" / "parda" / "utils"))
from semi_pacing import SemiPacer

WORD_START_RE = re.compile(r"^(?:\s+|▁)")
PUNCTUATION_RE = re.compile(r"^[\W_]+$", re.UNICODE)


# Keep non-ORT numerical libraries from competing with the explicitly allocated workers.
def lock_external_thread_pools() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


# Retain speech frames for speaker embedding without including long silent regions in its input.
def speech_only(audio: np.ndarray, sample_rate: int, minimum_rms: float = 1e-4) -> np.ndarray:
    frame_samples = max(1, round(0.1 * sample_rate))
    frames = []
    for start in range(0, len(audio), frame_samples):
        frame = np.asarray(audio[start : start + frame_samples], dtype=np.float32)
        if len(frame) and float(np.sqrt(np.mean(frame * frame))) >= minimum_rms:
            frames.append(frame)
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


# Feed silence to an FFmpeg pipe without allocating a recording-length array.
def write_silence(stream, samples: int, sample_rate: int) -> None:
    block = np.zeros(sample_rate, dtype="<f4").tobytes()
    while samples > 0:
        count = min(samples, sample_rate)
        stream.write(block[: count * 4])
        samples -= count


# Summarize a list of stage durations without retaining a second copy of raw events.
def timing_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_seconds": None, "maximum_seconds": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean_seconds": round(float(array.mean()), 6),
        "maximum_seconds": round(float(array.max()), 6),
    }


# One speaker track owns one growing MKV file and one independent ReDimNet-B1 session.
def track_worker(
    speaker: str,
    output_path: str,
    embedding_path: str,
    total_samples: int | None,
    speaker_encoder_model: str,
    speaker_encoder_threads: int,
    profile_updates,
    audio_queue,
    profile_update_seconds: float,
    ready_event,
    idle_ack=None,
) -> None:
    process = None
    try:
        encoder = ReDimNetEncoder(
            Path(speaker_encoder_model),
            cpu_threads=speaker_encoder_threads,
            onnx_optimization="all",
            memory_pattern=False,
            allow_spinning=False,
        )
        ready_event.set()
    except Exception:
        ready_event.set()
        raise

    cursor = 0
    pending = []
    pending_samples = 0
    weighted_embedding = None
    total_embedding_samples = 0
    update_samples = max(8000, round(profile_update_seconds * 16000))

    # Open a writer only when this prewarmed speaker slot receives audio.
    def start_writer():
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
            "-c:a", "pcm_s16le", output_path,
        ]
        writer = subprocess.Popen(command, stdin=subprocess.PIPE)
        if writer.stdin is None:
            raise RuntimeError("FFmpeg did not create an input pipe")
        return writer

    # Update a long-term fingerprint from speech only after enough new speech arrives.
    def update_profile(force: bool = False) -> None:
        nonlocal pending, pending_samples, weighted_embedding, total_embedding_samples
        if pending_samples < update_samples and not force:
            return
        if pending_samples < 8000:
            return
        speech = np.concatenate(pending)
        pending, pending_samples = [], 0
        started = time.perf_counter()
        embedding = encoder.embed_long_audio(speech, 16000)
        finished = time.perf_counter()
        profile_updates.put(("timing", speaker, {
            "stage": "track_speaker_embedding",
            "started": started,
            "finished": finished,
            "speech_seconds": round(len(speech) / 16000, 6),
        }))
        if embedding is None:
            return
        weight = len(speech)
        weighted_embedding = embedding * weight if weighted_embedding is None else weighted_embedding + embedding * weight
        total_embedding_samples += weight
        profile = normalize_embedding(weighted_embedding / total_embedding_samples)
        profile_updates.put(("profile", speaker, profile))

    try:
        while True:
            item = audio_queue.get()
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "idle_barrier":
                idle_ack.value = item[1]
                continue
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "finish":
                total_samples = int(item[1])
                break
            start_sample, audio = item
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if start_sample < cursor:
                audio = audio[cursor - start_sample :]
                start_sample = cursor
            if not len(audio):
                continue
            if process is None:
                process = start_writer()
            if start_sample > cursor:
                write_silence(process.stdin, start_sample - cursor, 16000)
                cursor = start_sample
            process.stdin.write(audio.astype("<f4", copy=False).tobytes())
            cursor += len(audio)
            speech = speech_only(audio, 16000)
            if len(speech):
                pending.append(speech)
                pending_samples += len(speech)
                update_profile()

        update_profile(force=True)
        if process is None:
            profile_updates.put(("finished", speaker, None))
            return
        if total_samples is not None and cursor < total_samples:
            write_silence(process.stdin, total_samples - cursor, 16000)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("FFmpeg failed while finalising the track")
        if weighted_embedding is not None:
            np.save(embedding_path, normalize_embedding(weighted_embedding / total_embedding_samples))
        profile_updates.put(("finished", speaker, None))
    except Exception as error:
        if process is not None and process.poll() is None:
            process.kill()
        profile_updates.put(("error", speaker, f"{type(error).__name__}: {error}"))


class TrackManager:
    # Start a writer/fingerprinter only when LS-EEND first activates a speaker slot.
    def __init__(self, output: Path, total_samples: int | None, speaker_encoder_model: Path, threads: int, update_seconds: float):
        self.output = output
        self.total_samples = total_samples
        self.speaker_encoder_model = speaker_encoder_model
        self.threads = threads
        self.update_seconds = update_seconds
        self.context = mp.get_context("spawn")
        self.updates = self.context.Queue()
        self.queues = {}
        self.processes = {}
        self.ready_events = {}
        self.profiles = {}
        self.errors = []
        self.timing_events = []
        self.idle_acks = {}
        self.idle_generation = 0

    # Create the speaker-specific process and its append-only audio queue.
    def ensure(self, speaker: str) -> None:
        if speaker in self.processes:
            return
        audio_queue = self.context.Queue(maxsize=64)
        ready_event = self.context.Event()
        idle_ack = self.context.Value("i", 0)
        process = self.context.Process(
            target=track_worker,
            args=(
                speaker,
                str(self.output / "tracks" / f"{speaker}.mkv"),
                str(self.output / "tracks" / f"{speaker}_speaker_embedding.npy"),
                self.total_samples,
                str(self.speaker_encoder_model),
                self.threads,
                self.updates,
                audio_queue,
                self.update_seconds,
                ready_event,
                idle_ack,
            ),
            name=f"track-{speaker}",
        )
        process.start()
        self.queues[speaker] = audio_queue
        self.processes[speaker] = process
        self.ready_events[speaker] = ready_event
        self.idle_acks[speaker] = idle_ack

    # A queue marker confirms every preceding write and embedding has completed.
    def idle_barriers(self) -> list:
        self.idle_generation += 1
        generation = self.idle_generation
        checks = []
        for speaker, audio_queue in self.queues.items():
            audio_queue.put(("idle_barrier", generation))
            acknowledgement = self.idle_acks[speaker]
            checks.append(lambda ack=acknowledgement, expected=generation: ack.value >= expected)
        return checks

    # Load one idle speaker-encoder worker for every possible LS-EEND output slot.
    def prewarm(self, speaker_count: int) -> None:
        speakers = [f"SPEAKER_{index:02d}" for index in range(speaker_count)]
        for speaker in speakers:
            self.ensure(speaker)
        for speaker in speakers:
            ready = self.ready_events[speaker]
            if not ready.wait(timeout=120):
                process = self.processes[speaker]
                raise RuntimeError(
                    f"Speaker-encoder worker did not become ready: {speaker} "
                    f"(exitcode={process.exitcode})"
                )

    # Append one correctly timestamped piece of audio to a speaker track.
    def send(self, speaker: str, start_sample: int, audio: np.ndarray) -> None:
        if not len(audio):
            return
        self.ensure(speaker)
        self.queues[speaker].put((start_sample, np.asarray(audio, dtype=np.float32)))

    # Incorporate asynchronous fingerprint updates before overlap channel matching.
    def drain_profiles(self) -> None:
        while True:
            try:
                kind, speaker, value = self.updates.get_nowait()
            except queue.Empty:
                return
            if kind == "profile":
                self.profiles[speaker] = value
            elif kind == "timing":
                value["speaker"] = speaker.lower()
                self.timing_events.append(value)
            elif kind == "error":
                self.errors.append(f"{speaker}: {value}")

    # Return the most recent fingerprints for exactly the active overlap speakers.
    def snapshot(self, speakers: list[str]) -> dict:
        self.drain_profiles()
        return {speaker: self.profiles.get(speaker) for speaker in speakers}

    # Flush writers, finalise MKVs and surface any child-process failure.
    def finish(self, total_samples: int | None = None) -> None:
        final_samples = self.total_samples if total_samples is None else total_samples
        for audio_queue in self.queues.values():
            if final_samples is None:
                audio_queue.put(None)
            else:
                audio_queue.put(("finish", final_samples))
        for speaker, process in self.processes.items():
            process.join()
            if process.exitcode != 0:
                self.errors.append(f"{speaker}: exited with {process.exitcode}")
        self.drain_profiles()
        if self.errors:
            raise RuntimeError("Track worker failures:\n" + "\n".join(self.errors))


class OverlapWorker:
    # Keep the selected separator and overlap ReDimNet-B1 matcher resident in one worker.
    def __init__(self, args: argparse.Namespace):
        self.jobs = queue.Queue(maxsize=4)
        self.results = queue.Queue(maxsize=4)
        self.args = args
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="overlap-worker", daemon=True)
        self.thread.start()

    # Construct sessions once, then resolve each queued overlap in timeline order.
    def _run(self) -> None:
        try:
            if self.args.separator_backend == "sandglasset":
                separator = SandglassetSeparator(
                    self.args.sandglasset_2_onnx,
                    self.args.sandglasset_3_onnx,
                    onnx_threads=self.args.separation_threads,
                )
            else:
                separator = ConvTasNetSeparator(
                    self.args.convtasnet_2,
                    self.args.convtasnet_3,
                    "cpu",
                    two_speaker_onnx_model=self.args.convtasnet_2_onnx if self.args.convtasnet_2_onnx.exists() else None,
                    three_speaker_onnx_model=self.args.convtasnet_3_onnx if self.args.convtasnet_3_onnx.exists() else None,
                    onnx_threads=self.args.separation_threads,
                    onnx_optimization="all",
                    onnx_memory_pattern=False,
                    onnx_allow_spinning=False,
                    input_sample_rate=16000,
                    model_sample_rate=8000,
                )
            separator.get_onnx_session(2)
            separator.get_onnx_session(3)
            matcher = ReDimNetEncoder(
                self.args.speaker_encoder_model,
                cpu_threads=self.args.overlap_speaker_encoder_threads,
                onnx_optimization="all",
                memory_pattern=False,
                allow_spinning=False,
            )
            self.ready.set()
            while True:
                job = self.jobs.get()
                if job is STOP:
                    return
                started = time.perf_counter()
                separation_started = time.perf_counter()
                sources = separator.separate(job["context_audio"], len(job["speakers"]))
                separation_finished = time.perf_counter()
                timings = [{
                    "stage": "overlap_separation",
                    "started": separation_started,
                    "finished": separation_finished,
                    "speaker_count": len(job["speakers"]),
                }]
                # Saturation fixtures keep real separation in the critical path,
                # while feeding known dense speech to the benchmark-only matcher.
                matching_sources = job.get("benchmark_clean_sources") or sources
                speech_sources = [speech_only(source, 16000) for source in matching_sources]
                if self.args.speaker_encoder_batch_overlap and all(len(source) >= 8000 for source in speech_sources):
                    embedding_started = time.perf_counter()
                    embeddings = matcher.embed_many(speech_sources)
                    embedding_finished = time.perf_counter()
                    timings.append({
                        "stage": "overlap_speaker_embedding_batch",
                        "started": embedding_started,
                        "finished": embedding_finished,
                        "channels": list(range(len(speech_sources))),
                        "batch_size": len(speech_sources),
                        "speech_seconds": [round(len(source) / 16000, 6) for source in speech_sources],
                    })
                else:
                    embeddings = []
                    for channel, source in enumerate(speech_sources):
                        embedding_started = time.perf_counter()
                        embedding = matcher.embed_long_audio(source, 16000) if len(source) >= 8000 else None
                        embedding_finished = time.perf_counter()
                        timings.append({
                            "stage": "overlap_speaker_embedding",
                            "started": embedding_started,
                            "finished": embedding_finished,
                            "channel": channel,
                            "speech_seconds": round(len(source) / 16000, 6),
                        })
                        embeddings.append(embedding if embedding is not None else np.zeros(matcher.embedding_dimension, dtype=np.float32))
                assignments = assign_channels(
                    embeddings,
                    job["speakers"],
                    job["profiles"],
                    self.args.minimum_similarity,
                )
                crop_start = job["core_start"] - job["context_start"]
                crop_end = crop_start + job["core_samples"]
                cropped = {
                    item["speaker"]: sources[item["channel"]][crop_start:crop_end].astype(np.float32)
                    for item in assignments
                }
                self.results.put({
                    "cropped": cropped,
                    "assignments": assignments,
                    "seconds": time.perf_counter() - started,
                    "timings": timings,
                    "diagnostics": separator.last_diagnostics,
                })
        except Exception as error:
            self.ready.set()
            self.results.put({"error": f"{type(error).__name__}: {error}"})

    # Ensure separator and overlap CAM++ sessions are resident before timing.
    def wait_until_ready(self) -> None:
        if not self.ready.wait(timeout=120):
            raise RuntimeError("Overlap worker did not become ready")
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return
        if "error" in result:
            raise RuntimeError(result["error"])
        self.results.put(result)

    # Submit one overlap and wait for its matching ordered result.
    def resolve(self, job: dict) -> dict:
        self.jobs.put(job)
        result = self.results.get()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    # Release all resident model resources after the source stream ends.
    def finish(self) -> None:
        self.jobs.put(STOP)
        self.thread.join()


class StreamingASR:
    # Consume identified speaker audio with persistent Nemotron stream states.
    def __init__(self, args: argparse.Namespace, output: Path):
        import sherpa_onnx

        self.args = args
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(args.nemotron_encoder), decoder=str(args.nemotron_decoder),
            joiner=str(args.nemotron_joiner), tokens=str(args.nemotron_tokens),
            num_threads=args.asr_threads, sample_rate=16000, feature_dim=80,
            decoding_method="greedy_search", provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=args.turn_fallback_silence_seconds,
            rule2_min_trailing_silence=args.turn_hangover_seconds,
            rule3_min_utterance_length=args.turn_max_utterance_seconds,
        )
        self.streams = {}
        self.queue = queue.Queue(maxsize=64)
        self.hypotheses_output = output / "transcript.jsonl"
        self.lines_output = output / "timestamped_lines.jsonl"
        self.words_output = output / "word_timestamps.jsonl"
        self.endpoint_output = output / "turn_endpointing.jsonl"
        self.benchmark_origin = None
        self.timing_events = []
        self.detailed_events = []
        self.timing_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="nemotron", daemon=True)
        self.thread.start()

    # Align ASR completion records with the same live-clock origin as LS-EEND.
    def set_benchmark_origin(self, origin: float) -> None:
        self.benchmark_origin = origin

    # Record one ASR phase on the common pipeline clock for a detailed trace.
    def record_detailed(self, stage: str, speaker: str, started: float, finished: float, **details) -> None:
        if self.benchmark_origin is None:
            return
        event = {
            "stage": stage,
            "speaker": speaker.lower(),
            "start_seconds": round(started - self.benchmark_origin, 6),
            "end_seconds": round(finished - self.benchmark_origin, 6),
        }
        event.update(details)
        with self.timing_lock:
            self.detailed_events.append(event)

    # Get the persistent recognizer cache belonging to one output speaker track.
    def stream(self, speaker: str):
        if speaker not in self.streams:
            stream = self.recognizer.create_stream()
            stream.set_option("language", "en")
            self.streams[speaker] = stream
        return self.streams[speaker]

    # Create all persistent recognizer states before the live clock starts.
    def prewarm(self, speaker_count: int) -> None:
        for index in range(speaker_count):
            self.stream(f"SPEAKER_{index:02d}")

    # Decode a stream until its current output is stable enough to inspect.
    def decode(self, stream) -> str:
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)
        result = self.recognizer.get_result(stream)
        return result.text if hasattr(result, "text") else str(result)

    # Decode aligned ready streams together without changing their independent caches.
    def decode_many(self, streams: list[object]) -> list[str]:
        if len(streams) == 1:
            return [self.decode(streams[0])]
        while True:
            ready = [stream for stream in streams if self.recognizer.is_ready(stream)]
            if not ready:
                break
            self.recognizer.decode_streams(ready)
        results = [self.recognizer.get_result(stream) for stream in streams]
        return [result.text if hasattr(result, "text") else str(result) for result in results]

    # Convert Sherpa's subword pieces into display words with approximate time spans.
    def word_timings(self, result, state: dict) -> list[dict]:
        tokens = list(getattr(result, "tokens", []) or [])
        timestamps = list(getattr(result, "timestamps", []) or [])
        if len(tokens) != len(timestamps):
            return []

        words = []
        current = None
        origin = float(state["stream_origin"])
        token_times = []
        for token, timestamp in zip(tokens, timestamps):
            piece = str(token)
            clean = piece.lstrip().removeprefix("▁")
            if not clean or (clean.startswith("<") and clean.endswith(">")):
                continue
            starts_word = bool(WORD_START_RE.match(piece)) or piece.startswith("▁")
            is_punctuation = bool(PUNCTUATION_RE.fullmatch(clean))
            start_seconds = origin + float(timestamp)
            token_times.append(start_seconds)
            if current is None or (starts_word and not is_punctuation):
                if current is not None:
                    words.append(current)
                current = {
                    "word": clean,
                    "start_seconds": start_seconds,
                    "last_token_seconds": start_seconds,
                }
            else:
                current["word"] += clean
                current["last_token_seconds"] = start_seconds
        if current is not None:
            words.append(current)

        utterance_end = float(state["last_end"])
        distinct_times = sorted(set(token_times))
        gaps = [right - left for left, right in zip(distinct_times, distinct_times[1:]) if right > left]
        final_token_span = min(gaps) if gaps else 0.32
        final_token_span = min(1.12, max(0.02, final_token_span))
        for index, word in enumerate(words):
            next_start = (
                words[index + 1]["start_seconds"]
                if index + 1 < len(words)
                else word["last_token_seconds"] + final_token_span
            )
            word["start_seconds"] = round(max(float(state["line_start"]), word["start_seconds"]), 3)
            word["end_seconds"] = round(max(word["start_seconds"], min(utterance_end, next_start)), 3)
            del word["last_token_seconds"]
        return words

    # Write one complete native-ASR utterance after a confirmed endpoint.
    def commit_line(self, speaker: str, state: dict, result, boundary: str,
                    committed_seconds: float, lines, words) -> None:
        text = result.text if hasattr(result, "text") else str(result)
        text = text.strip()
        if not text:
            return
        line = {
            "speaker": speaker.lower(),
            "start_seconds": round(state["line_start"], 3),
            "end_seconds": round(state["last_end"], 3),
            "committed_seconds": round(committed_seconds, 3),
            "boundary": boundary,
            "text": text,
        }
        lines.write(json.dumps(line, ensure_ascii=False) + "\n")
        lines.flush()
        words.write(json.dumps({
            "speaker": line["speaker"],
            "start_seconds": line["start_seconds"],
            "end_seconds": line["end_seconds"],
            "boundary": boundary,
            "timing_source": "nemotron_token_timestamps",
            "words": self.word_timings(result, state),
        }, ensure_ascii=False) + "\n")
        words.flush()
        print(f"[TURN {line['start_seconds']:7.2f}-{line['end_seconds']:7.2f}s] {speaker}: {line['text']}", flush=True)

    # Keep the partial-hypothesis log separate from committed endpoint decisions.
    def log_hypothesis(self, speaker: str, observed_seconds: float, state: dict, text: str, hypotheses) -> None:
        if text and text != state["hypothesis"]:
            state["hypothesis"] = text
            event = {"speaker": speaker, "observed_seconds": round(observed_seconds, 3), "hypothesis": text}
            print(f"[ASR {observed_seconds:7.2f}s] {speaker}: {text}", flush=True)
            hypotheses.write(json.dumps(event, ensure_ascii=False) + "\n")
            hypotheses.flush()

    # Feed only enough elapsed silence for Sherpa's native endpoint rules to fire.
    def advance_streams(self, next_start: float, states: dict, lines, words,
                        endpoint_log, hypotheses) -> None:
        for speaker, state in states.items():
            silence_seconds = next_start - state["asr_clock"]
            if silence_seconds <= 0:
                continue
            if not state["active_utterance"]:
                state["asr_clock"] = next_start
                continue
            stream = self.stream(speaker)
            endpoint = False
            injected_seconds = 0.0
            endpoint_limit = min(
                silence_seconds,
                self.args.turn_fallback_silence_seconds + 0.5,
            )
            while injected_seconds < endpoint_limit and not endpoint:
                step_seconds = min(0.4, endpoint_limit - injected_seconds)
                silence = np.zeros(round(step_seconds * 16000), dtype=np.float32)
                accept_started = time.perf_counter()
                stream.accept_waveform(16000, silence)
                accept_finished = time.perf_counter()
                self.record_detailed(
                    "asr_silence_accept", speaker, accept_started, accept_finished,
                    silence_seconds=round(step_seconds, 6),
                )
                decode_started = time.perf_counter()
                text = self.decode(stream)
                decode_finished = time.perf_counter()
                self.record_detailed(
                    "asr_silence_decode", speaker, decode_started, decode_finished,
                    silence_seconds=round(step_seconds, 6),
                )
                injected_seconds += step_seconds
                observed_seconds = min(next_start, state["asr_clock"] + injected_seconds)
                self.log_hypothesis(speaker, observed_seconds, state, text, hypotheses)
                endpoint = self.recognizer.is_endpoint(stream)
            record = {
                "speaker": speaker.lower(),
                "previous_audio_end_seconds": round(state["last_end"], 3),
                "next_audio_start_seconds": round(next_start, 3),
                "elapsed_silence_seconds": round(silence_seconds, 3),
                "injected_silence_seconds": round(injected_seconds, 3),
                "native_endpoint": endpoint,
            }
            if endpoint:
                result = self.recognizer.get_result_all(stream)
                self.commit_line(
                    speaker,
                    state,
                    result,
                    "sherpa_native_endpoint",
                    state["last_end"] + self.args.turn_hangover_seconds,
                    lines,
                    words,
                )
                self.recognizer.reset(stream)
                state["hypothesis"] = ""
                state["new_utterance"] = True
                state["active_utterance"] = False
            endpoint_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            endpoint_log.flush()
            state["asr_clock"] = next_start

    # Decode timestamped speaker audio and use Sherpa's native endpoint detector.
    def _run(self) -> None:
        states = {}
        pending = []
        stopped = False
        with (
            self.hypotheses_output.open("w", encoding="utf-8") as hypotheses,
            self.lines_output.open("w", encoding="utf-8") as lines,
            self.words_output.open("w", encoding="utf-8") as words,
            self.endpoint_output.open("w", encoding="utf-8") as endpoint_log,
        ):
            while pending or not stopped:
                item = pending.pop(0) if pending else self.queue.get()
                if item is STOP:
                    stopped = True
                    continue
                if item[0] == "idle_barrier":
                    item[1].set()
                    continue
                batch = [item]
                if self.args.asr_batch_ready:
                    first_start, first_end = item[1:3]
                    while True:
                        try:
                            next_item = self.queue.get_nowait()
                        except queue.Empty:
                            break
                        if next_item is STOP:
                            stopped = True
                            break
                        if next_item[0] == "idle_barrier":
                            pending.append(next_item)
                            break
                        if next_item[1:3] == (first_start, first_end):
                            batch.append(next_item)
                        else:
                            pending.append(next_item)
                            break

                start, end = batch[0][1:3]
                self.advance_streams(start, states, lines, words, endpoint_log, hypotheses)
                streams, prepared = [], []
                for speaker, item_start, item_end, audio, enqueued in batch:
                    dequeued = time.perf_counter()
                    self.record_detailed(
                        "asr_queue_wait", speaker, enqueued, dequeued,
                        source_start_seconds=round(item_start, 6),
                        source_end_seconds=round(item_end, 6),
                    )
                    stream = self.stream(speaker)
                    state = states.setdefault(speaker, {
                        "line_start": item_start, "last_end": item_start,
                        "asr_clock": item_start, "hypothesis": "",
                        "new_utterance": False, "active_utterance": False,
                        "stream_origin": item_start,
                    })
                    if state["new_utterance"]:
                        state["line_start"] = item_start
                        state["stream_origin"] = item_start
                        state["new_utterance"] = False
                    accept_started = time.perf_counter()
                    stream.accept_waveform(16000, np.asarray(audio, dtype=np.float32))
                    accept_finished = time.perf_counter()
                    self.record_detailed(
                        "asr_speech_accept", speaker, accept_started, accept_finished,
                        source_start_seconds=round(item_start, 6),
                        source_end_seconds=round(item_end, 6),
                    )
                    state["last_end"] = item_end
                    state["asr_clock"] = item_end
                    state["active_utterance"] = True
                    streams.append(stream)
                    prepared.append((speaker, state, item_end))

                started = time.perf_counter()
                texts = self.decode_many(streams)
                finished = time.perf_counter()
                if len(batch) > 1:
                    self.record_detailed(
                        "asr_speech_decode_batch", "batch", started, finished,
                        speakers=[speaker.lower() for speaker, _, _ in prepared],
                        batch_size=len(batch), source_start_seconds=round(start, 6),
                        source_end_seconds=round(end, 6),
                    )
                for (speaker, state, item_end), stream, text in zip(prepared, streams, texts):
                    if len(batch) == 1:
                        self.record_detailed(
                            "asr_speech_decode", speaker, started, finished,
                            source_start_seconds=round(start, 6),
                            source_end_seconds=round(item_end, 6),
                        )
                    self.log_hypothesis(speaker, item_end, state, text, hypotheses)
                    if self.benchmark_origin is not None:
                        with self.timing_lock:
                            self.timing_events.append({
                                "speaker": speaker.lower(),
                                "source_end_seconds": round(item_end, 6),
                                "decode_seconds": round((finished - started) / len(batch), 6),
                                "shared_decode_seconds": round(finished - started, 6),
                                "batch_size": len(batch),
                                "completion_lag_seconds": round(
                                    finished - (self.benchmark_origin + item_end), 6
                                ),
                            })
                    if self.recognizer.is_endpoint(stream):
                        result = self.recognizer.get_result_all(stream)
                        self.commit_line(
                            speaker, state, result, "sherpa_native_endpoint",
                            item_end, lines, words,
                        )
                        self.recognizer.reset(stream)
                        state["hypothesis"] = ""
                        state["new_utterance"] = True
                        state["active_utterance"] = False
            for speaker, stream in self.streams.items():
                state = states.get(speaker)
                if state is None:
                    continue
                if not state["active_utterance"]:
                    continue
                stream.input_finished()
                self.decode(stream)
                result = self.recognizer.get_result_all(stream)
                self.commit_line(
                    speaker, state, result, "end_of_stream", state["last_end"], lines, words
                )
                endpoint_log.write(json.dumps({
                    "speaker": speaker.lower(),
                    "previous_audio_end_seconds": round(state["last_end"], 3),
                    "next_audio_start_seconds": round(state["last_end"], 3),
                    "injected_silence_seconds": 0.0,
                    "native_endpoint": False,
                    "boundary": "end_of_stream",
                }, ensure_ascii=False) + "\n")
                endpoint_log.flush()

    # Queue identified speech for ASR without blocking the router on decoding.
    def send(self, speaker: str, start_seconds: float, audio: np.ndarray) -> None:
        if len(audio):
            end_seconds = start_seconds + len(audio) / 16000
            self.queue.put((speaker, start_seconds, end_seconds, np.asarray(audio, dtype=np.float32), time.perf_counter()))

    # Flush every recognizer stream and finish the durable transcript log.
    def idle_barrier(self) -> threading.Event:
        completed = threading.Event()
        self.queue.put(("idle_barrier", completed))
        return completed

    # Flush every recognizer stream and finish the durable transcript log.
    def finish(self) -> None:
        self.queue.put(STOP)
        self.thread.join()


class MoonshineASR:
    """Bounded Moonshine v2 decoding for the live router.

    The supplied ONNX export has no encoder-cache input, unlike Nemotron's
    persistent Sherpa stream.  We therefore retain contiguous speaker audio
    until a fixed segment is available, decode that segment independently, and
    report the boundary-derived word times as approximate.
    """
    def __init__(self, args: argparse.Namespace, output: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.args, self.output = args, output
        model = args.moonshine_model
        suffix = ".ort" if (model / "encoder_model_int8.ort").is_file() else ".onnx"
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = args.asr_threads
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(model / f"encoder_model_int8{suffix}"), options, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(model / f"decoder_model_int8{suffix}"), options, providers=["CPUExecutionProvider"])
        self.decoder_past = ort.InferenceSession(str(model / f"decoder_with_past_model_int8{suffix}"), options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.segment_samples = round(args.moonshine_segment_seconds * 16000)
        self.past_inputs = {item.name for item in self.decoder_past.get_inputs()}
        self.queue = queue.Queue(maxsize=64)
        self.hypotheses_output = output / "transcript.jsonl"
        self.lines_output = output / "timestamped_lines.jsonl"
        self.words_output = output / "word_timestamps.jsonl"
        self.endpoint_output = output / "turn_endpointing.jsonl"
        self.benchmark_origin = None
        self.timing_events, self.detailed_events = [], []
        self.timing_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="moonshine", daemon=True)
        self.thread.start()

    def set_benchmark_origin(self, origin: float) -> None:
        self.benchmark_origin = origin

    def record_detailed(self, stage: str, speaker: str, started: float, finished: float, **details) -> None:
        if self.benchmark_origin is None:
            return
        event = {"stage": stage, "speaker": speaker.lower(),
                 "start_seconds": round(started - self.benchmark_origin, 6),
                 "end_seconds": round(finished - self.benchmark_origin, 6)}
        event.update(details)
        with self.timing_lock:
            self.detailed_events.append(event)

    def prewarm(self, speaker_count: int) -> None:
        # Loading occurs before the capture clock; do not add a synthetic
        # inference here because it would change the measured decoder state.
        return None

    def _cache(self, names, tensors):
        mapped = {}
        for name, tensor in zip(names, tensors):
            candidate = name.replace("present_", "past_", 1)
            if candidate in self.past_inputs:
                mapped[candidate] = tensor
            elif name + "_orig" in self.past_inputs:
                mapped[name + "_orig"] = tensor
            else:
                mapped[name] = tensor
        return mapped

    def decode(self, audio: np.ndarray) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        hidden = self.encoder.run(None, {"input_values": audio[None, :], "attention_mask": np.ones((1, len(audio)), dtype=np.int64)})[0]
        outputs = self.decoder.run(None, {"decoder_input_ids": np.array([[1]], dtype=np.int64), "encoder_hidden_states": hidden})
        token = int(np.argmax(outputs[0][0, -1]))
        tokens, cache = [token], self._cache([item.name for item in self.decoder.get_outputs()][1:], outputs[1:])
        while token != 2 and len(tokens) < self.args.moonshine_max_tokens:
            outputs = self.decoder_past.run(None, {"decoder_input_ids": np.array([[token]], dtype=np.int64), "encoder_hidden_states": hidden, **cache})
            token = int(np.argmax(outputs[0][0, -1]))
            tokens.append(token)
            cache = self._cache([item.name for item in self.decoder_past.get_outputs()][1:], outputs[1:])
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def _commit(self, speaker, start, end, audio, boundary, lines, words, hypotheses):
        started = time.perf_counter()
        text = self.decode(audio)
        finished = time.perf_counter()
        self.record_detailed("asr_speech_decode", speaker, started, finished,
                             source_start_seconds=round(start, 6), source_end_seconds=round(end, 6),
                             segment_seconds=round((end - start), 6))
        if self.benchmark_origin is not None:
            with self.timing_lock:
                self.timing_events.append({"speaker": speaker.lower(), "source_end_seconds": round(end, 6),
                    "decode_seconds": round(finished - started, 6), "shared_decode_seconds": round(finished - started, 6),
                    "batch_size": 1, "completion_lag_seconds": round(finished - (self.benchmark_origin + end), 6)})
        if not text:
            return
        line = {"speaker": speaker.lower(), "start_seconds": round(start, 3), "end_seconds": round(end, 3),
                "committed_seconds": round(end, 3), "boundary": boundary, "text": text}
        lines.write(json.dumps(line, ensure_ascii=False) + "\n"); lines.flush()
        tokens = text.split()
        duration = max(0.001, end - start)
        word_items = [{"word": word, "start_seconds": round(start + duration * i / len(tokens), 3),
                       "end_seconds": round(start + duration * (i + 1) / len(tokens), 3)} for i, word in enumerate(tokens)]
        words.write(json.dumps({"speaker": line["speaker"], "start_seconds": line["start_seconds"], "end_seconds": line["end_seconds"],
            "boundary": boundary, "timing_source": "moonshine_uniform_segment_approximation", "words": word_items}, ensure_ascii=False) + "\n"); words.flush()
        hypotheses.write(json.dumps({"speaker": speaker, "observed_seconds": round(end, 3), "hypothesis": text}, ensure_ascii=False) + "\n"); hypotheses.flush()
        print(f"[MOONSHINE {start:7.2f}-{end:7.2f}s] {speaker}: {text}", flush=True)

    def _run(self) -> None:
        buffers = {}
        with (self.hypotheses_output.open("w", encoding="utf-8") as hypotheses,
              self.lines_output.open("w", encoding="utf-8") as lines,
              self.words_output.open("w", encoding="utf-8") as words,
              self.endpoint_output.open("w", encoding="utf-8") as endpoint_log):
            while True:
                item = self.queue.get()
                if item is STOP:
                    break
                if item[0] == "idle_barrier":
                    item[1].set(); continue
                speaker, start, end, audio, enqueued = item
                dequeued = time.perf_counter()
                self.record_detailed("asr_queue_wait", speaker, enqueued, dequeued,
                                     source_start_seconds=round(start, 6), source_end_seconds=round(end, 6))
                prior = buffers.get(speaker)
                if prior is None or abs(start - prior["end"]) > 0.02:
                    if prior is not None and len(prior["audio"]):
                        self._commit(speaker, prior["start"], prior["end"], prior["audio"], "moonshine_gap_flush", lines, words, hypotheses)
                    prior = {"start": start, "end": start, "audio": np.zeros(0, dtype=np.float32)}
                    buffers[speaker] = prior
                prior["audio"] = np.concatenate((prior["audio"], np.asarray(audio, dtype=np.float32)))
                prior["end"] = end
                while len(prior["audio"]) >= self.segment_samples:
                    segment = prior["audio"][:self.segment_samples]
                    segment_end = prior["start"] + len(segment) / 16000
                    self._commit(speaker, prior["start"], segment_end, segment, "moonshine_10s_segment", lines, words, hypotheses)
                    prior["audio"] = prior["audio"][self.segment_samples:]
                    prior["start"] = segment_end
                endpoint_log.write(json.dumps({"speaker": speaker.lower(), "next_audio_start_seconds": round(end, 3),
                    "native_endpoint": False, "boundary": "moonshine_segment_buffer"}) + "\n"); endpoint_log.flush()
            for speaker, prior in buffers.items():
                if len(prior["audio"]):
                    self._commit(speaker, prior["start"], prior["end"], prior["audio"], "end_of_stream", lines, words, hypotheses)

    def send(self, speaker: str, start_seconds: float, audio: np.ndarray) -> None:
        if len(audio):
            self.queue.put((speaker, start_seconds, start_seconds + len(audio) / 16000, np.asarray(audio, dtype=np.float32), time.perf_counter()))

    def idle_barrier(self) -> threading.Event:
        completed = threading.Event(); self.queue.put(("idle_barrier", completed)); return completed

    def finish(self) -> None:
        self.queue.put(STOP); self.thread.join()


class ActivityRouter:
    # Convert stable LS-EEND frames into ordered solo or source-separated speaker audio.
    def __init__(self, mixture: np.ndarray, frame_rate: float, args: argparse.Namespace,
                 tracks: TrackManager, overlap: OverlapWorker, asr: StreamingASR,
                 benchmark_clean_tracks: list[np.ndarray] | None = None):
        self.mixture = mixture
        self.frame_rate = frame_rate
        self.args = args
        self.tracks = tracks
        self.overlap = overlap
        self.asr = asr
        self.probabilities = []
        self.cursor = 0
        self.available_seconds = 0.0
        self.overlaps = []
        self.timing_events = []
        self.benchmark_clean_tracks = benchmark_clean_tracks

    # Add newly stable frame probabilities and note the furthest audio currently available.
    def add(self, events: list[tuple[float, np.ndarray]], available_seconds: float) -> None:
        for timestamp, probability in events:
            expected = len(self.probabilities) / self.frame_rate
            if abs(timestamp - expected) > 0.051:
                raise RuntimeError(f"Non-contiguous LS-EEND output: expected {expected:.3f}, got {timestamp:.3f}")
            self.probabilities.append(probability)
        self.available_seconds = max(self.available_seconds, available_seconds)

    # Extend the captured mixture before routing newly stable diarisation frames.
    def append_audio(self, audio: np.ndarray) -> None:
        block = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(block):
            self.mixture = np.concatenate((self.mixture, block))

    # Return binary activity for every stable LS-EEND frame observed so far.
    def activity(self) -> np.ndarray:
        if not self.probabilities:
            return np.zeros((0, 0), dtype=bool)
        return activity_matrix(np.stack(self.probabilities), self.args.activity_threshold)

    # Keep the three strongest active LS-EEND slots when a transient extra slot appears.
    def active_slots_at(self, activity: np.ndarray, frame: int) -> tuple[int, ...]:
        slots = np.flatnonzero(activity[frame])
        if len(slots) <= 3:
            return tuple(slots)
        scores = self.probabilities[frame][slots]
        strongest = slots[np.argsort(scores)[-3:]]
        return tuple(sorted(strongest))

    # Send one assigned speaker slice to both its track writer and persistent ASR cache.
    def deliver(self, speaker: str, start_sample: int, audio: np.ndarray) -> None:
        self.tracks.send(speaker, start_sample, audio)
        self.asr.send(speaker, start_sample / 16000, audio)

    # Split a masked overlap result into true speech intervals before writing or ASR.
    def deliver_masked(self, speaker: str, start_sample: int, audio: np.ndarray, mask: np.ndarray) -> None:
        active = np.asarray(mask[: len(audio)], dtype=bool)
        run_start = None
        for index in range(len(active) + 1):
            is_active = index < len(active) and active[index]
            if is_active and run_start is None:
                run_start = index
            if not is_active and run_start is not None:
                self.deliver(speaker, start_sample + run_start, audio[run_start:index])
                run_start = None

    # Run the selected forced routing mode on every released block.
    def process_always_separate(self, available_frames: int) -> None:
        speaker_count = self.args.always_separate_speakers
        block_frames = max(1, round(self.args.chunk_seconds * self.frame_rate))
        while self.cursor < available_frames:
            end = min(available_frames, self.cursor + block_frames)
            core_start = round(self.cursor / self.frame_rate * 16000)
            core_end = min(len(self.mixture), round(end / self.frame_rate * 16000))
            if core_end <= core_start:
                self.cursor = end
                continue

            speakers = [f"SPEAKER_{index:02d}" for index in range(speaker_count)]
            for speaker in speakers:
                self.tracks.ensure(speaker)

            # A forced one-speaker run does not need source separation. Keep
            # the original mixture as the single speaker track.
            if speaker_count == 1:
                self.deliver(speakers[0], core_start, self.mixture[core_start:core_end])
                self.overlaps.append({
                    "start_seconds": round(core_start / 16000, 3),
                    "end_seconds": round(core_end / 16000, 3),
                    "context_start_seconds": round(core_start / 16000, 3),
                    "context_end_seconds": round(core_end / 16000, 3),
                    "speakers": speakers,
                    "assignments": [],
                    "separation_seconds": 0.0,
                    "diagnostics": {"mode": "single_track_bypass"},
                    "forced_continuous_separation": False,
                })
                self.cursor = end
                continue

            benchmark_clean_sources = None
            if self.benchmark_clean_tracks is not None:
                benchmark_clean_sources = [
                    track[core_start:core_end] for track in self.benchmark_clean_tracks
                ]
            result = self.overlap.resolve({
                "context_audio": self.mixture[core_start:core_end],
                "context_start": core_start,
                "core_start": core_start,
                "core_samples": core_end - core_start,
                "speakers": speakers,
                "profiles": self.tracks.snapshot(speakers),
                "benchmark_clean_sources": benchmark_clean_sources,
            })
            self.timing_events.extend(result["timings"])
            for speaker_index, speaker in enumerate(speakers):
                separated = result["cropped"].get(speaker)
                if separated is None:
                    continue
                # The saturation benchmark executes real separation first, then
                # uses aligned dense reference speech for downstream CAM++/ASR.
                # Deployment runs leave benchmark_clean_tracks unset and consume
                # the actual separator output.
                downstream = separated
                if self.benchmark_clean_tracks is not None:
                    downstream = self.benchmark_clean_tracks[speaker_index][core_start:core_end]
                self.deliver(speaker, core_start, downstream)
            self.overlaps.append({
                "start_seconds": round(core_start / 16000, 3),
                "end_seconds": round(core_end / 16000, 3),
                "context_start_seconds": round(core_start / 16000, 3),
                "context_end_seconds": round(core_end / 16000, 3),
                "speakers": speakers,
                "assignments": result["assignments"],
                "separation_seconds": round(result["seconds"], 6),
                "diagnostics": result["diagnostics"],
                "forced_continuous_separation": True,
                "benchmark_clean_downstream": self.benchmark_clean_tracks is not None,
            })
            self.cursor = end

    # Advance through only the frames whose required context audio has already arrived.
    def process_ready(self, final: bool = False) -> None:
        activity = self.activity()
        available_frames = int(self.available_seconds * self.frame_rate)
        if self.args.always_separate_speakers is not None:
            self.process_always_separate(available_frames if not final else len(activity))
            return
        if not len(activity):
            return
        ready = len(activity) if final else min(len(activity), available_frames)
        while self.cursor < ready:
            slots = self.active_slots_at(activity, self.cursor)
            if len(slots) == 0:
                self.cursor += 1
                continue
            if len(slots) == 1:
                end = self.cursor + 1
                while end < ready and self.active_slots_at(activity, end) == slots:
                    end += 1
                start_sample = round(self.cursor / self.frame_rate * 16000)
                end_sample = min(len(self.mixture), round(end / self.frame_rate * 16000))
                self.deliver(f"SPEAKER_{slots[0]:02d}", start_sample, self.mixture[start_sample:end_sample])
                self.cursor = end
                continue
            end = self.cursor + 1
            limit = min(ready, self.cursor + round(self.args.maximum_overlap_seconds * self.frame_rate))
            while end < limit and self.active_slots_at(activity, end) == slots:
                end += 1
            if not final and end == ready and end < self.cursor + round(self.args.maximum_overlap_seconds * self.frame_rate):
                return
            core_start = round(self.cursor / self.frame_rate * 16000)
            core_end = min(len(self.mixture), round(end / self.frame_rate * 16000))
            collar = round(self.args.overlap_collar_seconds * 16000)
            context_start = max(0, core_start - collar)
            context_end = min(len(self.mixture), core_end + collar)
            if not final and context_end > round(self.available_seconds * 16000):
                return
            speakers = [f"SPEAKER_{slot:02d}" for slot in slots]
            for speaker in speakers:
                self.tracks.ensure(speaker)
            result = self.overlap.resolve({
                "context_audio": self.mixture[context_start:context_end],
                "context_start": context_start,
                "core_start": core_start,
                "core_samples": core_end - core_start,
                "speakers": speakers,
                "profiles": self.tracks.snapshot(speakers),
            })
            self.timing_events.extend(result["timings"])
            masks = sample_masks(activity[self.cursor:end], list(slots), self.frame_rate, 16000, core_end - core_start)
            for speaker, separated in result["cropped"].items():
                self.deliver_masked(speaker, core_start, separated, masks[speaker])
            self.overlaps.append({
                "start_seconds": round(core_start / 16000, 3),
                "end_seconds": round(core_end / 16000, 3),
                "context_start_seconds": round(context_start / 16000, 3),
                "context_end_seconds": round(context_end / 16000, 3),
                "speakers": speakers,
                "assignments": result["assignments"],
                "separation_seconds": round(result["seconds"], 6),
                "diagnostics": result["diagnostics"],
            })
            self.cursor = end


# Read five-second source chunks, pass them through a persistent LS-EEND session, and queue results.
def diarize_stream(engine: OnlineLSEEND, mixture: np.ndarray, args: argparse.Namespace, results: queue.Queue, origin: float, pacer=None) -> None:
    try:
        chunk_samples = round(args.chunk_seconds * 16000)
        for index, start in enumerate(range(0, len(mixture), chunk_samples)):
            end = min(start + chunk_samples, len(mixture))
            if pacer is not None:
                pacer.wait_for_chunk(index, end / 16000)
            elif args.pace:
                delay = origin + end / 16000 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            started = time.perf_counter()
            events = engine.accept(resample_audio(mixture[start:end], 16000, 8000))
            finished = time.perf_counter()
            results.put((events, end / 16000, False, index, origin, started, finished))
        started = time.perf_counter()
        events = engine.finish()
        finished = time.perf_counter()
        results.put((events, len(mixture) / 16000, True, index, origin, started, finished))
    except Exception as error:
        results.put((error, 0.0, True, -1, 0.0, 0.0, 0.0))


# Parse the deployment command line and default every model to raw downloaded provenance.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live LS-EEND, conditional separation, CAM++, and ASR pipeline")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--no-agc", action="store_true")
    parser.add_argument("--pace", action="store_true", help="Release the file at real-time five-second chunk boundaries.")
    parser.add_argument("--semi-paced", action="store_true", help="Orchestrator-controlled removal of fully idle arrival waits.")
    parser.add_argument("--benchmark-warmup-seconds", type=float, default=0.0)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--detailed-timing-log", type=Path)
    parser.add_argument("--warmup-models", action="store_true",
                        help="Execute a disposable LS-EEND step before the capture clock.")
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--activity-threshold", type=float, default=0.4)
    parser.add_argument("--overlap-collar-seconds", type=float, default=0.0)
    parser.add_argument("--maximum-overlap-seconds", type=float, default=3.0)
    parser.add_argument("--always-separate-speakers", type=int, choices=(1, 2, 3),
                        help="Force one direct track or the selected separator on every released block.")
    parser.add_argument("--benchmark-clean-track", type=Path, action="append",
                        help="Benchmark-only aligned clean track used after forced separation.")
    parser.add_argument("--minimum-similarity", type=float, default=0.2)
    parser.add_argument("--profile-update-seconds", type=float, default=1.0)
    parser.add_argument("--turn-hangover-seconds", type=float, default=1.2)
    parser.add_argument("--turn-fallback-silence-seconds", type=float, default=2.4)
    parser.add_argument("--turn-max-utterance-seconds", type=float, default=30.0)
    parser.add_argument("--lseend-threads", type=int, default=1)
    parser.add_argument("--separation-threads", type=int, default=2)
    parser.add_argument("--speaker-encoder-threads", "--campplus-threads", dest="speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--overlap-speaker-encoder-threads", "--overlap-cam-threads", dest="overlap_speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--asr-threads", type=int, default=3)
    parser.add_argument("--asr-backend", choices=("moonshine", "nemotron"), default="moonshine")
    parser.add_argument("--moonshine-model", type=Path,
                        default=RAW_MODELS / "moonshine_streaming_medium_onnx")
    parser.add_argument("--moonshine-segment-seconds", type=float, default=10.0)
    parser.add_argument("--moonshine-max-tokens", type=int, default=256)
    parser.add_argument("--separator-backend", choices=("convtasnet", "sandglasset"), default="convtasnet")
    parser.add_argument("--asr-batch-ready", action="store_true",
                        help="Batch aligned ready speaker streams in one Sherpa decode call.")
    parser.add_argument("--speaker-encoder-batch-overlap", "--cam-batch-overlap", dest="speaker_encoder_batch_overlap", action="store_true",
                        help="Batch equal-length overlap speaker-encoder inputs in one ONNX call.")
    parser.add_argument("--lseend-model", type=Path, default=RAW_MODELS / "lseend" / "DIHARD III" / "ls_eend_dih3_step.onnx")
    parser.add_argument("--lseend-metadata", type=Path, default=RAW_MODELS / "lseend" / "DIHARD III" / "ls_eend_dih3_step.json")
    parser.add_argument("--convtasnet-2", type=Path, default=RAW_MODELS / "convtasnet_2" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-3", type=Path, default=RAW_MODELS / "convtasnet_3" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-2-onnx", type=Path, default=ONNX_CACHE / "convtasnet_2_raw_export.onnx")
    parser.add_argument("--convtasnet-3-onnx", type=Path, default=ONNX_CACHE / "convtasnet_3_raw_export.onnx")
    parser.add_argument(
        "--sandglasset-2-onnx", type=Path,
        default=PROJECT_ROOT / "models" / "deployment" / "sandglasset_2spk_5s_ort_optimized.onnx",
    )
    parser.add_argument(
        "--sandglasset-3-onnx", type=Path,
        default=PROJECT_ROOT / "models" / "deployment" / "sandglasset_3spk_5s_ort_optimized.onnx",
    )
    parser.add_argument("--speaker-encoder-model", "--campplus-model", dest="speaker_encoder_model", type=Path,
                        default=PROJECT_ROOT / "models" / "deployment" / "redimnet_b1_5s_ort_enable_all.onnx")
    parser.add_argument("--nemotron-encoder", type=Path, default=RAW_MODELS / "nemotron_1120ms_int8" / "encoder.int8.onnx")
    parser.add_argument("--nemotron-decoder", type=Path, default=RAW_MODELS / "nemotron_1120ms_int8" / "decoder.int8.onnx")
    parser.add_argument("--nemotron-joiner", type=Path, default=RAW_MODELS / "nemotron_1120ms_int8" / "joiner.int8.onnx")
    parser.add_argument("--nemotron-tokens", type=Path, default=RAW_MODELS / "nemotron_1120ms_int8" / "tokens.txt")
    args = parser.parse_args()
    if args.output is None:
        name = args.name or args.input.stem
        if Path(name).name != name:
            parser.error("--name must be one folder name")
        args.output = OUTPUT_ROOT / name
    for value in (args.chunk_seconds, args.maximum_overlap_seconds, args.profile_update_seconds, args.turn_hangover_seconds, args.turn_fallback_silence_seconds, args.turn_max_utterance_seconds, args.moonshine_segment_seconds):
        if value <= 0:
            parser.error("chunk, maximum overlap, and profile/turn values must be positive")
    if args.overlap_collar_seconds < 0:
        parser.error("--overlap-collar-seconds must not be negative")
    if args.benchmark_warmup_seconds < 0:
        parser.error("--benchmark-warmup-seconds must not be negative")
    if args.turn_fallback_silence_seconds < args.turn_hangover_seconds:
        parser.error("--turn-fallback-silence-seconds must be at least --turn-hangover-seconds")
    for name in ("lseend_threads", "separation_threads", "speaker_encoder_threads", "overlap_speaker_encoder_threads", "asr_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    # A raw PyTorch checkpoint is needed only when its optimized ONNX graph is
    # unavailable; the deployment path uses the ONNX graph directly.
    required = [
        args.input,
        args.lseend_model,
        args.lseend_metadata,
        args.speaker_encoder_model,
    ]
    if args.asr_backend == "nemotron":
        required.extend([args.nemotron_encoder, args.nemotron_decoder, args.nemotron_joiner, args.nemotron_tokens])
    else:
        required.extend([args.moonshine_model / "encoder_model_int8.ort", args.moonshine_model / "decoder_model_int8.ort",
                         args.moonshine_model / "decoder_with_past_model_int8.ort", args.moonshine_model / "tokenizer.json"])
    if args.separator_backend == "sandglasset":
        required.extend([args.sandglasset_2_onnx, args.sandglasset_3_onnx])
    else:
        if not args.convtasnet_2_onnx.exists():
            required.append(args.convtasnet_2)
        if not args.convtasnet_3_onnx.exists():
            required.append(args.convtasnet_3)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("Missing required files:\n" + "\n".join(missing))
    if args.benchmark_clean_track:
        if args.always_separate_speakers is None:
            parser.error("--benchmark-clean-track requires --always-separate-speakers")
        if len(args.benchmark_clean_track) != args.always_separate_speakers:
            parser.error("Provide one --benchmark-clean-track per forced speaker")
        absent = [str(path) for path in args.benchmark_clean_track if not path.is_file()]
        if absent:
            parser.error("Missing benchmark clean tracks:\n" + "\n".join(absent))
    return args


# Run the complete streaming path and save the exact routing and output artifacts.
def main() -> None:
    args = parse_args()
    lock_external_thread_pools()
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for input conversion and MKV track writing")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tracks").mkdir(exist_ok=True)
    mixture = load_audio(args.input, args.output / "normalised_input.wav", 16000)
    if not args.no_agc:
        mixture = apply_agc(mixture, 16000)
        sf.write(args.output / "normalised_input.wav", mixture, 16000, subtype="FLOAT")
    benchmark_clean_tracks = None
    if args.benchmark_clean_track:
        benchmark_clean_tracks = []
        for index, path in enumerate(args.benchmark_clean_track):
            track = load_audio(path, args.output / f"benchmark_clean_{index:02d}.wav", 16000)
            if len(track) != len(mixture):
                raise ValueError("Benchmark clean tracks must match the mixture duration exactly")
            benchmark_clean_tracks.append(track)
    metadata = load_metadata(args.lseend_metadata)
    tracks = TrackManager(args.output, len(mixture), args.speaker_encoder_model, args.speaker_encoder_threads, args.profile_update_seconds)
    overlap = OverlapWorker(args)
    asr = MoonshineASR(args, args.output) if args.asr_backend == "moonshine" else StreamingASR(args, args.output)
    lseend = OnlineLSEEND(
        args.lseend_model, metadata, args.lseend_threads,
        onnx_optimization="all", memory_pattern=True, allow_spinning=False,
    )
    if args.warmup_models:
        lseend.warmup()
    # Remove first-use setup from the measured live path. LS-EEND exposes ten
    # speaker slots in this checkpoint, although at most three may be active
    # simultaneously in the deployment contract.
    max_speakers = int(metadata["real_output_dim"])
    prewarm_speakers = args.always_separate_speakers or max_speakers
    tracks.prewarm(prewarm_speakers)
    overlap.wait_until_ready()
    asr.prewarm(prewarm_speakers)
    benchmark_origin = time.perf_counter()
    pacer = SemiPacer(args.output, benchmark_origin) if args.semi_paced else None
    (args.output / "source_clock.json").write_text(json.dumps({
        "started_monotonic": benchmark_origin,
        "started_unix": time.time(),
    }) + "\n", encoding="utf-8")
    asr.set_benchmark_origin(benchmark_origin)
    router = ActivityRouter(
        mixture, float(metadata["frame_hz"]), args, tracks, overlap, asr,
        benchmark_clean_tracks,
    )
    decisions = queue.Queue(maxsize=8)
    worker = threading.Thread(
        target=diarize_stream,
        args=(lseend, mixture, args, decisions, benchmark_origin, pacer),
        name="lseend",
        daemon=True,
    )
    worker.start()
    benchmark_events = []
    detailed_events = []
    final = False
    while not final:
        events, available, final, chunk, origin, lseend_started, lseend_finished = decisions.get()
        if isinstance(events, Exception):
            raise RuntimeError("LS-EEND worker failed") from events
        routing_started = time.perf_counter()
        router.add(events, available)
        router.process_ready(final=final)
        routing_finished = time.perf_counter()
        if pacer is not None and not final:
            checks = tracks.idle_barriers()
            checks.append(asr.idle_barrier().is_set)
            pacer.mark_routed(chunk, checks)
        if args.detailed_timing_log is not None:
            source_start = min(
                max(0.0, chunk * args.chunk_seconds),
                available,
            )
            detailed_events.extend([
                {
                    "stage": "source_capture",
                    "start_seconds": round(source_start, 6),
                    "end_seconds": round(available, 6),
                    "chunk": chunk,
                },
                {
                    "stage": "lseend",
                    "start_seconds": round(lseend_started - origin, 6),
                    "end_seconds": round(lseend_finished - origin, 6),
                    "chunk": chunk,
                },
                {
                    "stage": "router",
                    "start_seconds": round(routing_started - origin, 6),
                    "end_seconds": round(routing_finished - origin, 6),
                    "chunk": chunk,
                },
            ])
        if args.benchmark_report is not None:
            benchmark_events.append({
                "chunk": chunk,
                "source_end_seconds": round(available, 6),
                "lseend_seconds": round(lseend_finished - lseend_started, 6),
                "routing_seconds": round(routing_finished - routing_started, 6),
                "routing_completion_lag_seconds": round(routing_finished - (origin + available), 6),
                "final_flush": final,
            })
        tracks.drain_profiles()
    worker.join()
    overlap.finish()
    asr.finish()
    tracks.finish()
    activity = router.activity()
    slots = [index for index in range(activity.shape[1]) if activity[:, index].any()] if activity.size else []
    (args.output / "diarization.json").write_text(json.dumps(build_segments(activity, slots, float(metadata["frame_hz"])), indent=2) + "\n", encoding="utf-8")
    (args.output / "overlap_assignments.json").write_text(json.dumps(router.overlaps, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "input": str(args.input), "output": str(args.output), "sample_rate": 16000,
        "chunk_seconds": args.chunk_seconds, "turn_hangover_seconds": args.turn_hangover_seconds, "turn_fallback_silence_seconds": args.turn_fallback_silence_seconds, "turn_max_utterance_seconds": args.turn_max_utterance_seconds, "overlap_collar_seconds": args.overlap_collar_seconds,
        "maximum_overlap_seconds": args.maximum_overlap_seconds, "paced": args.pace,
        "evaluation_mode": "semi-paced" if args.semi_paced else ("paced" if args.pace else "unpaced"),
        "always_separate_speakers": args.always_separate_speakers,
        "benchmark_clean_tracks": [str(path) for path in args.benchmark_clean_track] if args.benchmark_clean_track else [],
        "prewarmed_speaker_slots": prewarm_speakers,
        "separator_backend": args.separator_backend,
        "models": {
            "lseend": str(args.lseend_model),
            "convtasnet_2": str(args.convtasnet_2),
            "convtasnet_3": str(args.convtasnet_3),
            "sandglasset_2_onnx": str(args.sandglasset_2_onnx),
            "sandglasset_3_onnx": str(args.sandglasset_3_onnx),
            "speaker_encoder": str(args.speaker_encoder_model),
            "asr_backend": args.asr_backend,
            "nemotron_encoder": str(args.nemotron_encoder),
            "moonshine_model": str(args.moonshine_model),
        },
        "threads": {"lseend": args.lseend_threads, "separation": args.separation_threads, "track_speaker_encoder": args.speaker_encoder_threads, "overlap_speaker_encoder": args.overlap_speaker_encoder_threads, "asr": args.asr_threads},
        "batching": {"asr_ready": args.asr_batch_ready, "overlap_speaker_encoder": args.speaker_encoder_batch_overlap},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.benchmark_report is not None:
        measured = [
            event for event in benchmark_events
            if not event["final_flush"]
            and event["source_end_seconds"] > args.benchmark_warmup_seconds
        ]
        report = {
            "runner": str(Path(__file__).resolve()),
            "warmup_seconds": args.benchmark_warmup_seconds,
            "chunk_seconds": args.chunk_seconds,
            "measured_chunks": measured,
            "lseend": timing_summary([event["lseend_seconds"] for event in measured]),
            "routing": timing_summary([event["routing_seconds"] for event in measured]),
            "routing_completion_lag": timing_summary([
                event["routing_completion_lag_seconds"] for event in measured
            ]),
            "asr": timing_summary([
                event["decode_seconds"]
                for event in asr.timing_events
                if event["source_end_seconds"] > args.benchmark_warmup_seconds
            ]),
            "asr_completion_lag": timing_summary([
                event["completion_lag_seconds"]
                for event in asr.timing_events
                if event["source_end_seconds"] > args.benchmark_warmup_seconds
            ]),
            "asr_events": asr.timing_events,
            "overlap_regions": len(router.overlaps),
            "track_count": len(list((args.output / "tracks").glob("*.mkv"))),
        }
        args.benchmark_report.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.detailed_timing_log is not None:
        for event in router.timing_events + tracks.timing_events:
            event = dict(event)
            event["start_seconds"] = round(event.pop("started") - benchmark_origin, 6)
            event["end_seconds"] = round(event.pop("finished") - benchmark_origin, 6)
            detailed_events.append(event)
        detailed_events.extend(asr.detailed_events)
        trace = {
            "origin": "pipeline_start",
            "events": sorted(detailed_events, key=lambda event: event["start_seconds"]),
        }
        args.detailed_timing_log.parent.mkdir(parents=True, exist_ok=True)
        args.detailed_timing_log.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(f"wrote deployment output to {args.output}", flush=True)


if __name__ == "__main__":
    main()
