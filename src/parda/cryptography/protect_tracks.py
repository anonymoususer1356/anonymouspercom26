#!/usr/bin/env python3
# Encrypt separated audio tracks and seal ReDimNet2-B6 voice fingerprints for the TTP.

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from track_crypto import (
    FORMAT_VERSION,
    encrypt_bytes,
    encrypt_file,
    fingerprint_aad,
    load_ttp_public_key_pem,
    load_ttp_public_key,
    pem_from_env,
    public_key_fingerprint,
    track_aad,
    wrap_key,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SEPARATION = PROJECT_ROOT / "src" / "parda" / "audio"
sys.path.insert(0, str(SOURCE_SEPARATION))

from audio import load_audio
from speaker_embeddings import ReDimNetEncoder, concatenate_intervals


# Load the LS-EEND thresholded speaker segments emitted by the source pipeline.
def load_diarization(path: Path) -> dict[str, list[dict]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("diarization.json must contain a list of LS-EEND segments")
    tracks: dict[str, list[dict]] = {}
    for record in records:
        speaker = str(record.get("speaker", ""))
        start = float(record["start"])
        end = float(record["end"])
        if not speaker or end <= start:
            raise ValueError("diarization.json contains an invalid activity segment")
        tracks.setdefault(speaker, []).append(
            {"start_seconds": start, "end_seconds": end}
        )
    return tracks


# Load one MKV and retain only its LS-EEND-active speaker intervals.
def load_track_speech(track: Path, regions: list[dict]) -> tuple[object, dict]:
    with tempfile.TemporaryDirectory(prefix="voice-fingerprint-") as temporary:
        audio = load_audio(track, Path(temporary) / "track_16k.wav", 16000)
    intervals = [
        (round(region["start_seconds"] * 16000), round(region["end_seconds"] * 16000))
        for region in regions
    ]
    speech = concatenate_intervals(audio, intervals)
    return speech, {
        "source_audio_seconds": round(len(audio) / 16000, 3),
        "speech_seconds": round(len(speech) / 16000, 3),
        "speech_region_count": len(regions),
        "speech_regions": regions,
    }


# Find only actual speaker-track files, not prior embeddings or encrypted artifacts.
def find_tracks(tracks_dir: Path) -> list[Path]:
    tracks = sorted(path for path in tracks_dir.glob("*.mkv") if path.is_file())
    if not tracks:
        raise FileNotFoundError(f"No .mkv speaker tracks found in {tracks_dir}")
    names = [track.stem for track in tracks]
    if len(set(names)) != len(names):
        raise ValueError("Speaker track names must be unique")
    return tracks


# Copy only privacy-safe, word-timestamped text to the device package.
def copy_device_artifacts(source_output: Path, device: Path) -> list[dict]:
    stale_names = (
        "anonymised_transcript.txt",
        "anonymised_timestamped_lines.jsonl",
        "turn_endpointing.jsonl",
        "word_timestamps.jsonl",
    )
    for name in stale_names:
        (device / name).unlink(missing_ok=True)
    artifacts = []
    for name in ("anonymised_word_timestamps.jsonl",):
        source = source_output / name
        if not source.is_file():
            raise FileNotFoundError(f"Source output is missing required device artifact: {source}")
        destination = device / name
        shutil.copy2(source, destination)
        artifacts.append({"file": name, "bytes": destination.stat().st_size})
    return artifacts


# Batch the post-recording B6 work as soon as tracks are complete, independently of SLM draining.
def prepare_fingerprints(source_output: Path, encoder: ReDimNetEncoder) -> list[tuple[Path, dict, object]]:
    tracks = find_tracks(source_output / "tracks")
    diarization = load_diarization(source_output / "diarization.json")
    prepared = []
    for track in tracks:
        if track.stem not in diarization:
            raise ValueError(f"No LS-EEND activity segments found for {track.stem}")
        print(f"loading active speech from {track.name}", flush=True)
        speech, fingerprint = load_track_speech(track, diarization[track.stem])
        prepared.append((track, fingerprint, speech))
    print(f"batching ReDimNet2-B6 fingerprints for {len(prepared)} completed tracks", flush=True)
    started = time.perf_counter()
    embeddings = encoder.embed_long_audio_many([item[2] for item in prepared], 16000)
    elapsed = time.perf_counter() - started
    for (_, fingerprint, _), embedding in zip(prepared, embeddings):
        fingerprint["embedding_seconds"] = round(elapsed / max(1, len(prepared)), 6)
        fingerprint["embedding_dimension"] = int(len(embedding)) if embedding is not None else 0
        fingerprint["voice_fingerprint"] = embedding.tolist() if embedding is not None else None
    return prepared


# Encrypt every source-output track and seal final ReDimNet2-B6 fingerprints for the TTP.
def run(args: argparse.Namespace, encoder: ReDimNetEncoder | None = None, prepared=None) -> None:
    tracks = find_tracks(args.source_output / "tracks")
    diarization_path = args.source_output / "diarization.json"
    diarization = load_diarization(diarization_path)
    public_key = (
        load_ttp_public_key(args.ttp_public_key)
        if args.ttp_public_key is not None
        else load_ttp_public_key_pem(pem_from_env(args.env_file, "TTP_RSA_PUBLIC_KEY_B64"))
    )
    output = args.output
    device = output / "device"
    server = output / "server"
    encrypted_dir = device / "encrypted_tracks"
    encrypted_dir.mkdir(parents=True, exist_ok=True)
    server.mkdir(parents=True, exist_ok=True)
    if encoder is None:
        encoder = ReDimNetEncoder(
            args.fingerprint_model,
            cpu_threads=args.fingerprint_threads,
            onnx_optimization="all",
            memory_pattern=False,
            allow_spinning=False,
        )
    sealed_tracks = []
    fingerprint_keys = []
    model_sha256 = hashlib.sha256(args.fingerprint_model.read_bytes()).hexdigest()
    source_fingerprints = prepared if prepared is not None else prepare_fingerprints(args.source_output, encoder)
    for track, fingerprint, _ in source_fingerprints:
        track_id = track.stem
        voice_fingerprint = fingerprint.pop("voice_fingerprint")
        fingerprint_key = os.urandom(32)
        fingerprint_payload = json.dumps(
            {
                "track_id": track_id,
                "embedding_dimension": fingerprint["embedding_dimension"],
                "voice_fingerprint": voice_fingerprint,
                "trace": fingerprint,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint_keys.append(
            {
                "track_id": track_id,
                "source_file": track.name,
                "wrapped_key_b64": wrap_key(fingerprint_key, public_key),
                "fingerprint": encrypt_bytes(
                    fingerprint_payload,
                    fingerprint_key,
                    fingerprint_aad(track_id, model_sha256),
                ),
            }
        )
        key = os.urandom(32)
        destination = encrypted_dir / f"{track.name}.aesgcm"
        print(f"encrypting {track.name}", flush=True)
        details = encrypt_file(track, destination, key, track_aad(track_id, track.name))
        sealed_tracks.append({
            "track_id": track_id,
            "source_file": track.name,
            "encrypted_file": str(Path("..") / destination.relative_to(output)),
            "wrapped_key_b64": wrap_key(key, public_key),
            **details,
        })
    device_artifacts = copy_device_artifacts(args.source_output, device)
    write_json(server / "encrypted_voice_fingerprints.json", {
        "format": "encrypted-voice-fingerprint-v1",
        "fingerprint_model": args.fingerprint_model.name,
        "fingerprint_model_sha256": model_sha256,
        "speech_region_source": "source-output/diarization.json (LS-EEND thresholded activity)",
        "key_wrap_algorithm": "RSA-OAEP-SHA256",
        "tracks": fingerprint_keys,
    })
    write_json(server / "sealed_track_keys.json", {
        "format": FORMAT_VERSION,
        "key_wrap_algorithm": "RSA-OAEP-SHA256",
        "ttp_public_key_sha256": public_key_fingerprint(public_key),
        "tracks": sealed_tracks,
    })
    write_json(device / "device_manifest.json", {
        "format": "voice-track-device-package-v1",
        "encrypted_tracks": [item["encrypted_file"].removeprefix("../device/") for item in sealed_tracks],
        "unencrypted_artifacts": device_artifacts,
    }, mode=0o644)
    print(f"wrote device package: {device}", flush=True)
    print(f"wrote TTP package: {server}", flush=True)


# Parse the encryption and final fingerprint settings required for a reproducible protection pass.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypt separated MKV tracks and create TTP-sealed ReDimNet2-B6 voice fingerprints."
    )
    parser.add_argument("--source-output", type=Path, required=True,
                        help="One source-pipeline output directory containing tracks/ and diarization.json.")
    parser.add_argument("--output", type=Path,
                        help="Defaults to Outputs/Output Bundle/<source output name>.")
    parser.add_argument("--ttp-public-key", type=Path,
                        help="Optional PEM path; otherwise use TTP_RSA_PUBLIC_KEY_B64 from --env-file.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--fingerprint-model", type=Path,
        default=PROJECT_ROOT / "models" / "deployment" / "redimnet2_b6_5s_ort_enable_all.onnx",
    )
    parser.add_argument("--fingerprint-threads", type=int, default=4)
    args = parser.parse_args()
    if args.fingerprint_threads < 1:
        parser.error("--fingerprint-threads must be positive")
    if not args.source_output.is_dir():
        parser.error("--source-output must exist")
    if args.ttp_public_key is not None and not args.ttp_public_key.is_file():
        parser.error("--ttp-public-key must exist when supplied")
    if args.ttp_public_key is None and not args.env_file.is_file():
        parser.error("Provide --ttp-public-key or an --env-file containing TTP_RSA_PUBLIC_KEY_B64")
    if not (args.source_output / "diarization.json").is_file():
        parser.error("Source output is missing diarization.json")
    if not args.fingerprint_model.is_file():
        parser.error(f"ReDimNet2-B6 fingerprint model is missing: {args.fingerprint_model}")
    if args.output is None:
        args.output = PROJECT_ROOT / "outputs" / "bundles" / args.source_output.name
    return args


if __name__ == "__main__":
    run(parse_args())
