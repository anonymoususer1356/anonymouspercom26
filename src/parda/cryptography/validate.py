#!/usr/bin/env python3
# Validate the full protect -> TTP release -> restore path against real separated tracks.

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from track_crypto import b64decode, decrypt_file


SCRIPT_DIR = Path(__file__).resolve().parent


# Hash a file in bounded memory for an exact source-versus-restored comparison.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# Run a child command and fail with its exact command line if validation breaks.
def call(command: list[str]) -> None:
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(command, check=True)


# Verify that a modified ciphertext fails GCM authentication before the normal restore.
def verify_tamper_rejected(protected: Path, release: Path, track_id: str,
                           output: Path) -> None:
    server = protected / "server"
    manifest = json.loads((server / "sealed_track_keys.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["tracks"] if item["track_id"] == track_id)
    release_data = json.loads(release.read_text(encoding="utf-8"))
    key_record = next(item for item in release_data["approved_tracks"] if item["track_id"] == track_id)
    encrypted = server / record["encrypted_file"]
    tampered = output / "tampered.aesgcm"
    shutil.copyfile(encrypted, tampered)
    with tampered.open("r+b") as handle:
        handle.seek(0)
        byte = handle.read(1)
        handle.seek(0)
        handle.write(bytes([byte[0] ^ 0x01]))
    try:
        decrypt_file(tampered, output / "tampered-restored.mkv", b64decode(key_record["key_b64"]), record)
    except Exception:
        print("PASS: modified ciphertext was rejected by AES-GCM authentication.")
        return
    raise RuntimeError("Modified ciphertext was incorrectly accepted")


# Exercise one real source output with an ephemeral RSA key and verify byte-exact restoration.
def run(args: argparse.Namespace) -> None:
    tracks = sorted((args.source_output / "tracks").glob("*.mkv"))
    if not tracks:
        raise FileNotFoundError(f"No MKV tracks found in {args.source_output / 'tracks'}")
    track = tracks[0]
    work = args.output
    if work.exists():
        raise FileExistsError(
            f"Refusing to overwrite validation output: {work}. Choose a new empty path."
        )
    work.mkdir(parents=True)
    private_key = work / "ttp.private.pem"
    public_key = work / "ttp.public.pem"
    protected = work / "protected"
    release = work / "consent_release.json"
    restored = work / "restored"
    call([sys.executable, str(SCRIPT_DIR / "generate_ttp_keypair.py"),
          "--private-key", str(private_key), "--public-key", str(public_key)])
    call([sys.executable, str(SCRIPT_DIR / "protect_tracks.py"),
          "--source-output", str(args.source_output), "--output", str(protected),
          "--ttp-public-key", str(public_key), "--campplus-model", str(args.campplus_model)])
    call([sys.executable, str(SCRIPT_DIR / "release_approved_keys.py"),
          "--sealed-manifest", str(protected / "server" / "sealed_track_keys.json"),
          "--ttp-private-key", str(private_key), "--track", track.stem,
          "--consent-reference", "local-validation", "--output", str(release)])
    verify_tamper_rejected(protected, release, track.stem, work)
    call([sys.executable, str(SCRIPT_DIR / "restore_tracks.py"),
          "--sealed-manifest", str(protected / "server" / "sealed_track_keys.json"),
          "--consent-release", str(release), "--output", str(restored)])
    recovered = restored / track.name
    if sha256(track) != sha256(recovered):
        raise RuntimeError("Restored track bytes differ from the original")
    print("PASS: AES-GCM authentication, RSA-OAEP release, and byte-exact restoration verified.")


# Parse an explicit real-track validation input and an isolated temporary output location.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the voice-track cryptography workflow.")
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campplus-model", type=Path, required=True)
    args = parser.parse_args()
    if not (args.source_output / "tracks").is_dir() or not args.campplus_model.is_file():
        parser.error("The source output's tracks directory and CAM++ model must exist")
    if not (args.source_output / "diarization.json").is_file():
        parser.error("The source output is missing diarization.json")
    return args


if __name__ == "__main__":
    run(parse_args())
