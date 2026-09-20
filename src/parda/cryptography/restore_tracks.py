#!/usr/bin/env python3
# Restore only tracks whose AES keys were released by the trusted third party.

import argparse
import json
from pathlib import Path

from track_crypto import b64decode, decrypt_file


# Map a consent release's approved keys by track ID.
def released_keys(release: dict) -> dict[str, bytes]:
    keys = {}
    for record in release.get("approved_tracks", []):
        track_id = str(record.get("track_id", ""))
        if not track_id or track_id in keys:
            raise ValueError("Consent release contains an invalid or duplicate track ID")
        keys[track_id] = b64decode(record["key_b64"])
    return keys


# Decrypt only released tracks and verify each GCM tag plus original SHA-256.
def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.sealed_manifest.read_text(encoding="utf-8"))
    release = json.loads(args.consent_release.read_text(encoding="utf-8"))
    keys = released_keys(release)
    restored = 0
    for record in manifest.get("tracks", []):
        key = keys.get(record["track_id"])
        if key is None:
            continue
        encrypted = args.sealed_manifest.parent / record["encrypted_file"]
        destination = args.output / record["source_file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        decrypt_file(encrypted, destination, key, record)
        restored += 1
        print(f"restored {destination}", flush=True)
    if not restored:
        raise ValueError("No approved tracks appeared in the sealed manifest")


# Parse a wearer-side restoration operation using a TTP-issued consent release.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore AES-GCM speaker tracks from a TTP consent release."
    )
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--consent-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sealed_manifest.is_file() or not args.consent_release.is_file():
        parser.error("The sealed manifest and consent release must exist")
    return args


if __name__ == "__main__":
    run(parse_args())
