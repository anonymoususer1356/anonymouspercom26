#!/usr/bin/env python3
# Run inside the trusted third party to release keys for explicitly approved tracks.

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from track_crypto import (
    b64encode,
    load_ttp_private_key,
    load_ttp_private_key_pem,
    pem_from_env,
    unwrap_key,
    write_json,
)


# Select only the explicitly approved track records from a sealed key manifest.
def approved_records(manifest: dict, track_ids: set[str]) -> list[dict]:
    records = {record["track_id"]: record for record in manifest.get("tracks", [])}
    missing = sorted(track_ids - set(records))
    if missing:
        raise ValueError("Unknown approved track IDs: " + ", ".join(missing))
    return [records[track_id] for track_id in sorted(track_ids)]


# Unwrap selected keys and write the package delivered to the wearer after consent.
def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.sealed_manifest.read_text(encoding="utf-8"))
    private_key = (
        load_ttp_private_key(args.ttp_private_key)
        if args.ttp_private_key is not None
        else load_ttp_private_key_pem(pem_from_env(args.env_file, "TTP_RSA_PRIVATE_KEY_B64"))
    )
    records = approved_records(manifest, set(args.track))
    release = {
        "format": "voice-track-consent-release-v1",
        "issued_at_utc": datetime.now(UTC).isoformat(),
        "consent_reference": args.consent_reference,
        "approved_tracks": [
            {
                "track_id": record["track_id"],
                "source_file": record["source_file"],
                "key_b64": b64encode(unwrap_key(record["wrapped_key_b64"], private_key)),
            }
            for record in records
        ],
    }
    write_json(args.output, release)
    print(f"released {len(records)} approved key(s) to {args.output}", flush=True)


# Parse a deliberate TTP-only approval operation; it never approves every track by default.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTP-only: unwrap keys for explicitly consented speaker tracks."
    )
    parser.add_argument("--sealed-manifest", type=Path, required=True)
    parser.add_argument("--ttp-private-key", type=Path,
                        help="Optional PEM path; otherwise use TTP_RSA_PRIVATE_KEY_B64 from --env-file.")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[3] / ".env")
    parser.add_argument("--track", action="append", required=True,
                        help="Approved track ID; repeat for each consented track.")
    parser.add_argument("--consent-reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.sealed_manifest.is_file():
        parser.error("The sealed manifest must exist")
    if args.ttp_private_key is not None and not args.ttp_private_key.is_file():
        parser.error("--ttp-private-key must exist when supplied")
    if args.ttp_private_key is None and not args.env_file.is_file():
        parser.error("Provide --ttp-private-key or an --env-file containing TTP_RSA_PRIVATE_KEY_B64")
    return args


if __name__ == "__main__":
    run(parse_args())
