#!/usr/bin/env python3
# Run inside the TTP to decrypt voice fingerprints before matching enrolled voices.

import argparse
import json
from pathlib import Path

from track_crypto import (
    decrypt_bytes,
    load_ttp_private_key,
    load_ttp_private_key_pem,
    pem_from_env,
    unwrap_key,
    write_json,
)


# Decrypt every fingerprint record in the TTP package for its local matching service.
def run(args: argparse.Namespace) -> None:
    package = json.loads(args.package.read_text(encoding="utf-8"))
    private_key = (
        load_ttp_private_key(args.ttp_private_key)
        if args.ttp_private_key is not None
        else load_ttp_private_key_pem(pem_from_env(args.env_file, "TTP_RSA_PRIVATE_KEY_B64"))
    )
    fingerprints = []
    for record in package.get("tracks", []):
        key = unwrap_key(record["wrapped_key_b64"], private_key)
        fingerprints.append(json.loads(decrypt_bytes(record["fingerprint"], key)))
    write_json(args.output, {
        "format": "ttp-decrypted-voice-fingerprints-v1",
        "fingerprint_model": package.get("fingerprint_model"),
        "fingerprint_model_sha256": package.get("fingerprint_model_sha256"),
        "tracks": fingerprints,
    })
    print(f"decrypted {len(fingerprints)} fingerprint(s) for local TTP matching", flush=True)


# Parse a TTP-local decrypt operation; the resulting plaintext stays in the TTP environment.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTP-only: decrypt protected CAM++ fingerprints for local matching."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--ttp-private-key", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).resolve().parents[3] / ".env")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.package.is_file():
        parser.error("The encrypted fingerprint package must exist")
    if args.ttp_private_key is not None and not args.ttp_private_key.is_file():
        parser.error("--ttp-private-key must exist when supplied")
    if args.ttp_private_key is None and not args.env_file.is_file():
        parser.error("Provide --ttp-private-key or an --env-file containing TTP_RSA_PRIVATE_KEY_B64")
    return args


if __name__ == "__main__":
    run(parse_args())
