#!/usr/bin/env python3
# Generate a TTP-only RSA keypair without placing the private key in source control.

import argparse
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from track_crypto import b64encode, write_env


# Generate a fresh RSA-4096 TTP keypair with filesystem permissions appropriate for a secret.
# Generate an RSA-4096 keypair and save it either as PEM files or in a gitignored environment file.
def run(args: argparse.Namespace) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if args.env_file is not None:
        write_env(args.env_file, {
            "TTP_RSA_PRIVATE_KEY_B64": b64encode(private_pem),
            "TTP_RSA_PUBLIC_KEY_B64": b64encode(public_pem),
        })
        print(f"saved reusable TTP keys in {args.env_file}", flush=True)
        return
    if args.private_key.exists() or args.public_key.exists():
        raise FileExistsError("Refusing to overwrite an existing TTP key")
    args.private_key.parent.mkdir(parents=True, exist_ok=True)
    args.public_key.parent.mkdir(parents=True, exist_ok=True)
    args.private_key.write_bytes(private_pem)
    os.chmod(args.private_key, 0o600)
    args.public_key.write_bytes(public_pem)
    os.chmod(args.public_key, 0o644)
    print(f"private TTP key: {args.private_key}", flush=True)
    print(f"public TTP key: {args.public_key}", flush=True)


# Parse key destinations without supplying a project-default secret location.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a fresh RSA-4096 TTP keypair.")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--env-file", type=Path,
                             help="Store Base64 PEM values in this gitignored environment file.")
    destination.add_argument("--private-key", type=Path)
    parser.add_argument("--public-key", type=Path)
    args = parser.parse_args()
    if args.env_file is None and args.public_key is None:
        parser.error("--public-key is required when --private-key is used")
    return args


if __name__ == "__main__":
    run(parse_args())
