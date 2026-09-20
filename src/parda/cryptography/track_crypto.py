#!/usr/bin/env python3
# Shared encryption helpers for the separated-speaker-track consent workflow.

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


FORMAT_VERSION = "voice-track-envelope-v1"
KEY_BYTES = 32
NONCE_BYTES = 12
COPY_BLOCK_BYTES = 1024 * 1024
KEY_LABEL = b"SmartGlasses voice-track key v1"


# Encode binary metadata for JSON sidecars without losing bytes.
def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# Decode a JSON sidecar field back into its original bytes.
def b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


# Write a JSON document atomically so readers never see a partial consent record.
def write_json(path: Path, value: dict, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# Load a PEM public key and reject non-RSA or undersized TTP keys.
def load_ttp_public_key(path: Path) -> rsa.RSAPublicKey:
    return load_ttp_public_key_pem(path.read_bytes())


# Load and validate a TTP public-key PEM held in a local environment file.
def load_ttp_public_key_pem(pem: bytes) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(pem)
    if not isinstance(key, rsa.RSAPublicKey):
        raise ValueError("TTP public key must be an RSA public key")
    if key.key_size < 3072:
        raise ValueError("TTP RSA key must be at least 3072 bits")
    return key


# Load the TTP private key used only to create an approved key-release package.
def load_ttp_private_key(path: Path) -> rsa.RSAPrivateKey:
    return load_ttp_private_key_pem(path.read_bytes())


# Load and validate a TTP private-key PEM held in a local environment file.
def load_ttp_private_key_pem(pem: bytes) -> rsa.RSAPrivateKey:
    key = serialization.load_pem_private_key(pem, password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("TTP private key must be an RSA private key")
    if key.key_size < 3072:
        raise ValueError("TTP RSA key must be at least 3072 bits")
    return key


# Read simple KEY=value values without expanding shell syntax or leaking secrets to logs.
def read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip()
    return values


# Read a base64 PEM secret from the root environment file.
def pem_from_env(path: Path, name: str) -> bytes:
    value = read_env(path).get(name)
    if not value:
        raise ValueError(f"{name} is missing from {path}")
    return b64decode(value)


# Update only the requested environment variables and preserve unrelated local settings.
def write_env(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = []
    pending = dict(values)
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            name = line.split("=", 1)[0].strip()
            if name in pending:
                remaining.append(name + "=" + pending.pop(name))
                continue
        remaining.append(line)
    for name, value in pending.items():
        remaining.append(name + "=" + value)
    path.write_text("\n".join(remaining).rstrip() + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


# Create the authenticated metadata bound to one encrypted track ciphertext.
def track_aad(track_id: str, source_name: str) -> bytes:
    value = {
        "format": FORMAT_VERSION,
        "track_id": track_id,
        "source_name": source_name,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Encrypt one track with AES-256-GCM while hashing it in bounded memory.
def encrypt_file(source: Path, destination: Path, key: bytes, aad: bytes) -> dict:
    if len(key) != KEY_BYTES:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    nonce = os.urandom(NONCE_BYTES)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(aad)
    plaintext_hash = hashlib.sha256()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            while block := input_handle.read(COPY_BLOCK_BYTES):
                plaintext_hash.update(block)
                output_handle.write(encryptor.update(block))
            output_handle.write(encryptor.finalize())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return {
        "algorithm": "AES-256-GCM",
        "nonce_b64": b64encode(nonce),
        "tag_b64": b64encode(encryptor.tag),
        "aad_b64": b64encode(aad),
        "plaintext_bytes": source.stat().st_size,
        "plaintext_sha256": plaintext_hash.hexdigest(),
        "ciphertext_bytes": destination.stat().st_size,
    }


# Encrypt a small metadata payload such as a voice fingerprint with AES-256-GCM.
def encrypt_bytes(data: bytes, key: bytes, aad: bytes) -> dict:
    if len(key) != KEY_BYTES:
        raise ValueError("AES-256-GCM requires a 32-byte key")
    nonce = os.urandom(NONCE_BYTES)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(aad)
    ciphertext = encryptor.update(data) + encryptor.finalize()
    return {
        "algorithm": "AES-256-GCM",
        "nonce_b64": b64encode(nonce),
        "tag_b64": b64encode(encryptor.tag),
        "aad_b64": b64encode(aad),
        "ciphertext_b64": b64encode(ciphertext),
        "plaintext_sha256": hashlib.sha256(data).hexdigest(),
    }


# Decrypt a small AES-GCM metadata payload and verify its plaintext hash.
def decrypt_bytes(record: dict, key: bytes) -> bytes:
    nonce = b64decode(record["nonce_b64"])
    tag = b64decode(record["tag_b64"])
    aad = b64decode(record["aad_b64"])
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(aad)
    plaintext = decryptor.update(b64decode(record["ciphertext_b64"])) + decryptor.finalize()
    if hashlib.sha256(plaintext).hexdigest() != record["plaintext_sha256"]:
        raise ValueError("Recovered metadata hash does not match its encrypted record")
    return plaintext


# Build distinct associated data for a TTP-only encrypted voice fingerprint.
def fingerprint_aad(track_id: str, model_sha256: str) -> bytes:
    value = {
        "format": FORMAT_VERSION,
        "purpose": "voice_fingerprint",
        "track_id": track_id,
        "model_sha256": model_sha256,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


# Decrypt one authenticated track and verify its original SHA-256 before release.
def decrypt_file(source: Path, destination: Path, key: bytes, record: dict) -> None:
    nonce = b64decode(record["nonce_b64"])
    tag = b64decode(record["tag_b64"])
    aad = b64decode(record["aad_b64"])
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    decryptor.authenticate_additional_data(aad)
    plaintext_hash = hashlib.sha256()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            while block := input_handle.read(COPY_BLOCK_BYTES):
                plaintext = decryptor.update(block)
                plaintext_hash.update(plaintext)
                output_handle.write(plaintext)
            plaintext = decryptor.finalize()
            plaintext_hash.update(plaintext)
            output_handle.write(plaintext)
        if plaintext_hash.hexdigest() != record["plaintext_sha256"]:
            raise ValueError("Recovered track hash does not match its sealed manifest")
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


# Wrap an AES track key so only the TTP private key can recover it.
def wrap_key(key: bytes, public_key: rsa.RSAPublicKey) -> str:
    encrypted = public_key.encrypt(
        key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=KEY_LABEL,
        ),
    )
    return b64encode(encrypted)


# Recover a sealed AES key inside the TTP environment.
def unwrap_key(value: str, private_key: rsa.RSAPrivateKey) -> bytes:
    key = private_key.decrypt(
        b64decode(value),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=KEY_LABEL,
        ),
    )
    if len(key) != KEY_BYTES:
        raise ValueError("Recovered key is not an AES-256-GCM key")
    return key


# Compute a stable public identifier for the TTP public key in the sealed manifest.
def public_key_fingerprint(public_key: rsa.RSAPublicKey) -> str:
    encoded = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(encoded).hexdigest()
