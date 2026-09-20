#!/usr/bin/env python3
# Download the exact source artifacts used to build the deployment models.
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "Models" / "HF Downloads"
NEMOTRON_PACKAGE = (
    "sherpa-onnx-nemotron-speech-streaming-en-0.6b-1120ms-int8-2026-04-25"
)
NEMOTRON_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{NEMOTRON_PACKAGE}.tar.bz2"
)


# Calculate an artifact digest for the download manifest.
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Download a Hugging Face file and copy it out of the shared cache.
def download_hf(repo_id: str, filename: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"kept existing {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    shutil.copy2(cached, destination)
    print(f"downloaded {repo_id}:{filename} -> {destination}")


# Extract only named files from a release archive without trusting its paths.
def extract_release_files(archive_path: Path, destination: Path, force: bool) -> None:
    expected = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "README.md")
    if not force and all((destination / name).exists() for name in expected):
        print(f"kept existing {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name in expected
        }
        missing = set(expected) - set(members)
        if missing:
            raise RuntimeError(f"Nemotron release is missing: {sorted(missing)}")
        for name in expected:
            source = archive.extractfile(members[name])
            if source is None:
                raise RuntimeError(f"Could not read {name} from {archive_path}")
            with source, (destination / name).open("wb") as output:
                shutil.copyfileobj(source, output)


# Download NVIDIA's released 1120 ms INT8 ONNX package exactly as used on the Pi.
def download_nemotron(destination: Path, force: bool) -> None:
    expected = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt", "README.md")
    if not force and all((destination / name).exists() for name in expected):
        print(f"kept existing {destination}")
        return
    with tempfile.TemporaryDirectory(prefix="nemotron-download-") as temporary:
        archive = Path(temporary) / f"{NEMOTRON_PACKAGE}.tar.bz2"
        urllib.request.urlretrieve(NEMOTRON_URL, archive)
        extract_release_files(archive, destination, force)
    print(f"downloaded {NEMOTRON_URL} -> {destination}")


# Write a small, portable manifest for the downloaded source artifacts.
def write_manifest(root: Path, destination: Path) -> None:
    artifacts = {
        "lseend": {
            "source": "GradientDescent2718/LS-EEND-ONNX",
            "files": [
                root / "lseend" / "DIHARD III" / "ls_eend_dih3_step.onnx",
                root / "lseend" / "DIHARD III" / "ls_eend_dih3_step.json",
            ],
        },
        "convtasnet_2": {
            "source": "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
            "files": [root / "convtasnet_2" / "pytorch_model.bin"],
        },
        "convtasnet_3": {
            "source": "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k",
            "files": [root / "convtasnet_3" / "pytorch_model.bin"],
        },
        "campplus": {
            "source": "welcomyou/campplus-3dspeaker-200k-onnx",
            "files": [root / "campplus" / "campplus_cn_en_common_200k.onnx"],
        },
        "nemotron": {
            "source": NEMOTRON_URL,
            "files": [
                root / "nemotron_1120ms_int8" / name
                for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
            ],
        },
    }
    result = {
        "root": str(root),
        "artifacts": {
            name: {
                "source": item["source"],
                "files": [
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for path in item["files"]
                ],
            }
            for name, item in artifacts.items()
        },
    }
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


# Download every raw artifact needed to reproduce the current deployment set.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-lseend", action="store_true")
    parser.add_argument("--skip-convtasnet", action="store_true")
    parser.add_argument("--skip-campplus", action="store_true")
    parser.add_argument("--skip-nemotron", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    root = args.output_root
    if not args.skip_lseend:
        download_hf(
            "GradientDescent2718/LS-EEND-ONNX",
            "DIHARD III/ls_eend_dih3_step.onnx",
            root / "lseend" / "DIHARD III" / "ls_eend_dih3_step.onnx",
            args.force,
        )
        download_hf(
            "GradientDescent2718/LS-EEND-ONNX",
            "DIHARD III/ls_eend_dih3_step.json",
            root / "lseend" / "DIHARD III" / "ls_eend_dih3_step.json",
            args.force,
        )
    if not args.skip_convtasnet:
        download_hf(
            "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
            "pytorch_model.bin",
            root / "convtasnet_2" / "pytorch_model.bin",
            args.force,
        )
        download_hf(
            "JorisCos/ConvTasNet_Libri3Mix_sepnoisy_16k",
            "pytorch_model.bin",
            root / "convtasnet_3" / "pytorch_model.bin",
            args.force,
        )
    if not args.skip_campplus:
        download_hf(
            "welcomyou/campplus-3dspeaker-200k-onnx",
            "campplus_cn_en_common_200k.onnx",
            root / "campplus" / "campplus_cn_en_common_200k.onnx",
            args.force,
        )
    if not args.skip_nemotron:
        download_nemotron(root / "nemotron_1120ms_int8", args.force)

    write_manifest(root, args.manifest or root / "SOURCE_MANIFEST.json")


if __name__ == "__main__":
    main()
