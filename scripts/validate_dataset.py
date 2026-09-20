#!/usr/bin/env python3
"""Validate released labels and optionally regenerate split manifests."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

CATEGORIES = {"identity", "location", "demographic", "health", "legal",
              "economic", "relational", "lifestyle", "appearance"}
EXPECTED = {"train": 0, "val": 0, "test": 143, "pending_assignment": 574}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(root: Path) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for directory_name in EXPECTED:
        split_root = root / "silver_labels" / directory_name
        for directory in sorted(path for path in split_root.iterdir() if path.is_dir()):
            label = directory / "labels.json"
            if not label.is_file():
                raise ValueError(f"missing {label}")
            if directory.name in seen:
                raise ValueError(f"duplicate conversation ID: {directory.name}")
            seen.add(directory.name)
            payload = json.loads(label.read_text(encoding="utf-8"))
            for speaker in payload.get("speakers", []):
                unknown = set(speaker) - ({"speaker"} | CATEGORIES)
                if unknown:
                    raise ValueError(f"{label}: unknown categories {sorted(unknown)}")
                for category in CATEGORIES:
                    for item in speaker.get(category, []):
                        required = {"reasoning", "value", "lines", "certainty"}
                        if not required <= set(item):
                            raise ValueError(f"{label}: malformed {category} item")
            split = directory_name if directory_name != "pending_assignment" else "pending"
            rows.append({
                "conversation_id": directory.name,
                "candor_relative_path": directory.name,
                "transcript_filename": f"{directory.name}.txt",
                "split": split,
                "label_path": str(label.relative_to(root)),
                "membership_status": "verified" if split == "test" else "pending",
                "sha256": digest(label),
            })
    counts = {"train": 0, "val": 0, "test": 0, "pending_assignment": 0}
    for row in rows:
        key = "pending_assignment" if row["split"] == "pending" else row["split"]
        counts[key] += 1
    if counts != EXPECTED:
        raise ValueError(f"unexpected counts: {counts}; expected {EXPECTED}")
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["conversation_id", "candor_relative_path", "transcript_filename", "split",
              "label_path", "membership_status", "sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = inspect(args.dataset_root)
    if args.write:
        write_csv(args.dataset_root / "candor_manifest.csv", rows)
        for split in ("train", "val", "test"):
            write_csv(args.dataset_root / "splits" / f"{split}.csv",
                      [row for row in rows if row["split"] == split])
    print(f"validated {len(rows)} label sets: 143 test, 574 pending")


if __name__ == "__main__":
    main()
