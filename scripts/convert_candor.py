#!/usr/bin/env python3
"""Convert a local CANDOR checkout into PARDA's numbered transcript format."""

import argparse
import csv
from pathlib import Path


def convert_conversation(source: Path, destination: Path) -> int:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return 0
    speaker_map: dict[str, str] = {}
    output = []
    for index, row in enumerate(rows, start=1):
        speaker = row["speaker"]
        if speaker not in speaker_map:
            label = chr(ord("a") + len(speaker_map))
            speaker_map[speaker] = f"speaker_{label}"
        output.append(f"{index}: {speaker_map[speaker]}: {row['utterance']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--manifest", type=Path,
                        help="Optional split CSV; process only its conversation_id values.")
    args = parser.parse_args()
    selected = None
    if args.manifest:
        with args.manifest.open(newline="", encoding="utf-8") as handle:
            selected = {row["conversation_id"] for row in csv.DictReader(handle)}
    written = skipped = turns = 0
    for conversation in sorted(path for path in args.candor_root.iterdir() if path.is_dir()):
        if selected is not None and conversation.name not in selected:
            continue
        source = conversation / "transcription" / "transcript_cliffhanger.csv"
        if not source.is_file():
            skipped += 1
            continue
        count = convert_conversation(source, args.output / f"{conversation.name}.txt")
        if count:
            written += 1
            turns += count
        else:
            skipped += 1
    print(f"written={written} skipped={skipped} turns={turns} output={args.output}")


if __name__ == "__main__":
    main()
