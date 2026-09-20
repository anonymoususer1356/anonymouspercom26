#!/usr/bin/env python3
# Replace raw ASR words with anonymized words while preserving trustworthy timings.

import argparse
import difflib
import json
import re
import unicodedata
from pathlib import Path


NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")
SPEAKER_PREFIX_RE = re.compile(r"^speaker(?:[ _-][A-Za-z0-9]+)*\s*:\s*(.*)$", re.IGNORECASE)


# Convert text to a punctuation-insensitive form for stable word matching.
def word_key(word: str) -> str:
    normalized = unicodedata.normalize("NFKD", word).casefold()
    return "".join(char for char in normalized if char.isalnum())


# Read one durable JSON record per line and fail rather than silently skipping corruption.
def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        records.append(record)
    return records


# Read numbered SLM output while keeping only the anonymized text for each source line.
def read_anonymised_lines(path: Path) -> dict[int, str]:
    lines = {}
    for physical_line, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = NUMBERED_LINE_RE.match(raw_line)
        if match is None:
            raise ValueError(f"{path}:{physical_line} is not a numbered transcript line")
        number = int(match.group(1))
        if number in lines:
            raise ValueError(f"{path}:{physical_line} repeats line {number}")
        text = match.group(2).strip()
        speaker_prefix = SPEAKER_PREFIX_RE.match(text)
        if speaker_prefix is not None:
            text = speaker_prefix.group(1).strip()
        lines[number] = text
    return lines


# Build a fallback timing sequence only when Sherpa did not return token timestamps.
def fallback_words(text: str, start_seconds: float, end_seconds: float) -> list[dict]:
    words = text.split()
    if not words:
        return []
    total_weight = sum(max(1, len(word_key(word))) for word in words)
    cursor = start_seconds
    output = []
    for index, word in enumerate(words):
        weight = max(1, len(word_key(word)))
        end = end_seconds if index + 1 == len(words) else cursor + (end_seconds - start_seconds) * weight / total_weight
        output.append({"word": word, "start_seconds": cursor, "end_seconds": end})
        cursor = end
    return output


# Find conservative near-spelling matches inside regions that did not match exactly.
def fuzzy_pairs(old_keys: list[str], new_keys: list[str], old_start: int, new_start: int) -> list[tuple[int, int]]:
    pairs = []
    old_cursor = 0
    for new_index, new_key in enumerate(new_keys):
        if len(new_key) < 4:
            continue
        best = None
        for old_index in range(old_cursor, len(old_keys)):
            old_key = old_keys[old_index]
            if len(old_key) < 4:
                continue
            similarity = difflib.SequenceMatcher(None, old_key, new_key).ratio()
            if similarity >= 0.88 and (best is None or similarity > best[0]):
                best = (similarity, old_index)
        if best is not None:
            old_cursor = best[1] + 1
            pairs.append((old_start + best[1], new_start + new_index))
    return pairs


# Match unchanged words exactly first, then only accept strong spelling-level fuzzy matches.
def match_words(source_words: list[dict], target_words: list[str]) -> dict[int, int]:
    old_keys = [word_key(str(item["word"])) for item in source_words]
    new_keys = [word_key(word) for word in target_words]
    matcher = difflib.SequenceMatcher(None, old_keys, new_keys, autojunk=False)
    pairs = []
    previous_old = previous_new = 0
    for old_start, new_start, size in matcher.get_matching_blocks():
        pairs.extend(fuzzy_pairs(
            old_keys[previous_old:old_start], new_keys[previous_new:new_start],
            previous_old, previous_new,
        ))
        pairs.extend((old_start + index, new_start + index) for index in range(size))
        previous_old = old_start + size
        previous_new = new_start + size
    return {new_index: old_index for old_index, new_index in pairs}


# Spread a replacement's duration across its new words without changing other word timings.
def replacement_timing(target_words: list[str], start_seconds: float, end_seconds: float) -> list[dict]:
    if not target_words:
        return []
    start_seconds = float(start_seconds)
    end_seconds = max(start_seconds, float(end_seconds))
    total_weight = sum(max(1, len(word_key(word))) for word in target_words)
    cursor = start_seconds
    output = []
    for index, word in enumerate(target_words):
        weight = max(1, len(word_key(word)))
        end = end_seconds if index + 1 == len(target_words) else cursor + (end_seconds - start_seconds) * weight / total_weight
        output.append({
            "word": word,
            "start_seconds": round(cursor, 3),
            "end_seconds": round(end, 3),
        })
        cursor = end
    return output


# Preserve matched word timings and replace only the time spans changed by the anonymizer.
def anonymise_words(source_words: list[dict], target_text: str,
                    line_start: float, line_end: float) -> tuple[list[dict], dict]:
    target_words = target_text.split()
    matches = match_words(source_words, target_words)
    output = []
    target_index = 0
    while target_index < len(target_words):
        source_index = matches.get(target_index)
        if source_index is not None:
            source = source_words[source_index]
            output.append({
                "word": target_words[target_index],
                "start_seconds": round(float(source["start_seconds"]), 3),
                "end_seconds": round(float(source["end_seconds"]), 3),
            })
            target_index += 1
            continue

        replacement_start = target_index
        while target_index < len(target_words) and target_index not in matches:
            target_index += 1
        replacement_end = target_index
        previous_source = matches.get(replacement_start - 1)
        next_source = matches.get(replacement_end)
        source_start = 0 if previous_source is None else previous_source + 1
        source_end = len(source_words) if next_source is None else next_source
        if source_start < source_end:
            start_seconds = source_words[source_start]["start_seconds"]
            end_seconds = source_words[source_end - 1]["end_seconds"]
        elif output:
            start_seconds = end_seconds = output[-1]["end_seconds"]
        elif next_source is not None:
            start_seconds = end_seconds = source_words[next_source]["start_seconds"]
        else:
            start_seconds = line_start
            end_seconds = line_end
        output.extend(replacement_timing(
            target_words[replacement_start:replacement_end], start_seconds, end_seconds,
        ))

    if [word_key(item["word"]) for item in output] != [word_key(word) for word in target_words]:
        raise RuntimeError("The aligned word sequence does not reconstruct the anonymized line")
    exact_matches = sum(
        1 for target_index, source_index in matches.items()
        if word_key(target_words[target_index]) == word_key(str(source_words[source_index]["word"]))
    )
    return output, {
        "source_word_count": len(source_words),
        "anonymised_word_count": len(target_words),
        "preserved_word_count": exact_matches,
        "fuzzy_preserved_word_count": len(matches) - exact_matches,
    }


# Join raw word timing records to their finalized transcript lines and SLM replacements.
def run(args: argparse.Namespace) -> None:
    source = args.source_output
    word_records = read_jsonl(source / "word_timestamps.jsonl")
    turns = read_jsonl(source / "timestamped_lines.jsonl")
    anonymised_lines = read_anonymised_lines(source / "anonymised_transcript.txt")
    if len(word_records) != len(turns):
        raise ValueError("word_timestamps.jsonl and timestamped_lines.jsonl must have the same record count")

    output = []
    dropped_lines = 0
    fallback_lines = 0
    for line_number, (word_record, turn) in enumerate(zip(word_records, turns), 1):
        speaker = str(turn.get("speaker", ""))
        if str(word_record.get("speaker", "")) != speaker:
            raise ValueError(f"Line {line_number} has mismatched speakers in the raw timing files")
        target_text = anonymised_lines.get(line_number)
        if not target_text:
            dropped_lines += 1
            continue
        source_words = word_record.get("words", [])
        if not isinstance(source_words, list) or not all(isinstance(item, dict) for item in source_words):
            raise ValueError(f"Line {line_number} has invalid word timing data")
        if not source_words:
            fallback_lines += 1
            source_words = fallback_words(
                str(turn.get("text", "")),
                float(turn["start_seconds"]),
                float(turn["end_seconds"]),
            )
        aligned_words, alignment = anonymise_words(
            source_words,
            target_text,
            float(turn["start_seconds"]),
            float(turn["end_seconds"]),
        )
        output.append({
            "line_number": line_number,
            "speaker": speaker,
            "start_seconds": turn["start_seconds"],
            "end_seconds": turn["end_seconds"],
            "boundary": turn.get("boundary"),
            "text": target_text,
            "timing_source": "nemotron_token_timestamps_with_anonymizer_alignment",
            "words": aligned_words,
            "alignment": alignment,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in output:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        f"wrote {args.output}: {len(output)} anonymized lines, "
        f"{dropped_lines} dropped lines, {fallback_lines} fallback-timed lines",
        flush=True,
    )


# Parse one completed orchestrator output directory and an optional output override.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align anonymized transcript words to Nemotron word timestamps."
    )
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.source_output = args.source_output.resolve()
    if args.output is None:
        args.output = args.source_output / "anonymised_word_timestamps.jsonl"
    args.output = args.output.resolve()
    for name in ("word_timestamps.jsonl", "timestamped_lines.jsonl", "anonymised_transcript.txt"):
        if not (args.source_output / name).is_file():
            parser.error(f"Source output is missing {name}")
    return args


if __name__ == "__main__":
    run(parse_args())
