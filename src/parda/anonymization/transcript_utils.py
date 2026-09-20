"""Transcript loading, model-output parsing, and safe line edits."""

import json
import re
from pathlib import Path

from validation import extract_json as extract_validated_json


NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")
SPEAKER_LINE = re.compile(r"^\s*([A-Za-z_][\w .-]*?)\s*:\s*(.*)$")
EDIT_LINE = re.compile(r"^edit\s+(\d+)\s*(.*)$", re.IGNORECASE)
SPEAKER_PREFIX = re.compile(r"^speaker(?:[ _.-]|$)", re.IGNORECASE)

# Format the input transcript into the numbered form used by the prompts.
def format_transcript(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{number}: {speaker}: {text}" for number, (speaker, text) in enumerate(turns, 1))


# Read plain transcript text and turn each line into a speaker turn.
def parse_turns(text: str) -> list[tuple[str, str]]:
    turns = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Remove an existing line number before parsing the speaker and text.
        numbered = NUMBERED_LINE.match(line)
        if numbered:
            line = numbered.group(2).strip()
        # Split a labelled line into its speaker and spoken content.
        speaker = SPEAKER_LINE.match(line)
        turns.append((speaker.group(1).strip(), speaker.group(2).strip()) if speaker else ("speaker_a", line))
    return turns


# Load one or more transcripts from JSONL, JSON, or ordinary text.
def load_records(source: str | Path) -> list[list[tuple[str, str]]]:
    path = Path(source)
    text = path.read_text(encoding="utf-8") if path.exists() else str(source)
    # JSONL contains one transcript record per line.
    if path.suffix == ".jsonl":
        return [_turns_from_json(json.loads(line)) for line in text.splitlines() if line.strip()]
    # JSON may contain one record or a list of records.
    if path.suffix == ".json":
        data = json.loads(text)
        return [_turns_from_json(item) for item in (data if isinstance(data, list) else [data])]
    return [parse_turns(text)]


# Convert one JSON transcript record into the common speaker/text structure.
def _turns_from_json(record: object) -> list[tuple[str, str]]:
    if isinstance(record, dict) and "text" in record:
        return parse_turns(str(record["text"]))
    values = record.get("turns", []) if isinstance(record, dict) else record
    return [(str(item.get("speaker", "speaker_a")), str(item.get("text", ""))) if isinstance(item, dict)
            else (str(item[0]), str(item[1])) for item in values]


# Models sometimes add commentary before the JSON, so try both JSON starts.
def extract_json(text: str):
    # Recover a model JSON payload using the shared validation parser.
    return extract_validated_json(text)


# Keep only lines in the model's `edit 12 ...` format.
def extract_edits(text: str) -> list[tuple[int, str]]:
    edits = []
    for line in text.splitlines():
        match = EDIT_LINE.match(line.strip())
        if match:
            edits.append((int(match.group(1)), match.group(2).strip()))
    return edits


# Recover the source speaker label used by a numbered transcript line.
def source_speaker(line: str) -> str | None:
    numbered = NUMBERED_LINE.match(line)
    speaker = SPEAKER_LINE.match(numbered.group(2)) if numbered else None
    return speaker.group(1).strip() if speaker else None


# Repair harmless edit-format slips without allowing a model to change speakers.
def normalise_replacement(number: int, replacement: str, source_line: str,
                          errors: list[str]) -> str:
    replacement = replacement.strip()
    if not replacement:
        return ""

    # Accept both prompt-style `edit 3 speaker_a: ...` and `edit 3: speaker_a: ...`.
    if replacement.startswith(":"):
        replacement = replacement[1:].lstrip()
    if not replacement:
        return ""

    expected_speaker = source_speaker(source_line)
    if expected_speaker is None:
        return replacement

    parsed = SPEAKER_LINE.match(replacement)
    if parsed and SPEAKER_PREFIX.match(parsed.group(1).strip()):
        supplied_speaker = parsed.group(1).strip()
        replacement_text = parsed.group(2).strip()
        if speaker_key(supplied_speaker) != speaker_key(expected_speaker):
            errors.append(
                f"edit {number}: repaired speaker '{supplied_speaker}' to '{expected_speaker}'"
            )
        return f"{expected_speaker}: {replacement_text}"

    errors.append(f"edit {number}: repaired missing speaker label '{expected_speaker}'")
    return f"{expected_speaker}: {replacement}"


# Index the current transcript by line number without renumbering it.
def apply_edits(transcript: str, edits: list[tuple[int, str]],
                errors: list[str]) -> tuple[str, list[tuple[int, str]]]:
    lines = {int(match.group(1)): match.group(0).strip() for raw in transcript.splitlines()
             if (match := NUMBERED_LINE.match(raw.strip()))}
    applied = []
    for number, replacement in edits:
        if number not in lines:
            # The model referred to text that was not present in this window.
            errors.append(f"edit {number}: line is not in this transcript")
            continue

        replacement = normalise_replacement(number, replacement, lines[number], errors)
        if not replacement:
            # An empty edit is the pipeline's explicit delete-line protocol.
            del lines[number]
            applied.append((number, replacement))
            continue

        updated = f"{number}: {replacement}"
        if updated == lines[number]:
            # Do not index a textually unchanged line as a retrieved example.
            continue
        if len(replacement) > 2.5 * len(lines[number].split(":", 1)[1].strip()):
            errors.append(f"edit {number}: replacement is implausibly long and was ignored")
            continue
        lines[number] = updated
        applied.append((number, replacement))
    return "\n".join(lines[number] for number in sorted(lines)), applied


# Normalise variants such as `speaker a` and `speaker_a` to one key.
def speaker_key(value: object) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return "speaker_" + cleaned.removeprefix("speaker_")
