"""Validation and recovery for model-produced adversary profiles.

The adversary profile is persistent context: a bad record accepted here is
shown to every later anonymiser window.  This module therefore keeps parsing,
normalisation, validation, and duplicate detection together instead of
spreading those rules across the pipeline and the storage class.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import json_repair


ALLOWED_PROFILE_ATTRIBUTES = frozenset({
    "legal", "economic", "relational", "lifestyle", "appearance",
    "identity", "location", "health", "demographic",
})

NO_INFERENCE_GUESS_RE = re.compile(
    r"^(?:\[\s*\]|none(?:\b|;)|no new\b|no additional\b|"
    r"no stable\b|no reliable\b|no profile\b|no speaker(?:_[ab])?\b|"
    r"no confirmed (?:new|personal|economic|specific|demographic|health|"
    r"legal|relationship)\b)",
    re.IGNORECASE,
)

GENERIC_GUESS_RE = re.compile(
    r"^(?:something|someone|somebody|a person|a name|name here|"
    r"placeholder(?: here)?|insert .+ here|unknown|not specified)$",
    re.IGNORECASE,
)

SPEAKER_KEY_RE = re.compile(r"[^a-z0-9]+")
LINE_FIELDS = ("lines", "line_numbers", "evidence_lines", "line_refs")


def speaker_key(value: object) -> str:
    # Canonicalise spelling variants such as ``speaker a`` and ``speaker_a``.
    cleaned = SPEAKER_KEY_RE.sub("_", str(value).strip().lower()).strip("_")
    while cleaned.startswith("speaker_"):
        cleaned = cleaned[len("speaker_"):]
    return f"speaker_{cleaned}" if cleaned else ""


def normalise_guess(value: str) -> str:
    # Create a stable comparison form without changing stored text.
    return re.sub(r"\s+", " ", value).strip().casefold()


def is_no_inference_guess(value: object) -> bool:
    # Recognise an empty answer hidden inside a schema-shaped record.
    return bool(NO_INFERENCE_GUESS_RE.match(str(value or "").strip()))


def is_generic_guess(value: str) -> bool:
    # Reject placeholder-like guesses that are not useful profile facts.
    return bool(GENERIC_GUESS_RE.fullmatch(value.strip()))


@dataclass
class ValidationReport:
    """Auditable outcome of one profile payload."""

    status: str = "not_checked"
    accepted: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reasons.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "accepted": self.accepted,
                "rejected": self.rejected, "reasons": list(self.reasons)}


@dataclass
class JsonParseResult:
    value: Any
    status: str
    detail: str | None = None


def _candidate_starts(text: str) -> list[int]:
    return sorted(index for index, char in enumerate(text) if char in "{[")


def extract_json_result(text: str) -> JsonParseResult:
    # Recover the most plausible JSON object/list from model prose.
    # json_repair fixes common truncation, missing quotes, and trailing commas.
    # Every bracket position is tried because prose can contain a citation such
    # as [1] before the real answer. Non-empty objects win; otherwise a list
    # containing records wins. An empty list remains a valid adversary answer.
    text = text or ""
    starts = _candidate_starts(text)
    empty_list = False

    # Try each opening bracket in order. This is important for a flat adversary
    # list: the outer list must be accepted or repaired before any object brace
    # inside it is considered.
    for start in starts:
        candidate = text[start:]
        try:
            value = json.loads(candidate)
        except Exception:
            try:
                value = json_repair.loads(candidate)
            except Exception:
                continue
            status = "repaired"
        else:
            status = "valid"
        if isinstance(value, dict) and value:
            return JsonParseResult(value, status)
        if isinstance(value, list):
            if any(isinstance(item, dict) for item in value):
                return JsonParseResult(value, status)
            if not value:
                empty_list = True
    if empty_list:
        return JsonParseResult([], "valid_empty")
    return JsonParseResult(None, "malformed", "no recoverable JSON object or record list")


def extract_json(text: str) -> Any:
    # Compatibility wrapper returning only the recovered JSON value.
    return extract_json_result(text).value


def _record_items(payload: Any) -> tuple[list[Any], str | None]:
    # Flatten the supported old and current profile response shapes.
    if isinstance(payload, dict):
        for key in ("inferences", "records"):
            if key in payload:
                return _record_items(payload[key])
        if isinstance(payload.get("speakers"), list):
            payload = payload["speakers"]
        elif all(isinstance(value, dict) for value in payload.values()):
            # Legacy: {"speaker_a": {"identity": [...]}}
            flattened = []
            for speaker, attributes in payload.items():
                for attribute, values in attributes.items():
                    if not isinstance(values, list):
                        return [], f"attribute '{attribute}' is not a list"
                    for item in values:
                        if isinstance(item, dict):
                            flattened.append({"speaker": speaker, "attribute": attribute, **item})
                        else:
                            flattened.append({"speaker": speaker, "attribute": attribute, "guess": item})
            return flattened, None
        else:
            return [payload], None

    if not isinstance(payload, list):
        return [], "profile payload must be a list or object"

    flattened = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("speaker"), str):
            attributes = [key for key in item if key not in {"speaker", "attribute", "guess", "reasoning"}]
            if "attribute" in item or not attributes:
                flattened.append(item)
                continue
            # Nested: {"speaker": "speaker_a", "identity": [{...}]}
            for attribute in attributes:
                values = item[attribute]
                if not isinstance(values, list):
                    return [], f"attribute '{attribute}' is not a list"
                for value in values:
                    if isinstance(value, dict):
                        flattened.append({"speaker": item["speaker"], "attribute": attribute, **value})
                    else:
                        flattened.append({"speaker": item["speaker"], "attribute": attribute, "guess": value})
        else:
            flattened.append(item)
    return flattened, None


class ProfileValidator:
    """Validate and exact-deduplicate profile records before they become context."""

    def __init__(self, *, max_guess_length: int = 1000, max_records_per_payload: int = 100):
        self.max_guess_length = max_guess_length
        self.max_records_per_payload = max_records_per_payload

    def _duplicate(self, record: dict, existing: list[dict]) -> bool:
        slot = (record["speaker"], record["attribute"])
        prior = [item for item in existing if (item["speaker"], item["attribute"]) == slot]
        new_key = normalise_guess(record["guess"])
        if any(normalise_guess(item["guess"]) == new_key for item in prior):
            return True
        # Do not infer semantic equivalence here. Paraphrase deduplication was
        # untested and can incorrectly discard two distinct facts.
        return False

    def validate(self, payload: Any, *, existing: list[dict] | None = None,
                 line_numbers: set[int] | None = None, source: str = "profile") -> tuple[list[dict], ValidationReport]:
        kept: list[dict] = []
        report = ValidationReport()
        if payload == []:
            report.status = "valid_empty"
            return kept, report
        items, shape_error = _record_items(payload)
        if shape_error:
            report.status = "malformed"
            report.reject(f"{source}: {shape_error}")
            return kept, report
        if len(items) > self.max_records_per_payload:
            report.status = "malformed"
            report.reject(f"{source}: payload contains {len(items)} records; limit is {self.max_records_per_payload}")
            return kept, report

        existing = existing or []
        for index, item in enumerate(items):
            label = f"{source}[{index}]"
            if not isinstance(item, dict):
                report.reject(f"{label}: record is not an object")
                continue
            speaker, attribute, guess = item.get("speaker"), item.get("attribute"), item.get("guess")
            if not isinstance(speaker, str) or not speaker.strip():
                report.reject(f"{label}: speaker is missing or not a string")
                continue
            if not isinstance(attribute, str) or not attribute.strip():
                report.reject(f"{label}: attribute is missing or not a string")
                continue
            if not isinstance(guess, str) or not guess.strip():
                report.reject(f"{label}: guess is missing or not a string")
                continue
            canonical_speaker = speaker_key(speaker)
            canonical_attribute = attribute.strip().casefold()
            if not canonical_speaker:
                report.reject(f"{label}: speaker is empty after normalisation")
                continue
            if canonical_attribute not in ALLOWED_PROFILE_ATTRIBUTES:
                report.reject(f"{label}: invalid attribute '{attribute}'")
                continue
            clean_guess = guess.strip()
            if len(clean_guess) > self.max_guess_length:
                report.reject(f"{label}: guess exceeds {self.max_guess_length} characters")
                continue
            if is_no_inference_guess(clean_guess):
                report.reject(f"{label}: guess is a no-inference answer")
                continue
            if is_generic_guess(clean_guess):
                report.reject(f"{label}: guess is a generic placeholder")
                continue
            invalid_lines = self._invalid_line_refs(item, line_numbers)
            if invalid_lines:
                report.reject(f"{label}: invalid line references {invalid_lines}")
                continue
            record = {"speaker": canonical_speaker, "attribute": canonical_attribute,
                      "reasoning": item.get("reasoning", "") if isinstance(item.get("reasoning", ""), str) else "",
                      "guess": clean_guess}
            if self._duplicate(record, existing + kept):
                report.reject(f"{label}: duplicate of an existing {canonical_speaker}/{canonical_attribute} guess")
                continue
            kept.append(record)
            report.accepted += 1
        report.status = "valid" if not report.reasons else "valid_with_rejections"
        return kept, report

    @staticmethod
    def _invalid_line_refs(item: dict, allowed: set[int] | None) -> list[Any]:
        if not any(field in item for field in LINE_FIELDS) or allowed is None:
            return []
        invalid: list[Any] = []
        for field in LINE_FIELDS:
            if field not in item:
                continue
            values = item[field]
            if not isinstance(values, list):
                return [f"{field} is not a list"]
            invalid.extend(value for value in values if not isinstance(value, int) or value not in allowed)
        return invalid
