# Retrieval storage for edits made by earlier anonymiser windows.

import json
import time
from dataclasses import dataclass, field

import numpy as np

from validation import ProfileValidator, ValidationReport

# Remove the line number and speaker label before embedding transcript text.
def text_of(line: str) -> str:
    parts = line.split(":", 2)
    return parts[2].strip() if len(parts) == 3 and parts[0].strip().isdigit() else line.strip()


# Read the stable transcript line number used to recover nearby context.
def line_number_of(line: str) -> int | None:
    number = line.split(":", 1)[0].strip()
    return int(number) if number.isdigit() else None


# Mark a neighboring line so it cannot be mistaken for an edit pair.
def render_context_line(line: str) -> str:
    number, separator, rest = line.partition(":")
    if separator and number.strip().isdigit():
        return f"[{number.strip()}]{rest}"
    return line


class Embedder:
    # Load Nomic once; the configured CPU count applies to its PyTorch encoder.
    def __init__(self, model: str, device: str = "cpu", cpu_threads: int | None = None,
                 timing_log=None):
        if cpu_threads is not None:
            if cpu_threads < 1:
                raise ValueError("Embedding threads must be positive")
            import torch
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(1)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device, trust_remote_code=True)
        self.cpu_threads = cpu_threads
        self.timing_log = timing_log

    def encode(self, texts: list[str]) -> np.ndarray:
        started = time.perf_counter()
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self.timing_log is not None:
            record = {
                "started_monotonic": started,
                "elapsed_seconds": time.perf_counter() - started,
                "texts": len(texts),
                "threads": self.cpu_threads,
            }
            with open(self.timing_log, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class EditMemory:
    embedder: Embedder
    threshold: float = 0.774
    candidates: int = 20
    top_k: int = 8
    context_lines: int = 2
    edits: list[dict] = field(default_factory=list)
    vectors: np.ndarray | None = None
    transcript_lines: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.context_lines < 0:
            raise ValueError("context_lines cannot be negative")

    # Retain the first-seen form of each line so later edits do not alter context.
    def remember(self, lines: list[str]) -> None:
        for line in lines:
            number = line_number_of(line)
            if number is not None:
                self.transcript_lines.setdefault(number, line.strip())

    def add(self, edits: list[dict]) -> None:
        if not edits:
            return
        vectors = self.embedder.encode([text_of(edit["before"]) for edit in edits])
        self.edits.extend(edits)
        self.vectors = vectors if self.vectors is None else np.vstack((self.vectors, vectors))

    # Return up to the configured number of observed lines on either side.
    def context_for(self, edit: dict) -> tuple[list[str], list[str]]:
        number = line_number_of(edit["before"])
        if number is None or number not in self.transcript_lines:
            return [], []

        before_numbers = range(number - self.context_lines, number)
        after_numbers = range(number + 1, number + 1 + self.context_lines)
        before_numbers = [item for item in before_numbers if item in self.transcript_lines]
        after_numbers = [item for item in after_numbers if item in self.transcript_lines]
        before = [self.transcript_lines[item] for item in before_numbers]
        after = [self.transcript_lines[item] for item in after_numbers]
        return before, after

    # Render one prior edit together with its original neighboring lines.
    def render_edit(self, edit: dict) -> str:
        before_context, after_context = self.context_for(edit)
        lines = [render_context_line(line) for line in before_context]
        lines.append(f"Original: {edit['before']}")
        lines.append(f"Changed: {edit['after']}")
        lines.extend(render_context_line(line) for line in after_context)
        return "\n".join(lines)

    def query(self, window_lines: list[str]) -> str:
        self.remember(window_lines)
        if self.vectors is None or not window_lines:
            # Keep the RAG section explicit when no earlier edit is available.
            return "none yet, this is the first window."

        # The embedder may batch the work, but every row remains one independent
        # transcript-line query with its own threshold and top-k selection.
        query_texts = [text_of(line) for line in window_lines]
        query_texts = [text for text in query_texts if text]
        if not query_texts:
            return "none yet, this is the first window."

        scores = self.embedder.encode(query_texts) @ self.vectors.T
        chosen: dict[int, float] = {}
        for row in scores:
            kept_for_line = 0
            candidate_count = min(self.candidates, len(self.edits))
            for raw_index in np.argsort(-row)[:candidate_count]:
                index = int(raw_index)
                score = float(row[index])
                if score < self.threshold:
                    break
                chosen[index] = max(chosen.get(index, score), score)
                kept_for_line += 1
                if kept_for_line >= self.top_k:
                    break

        # The same prior edit may match several current lines. Include it once,
        # using its best score, without imposing another window-wide top-k cap.
        entries = sorted(chosen, key=lambda index: (chosen[index], index))
        rendered = [self.render_edit(self.edits[index]) for index in entries]
        return "\n\n".join(rendered) or "none yet, this is the first window."


@dataclass
class Profile:
    """Validated, append-only adversary findings shared across windows."""

    records: list[dict] = field(default_factory=list)
    validator: ProfileValidator = field(default_factory=ProfileValidator)
    last_report: ValidationReport = field(default_factory=ValidationReport, init=False)

    def add(self, value: object, *, source: str = "adversary",
            line_numbers: set[int] | None = None) -> list[dict]:
        # Validate a model payload and append only safe, canonical records.
        kept, self.last_report = self.validator.validate(
            value, existing=self.records, line_numbers=line_numbers, source=source
        )
        self.records.extend(kept)
        return kept

    def render(self) -> str:
        return json.dumps(self.records, indent=2, ensure_ascii=False)
