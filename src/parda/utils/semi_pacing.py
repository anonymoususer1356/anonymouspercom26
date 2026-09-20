# Coordinate optional idle-wait removal without changing model computation.
import json
import hashlib
import threading
import time
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_state(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class SemiPacer:
    # Keep real deadlines while recording the idle time removed from replay.
    def __init__(self, output: Path, origin: float):
        self.output = output
        self.origin = origin
        self.skipped = 0.0
        self.routed_chunk = -1
        self.idle_checks = []
        self.lock = threading.Lock()
        self.events = []
        write_state(output / "semi_paced_approval.json", {"chunk": None})
        (output / "semi_paced_skips.jsonl").write_text("", encoding="utf-8")
        (output / "semi_paced_decisions.jsonl").write_text("", encoding="utf-8")

    def mark_routed(self, chunk: int, idle_checks: list) -> None:
        with self.lock:
            self.routed_chunk = chunk
            self.idle_checks = idle_checks

    def audio_idle(self, previous_chunk: int) -> bool:
        with self.lock:
            return self.routed_chunk == previous_chunk and all(check() for check in self.idle_checks)

    # Ask the orchestrator only after all previously released audio has drained.
    def wait_for_chunk(self, chunk: int, source_end: float) -> None:
        request_path = self.output / "semi_paced_request.json"
        approval_path = self.output / "semi_paced_approval.json"
        deadline = self.origin + source_end - self.skipped
        wait_started = time.perf_counter()
        prior_skipped = self.skipped
        requested = False
        protected = False
        audio_idle_at = None
        write_state(request_path, {"chunk": chunk, "audio_idle": False,
                                   "deadline_monotonic": deadline})
        while time.perf_counter() < deadline:
            audio_idle = self.audio_idle(chunk - 1)
            if audio_idle and audio_idle_at is None:
                audio_idle_at = time.perf_counter()
            approval = read_state(approval_path)
            if approval.get("chunk") == chunk and approval.get("skip") is False:
                protected = True
            if not protected and audio_idle:
                if not requested:
                    write_state(request_path, {"chunk": chunk, "audio_idle": True,
                                               "deadline_monotonic": deadline})
                    requested = True
                approval = read_state(approval_path)
                if approval.get("chunk") == chunk and approval.get("skip") is True:
                    now = time.perf_counter()
                    removed = max(0.0, deadline - now)
                    self.skipped += removed
                    event = {"chunk": chunk, "at_monotonic": now,
                             "at_unix": time.time(), "skipped_seconds": removed,
                             "cumulative_skipped_seconds": self.skipped}
                    self.events.append(event)
                    with (self.output / "semi_paced_skips.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                    break
            time.sleep(min(0.01, max(0.0, deadline - time.perf_counter())))
        # Expired requests must not authorize a later block.
        write_state(request_path, {"chunk": chunk, "expired": True})
        decision = {"chunk": chunk, "source_end_seconds": source_end,
                    "wait_started_monotonic": wait_started,
                    "wait_finished_monotonic": time.perf_counter(),
                    "protected_by_slm": protected,
                    "audio_idle_monotonic": audio_idle_at,
                    "idle_slack_seconds": max(0.0, deadline - audio_idle_at) if audio_idle_at is not None else 0.0,
                    "skipped_seconds": self.skipped - prior_skipped}
        with (self.output / "semi_paced_decisions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision) + "\n")


def skipped_intervals(output: Path) -> list[dict]:
    path = output / "semi_paced_skips.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def virtual_offset(events: list[dict], unix: float) -> float:
    return sum(event["skipped_seconds"] for event in events if event["at_unix"] <= unix)


def reconstruct_audio_timeline(output: Path, started_unix: float, skips: list[dict]) -> None:
    path = output / "audio_timeline.json"
    if not path.exists():
        return
    trace = read_state(path)
    for event in trace.get("events", []):
        if event.get("stage") == "source_capture":
            continue
        for key in ("start_seconds", "end_seconds"):
            if key in event:
                event[key] += virtual_offset(skips, started_unix + event[key])
    trace["clock"] = "reconstructed_paced"
    trace["evaluation_mode"] = "semi-paced"
    write_state(output / "semi_paced_audio_timeline.json", trace)

    source_events = output / "end_to_end_timing.jsonl"
    if source_events.exists():
        reconstructed = []
        for line in source_events.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            for key in ("time_unix", "enqueued_unix", "completed_unix"):
                if isinstance(event.get(key), (int, float)):
                    event[key] += virtual_offset(skips, event[key])
            available = event.get("latest_source_available_unix")
            if available is not None and event.get("completed_unix") is not None:
                event["end_to_end_lag_seconds"] = event["completed_unix"] - available
            reconstructed.append(json.dumps(event))
        (output / "semi_paced_end_to_end_timing.jsonl").write_text(
            "\n".join(reconstructed) + "\n", encoding="utf-8"
        )

    benchmark_path = output / "audio_benchmark.json"
    if benchmark_path.exists():
        benchmark = read_state(benchmark_path)
        for events_key, lag_key in (("measured_chunks", "routing_completion_lag_seconds"),
                                    ("asr_events", "completion_lag_seconds")):
            for event in benchmark.get(events_key, []):
                completed = event["source_end_seconds"] + event[lag_key]
                event[lag_key] += virtual_offset(skips, started_unix + completed)
        for events_key, lag_key, summary_key in (
            ("measured_chunks", "routing_completion_lag_seconds", "routing_completion_lag"),
            ("asr_events", "completion_lag_seconds", "asr_completion_lag"),
        ):
            values = [event[lag_key] for event in benchmark.get(events_key, [])
                      if event["source_end_seconds"] > benchmark.get("warmup_seconds", 0)]
            benchmark[summary_key] = {"count": len(values),
                                      "mean_seconds": sum(values) / len(values) if values else None,
                                      "maximum_seconds": max(values) if values else None}
        benchmark["metric_clock"] = "reconstructed_paced"
        write_state(output / "semi_paced_audio_benchmark.json", benchmark)
