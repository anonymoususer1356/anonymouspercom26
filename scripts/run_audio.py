#!/usr/bin/env python3
# Run source separation, streaming ASR, and windowed SLM anonymisation together.

import argparse
import json
import hashlib
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = PROJECT_ROOT / "src" / "parda" / "anonymization"
UTILS_DIR = PROJECT_ROOT / "src" / "parda" / "utils"
SOURCE_PIPELINE = PROJECT_ROOT / "src" / "parda" / "audio" / "live_pipeline.py"
SOURCE_STREAM = PROJECT_ROOT / "src" / "parda" / "audio" / "live_stream.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "orchestrator"
DEPLOYMENT_MODELS = PROJECT_ROOT / "models" / "deployment"
RAW_MODELS = PROJECT_ROOT / "models" / "raw"
SLM_MODEL = PROJECT_ROOT / "models" / "qwen35-2b-sft-Q4_0.gguf"
UI_PYTHON = Path(sys.executable)
CRYPTO_PROTECTOR = PROJECT_ROOT / "src" / "parda" / "cryptography" / "protect_tracks.py"
CRYPTO_DIR = PROJECT_ROOT / "src" / "parda" / "cryptography"
SOURCE_SEPARATION_DIR = PROJECT_ROOT / "src" / "parda" / "audio"
WORD_TIMING_ANONYMISER = PROJECT_ROOT / "src" / "parda" / "utils" / "anonymise_word_timings.py"
OUTPUT_BUNDLE_ROOT = PROJECT_ROOT / "outputs" / "bundles"

NUMBERED_OUTPUT_RE = re.compile(r"^\s*(\d+)\s*:\s*(.*)$")
SPEAKER_PREFIX_RE = re.compile(r"^speaker(?:[ _-][A-Za-z0-9]+)*\s*:\s*(.*)$", re.IGNORECASE)
STOP = object()

sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(UTILS_DIR))
sys.path.insert(0, str(CRYPTO_DIR))
sys.path.insert(0, str(SOURCE_SEPARATION_DIR))

from backends import get_backend  # noqa: E402
from pipeline import WindowProcessor  # noqa: E402
from rag_store import EditMemory, Embedder  # noqa: E402
from windowing import get_tokenizer, lines_to_tokens  # noqa: E402
from protect_tracks import prepare_fingerprints, run as protect_tracks_run  # noqa: E402
from speaker_embeddings import ReDimNetEncoder  # noqa: E402
from slm_priority import SlmPriorityController  # noqa: E402
from semi_pacing import read_state, write_state, skipped_intervals, reconstruct_audio_timeline, file_sha256
from Prompts.prompt_generator import (  # noqa: E402
    build_anonymiser_rag_prompt,
    build_windowed_adversary_prompt,
)


# Write status files atomically so the web UI never reads half-written JSON or text.
def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# Return one compact health payload when a llama.cpp server is accepting requests.
def server_health(base_url: str, timeout: float = 1.0) -> dict | None:
    health_url = base_url.removesuffix("/v1") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


# Read recording duration without adding an audio library dependency to the harness.
def audio_duration_seconds(path: Path) -> float:
    command = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return float(result.stdout.strip())


class LlamaServer:
    # Start the selected SFT model once and expose two cacheable role slots.
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        self.args = args
        self.output_dir = output_dir
        self.process = None
        self.started_here = False

    def start(self) -> None:
        if server_health(self.args.llama_base_url):
            if self.args.benchmark_report is not None:
                raise RuntimeError(
                    "Benchmark port already has a llama-server. Use a free --llama-port "
                    "so the benchmark starts and records its own thread configuration."
                )
            print(f"Using llama.cpp already listening at {self.args.llama_base_url}", flush=True)
            return

        binary = Path(self.args.llama_server)
        if not binary.is_file():
            resolved = shutil.which(str(self.args.llama_server))
            if resolved is None:
                raise FileNotFoundError(f"llama-server was not found: {self.args.llama_server}")
            binary = Path(resolved)
        if not self.args.slm_model.is_file():
            raise FileNotFoundError(f"SLM model was not found: {self.args.slm_model}")

        command = [
            str(binary),
            "--model", str(self.args.slm_model),
            "--alias", "qwen35-2b-sft-q4_0",
            "--host", self.args.llama_host,
            "--port", str(self.args.llama_port),
            "--ctx-size", str(self.args.llama_total_context),
            "--parallel", "2",
            "--threads", str(self.args.llama_threads),
            "--threads-batch", str(self.args.llama_batch_threads),
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--flash-attn", "on",
            "--n-gpu-layers", "all",
        ]
        if self.args.slm_scope_unit:
            scope_prefix = [
                "systemd-run", "--scope", "--unit", self.args.slm_scope_unit,
                f"--property=CPUWeight={self.args.slm_cpu_weight}",
                f"--nice={self.args.slm_nice}",
            ]
            if self.args.slm_priority_use_sudo:
                scope_prefix = ["sudo", "-n", *scope_prefix]
            command = [
                *scope_prefix, *command,
            ]
        else:
            # ``nice`` is available without sudo and affects only the server
            # process started for this run. ``nice -n`` is relative, so derive
            # its increment from the launcher priority to honor the requested
            # absolute value even when SSH starts at elevated priority.
            increment = self.args.slm_nice - os.nice(0)
            command = ["nice", "-n", str(increment), *command]
        log_path = self.output_dir / "llama_server.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write("command=" + " ".join(command) + "\n")
        log_handle.flush()
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        self.started_here = True

        deadline = time.monotonic() + self.args.llama_start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(f"llama-server exited during startup:\n{tail}")
            if server_health(self.args.llama_base_url):
                print(f"llama.cpp ready at {self.args.llama_base_url}", flush=True)
                return
            time.sleep(0.25)
        raise TimeoutError(f"llama-server did not become ready within {self.args.llama_start_timeout}s")

    # Stop only a server started by this invocation and only when explicitly requested.
    def stop(self) -> None:
        if not self.started_here or self.process is None:
            return
        if self.args.slm_scope_unit:
            command = ["systemctl", "stop", self.args.slm_scope_unit]
            if self.args.slm_priority_use_sudo:
                command = ["sudo", "-n", *command]
            subprocess.run(command, check=False)
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class TranscriptAnonymiser:
    # Accumulate committed ASR turns and submit only whole Qwen-sized windows.
    def __init__(self, args: argparse.Namespace, output_dir: Path):
        self.args = args
        self.output_dir = output_dir
        self.tokenizer = get_tokenizer(args.tokenizer)
        self.token_budget = lines_to_tokens(args.window_lines)
        memory = EditMemory(
            Embedder(
                args.embed_model, args.embed_device, args.embed_threads,
                output_dir / "embedding_timing.jsonl",
            ),
            threshold=args.rag_threshold,
            candidates=args.rag_candidates,
            top_k=args.rag_top_k,
            context_lines=args.context_lines,
        )
        backend_options = {
            "model": "qwen35-2b-sft-q4_0",
            "stream": args.stream_slm_tokens,
            "no_think": True,
            "timing_log": output_dir / "slm_timing.jsonl",
            "seed": args.seed,
        }
        adversary = get_backend(
            "llama.cpp",
            base_url=args.llama_base_url,
            slot=0,
            **backend_options,
        )
        anonymiser = get_backend(
            "llama.cpp",
            base_url=args.llama_base_url,
            slot=1,
            **backend_options,
        )
        self.processor = WindowProcessor(
            adversary,
            anonymiser,
            memory,
            passes_per_window=args.passes_per_window,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        self.raw_lines = []
        self.raw_timing_records = {}
        self.input_started_unix = None
        self.end_to_end_timing_path = output_dir / "end_to_end_timing.jsonl"
        self.pending_lines = []
        self.pending_numbers = []
        self.pending_tokens = 0
        self.pending_since = None
        self.anonymised_windows = []
        self.benchmark_stress_outputs = []
        self.benchmark_slm_fixture = (
            args.benchmark_slm_fixture.read_text(encoding="utf-8").strip()
            if args.benchmark_slm_fixture is not None else None
        )
        self.benchmark_slm_empty_fixture = (
            args.benchmark_slm_empty_fixture.read_text(encoding="utf-8").strip()
            if args.benchmark_slm_empty_fixture is not None else None
        )

        # Validate one benchmark fixture and return its transcript lines without IDs.
        def fixture_lines(text: str | None, label: str) -> list[str]:
            lines = []
            if text is None:
                return lines
            for line in text.splitlines():
                match = NUMBERED_OUTPUT_RE.match(line)
                if match is None:
                    raise ValueError(f"Every {label} fixture line must have a numeric line ID")
                lines.append(match.group(2))
            text_tokens = sum(
                len(self.tokenizer.encode(line.split(":", 1)[-1].strip(), add_special_tokens=False))
                for line in lines
            )
            if text_tokens > self.token_budget:
                raise ValueError(f"{label} fixture exceeds i12 budget: {text_tokens}/{self.token_budget}")
            return lines

        self.fixture_lines = fixture_lines(self.benchmark_slm_fixture, "stress")
        self.empty_fixture_lines = fixture_lines(self.benchmark_slm_empty_fixture, "empty")
        self.window_records = []
        self.next_window_id = 0
        self.lock = threading.RLock()
        self.jobs = queue.Queue()
        self.queued_enqueued_at = []
        self.active_enqueued_at = None
        self.worker_error = None
        self.priority = SlmPriorityController(
            args.slm_priority_controller,
            args.slm_priority_mode,
            args.slm_scope_unit,
            args.slm_priority_pid,
            args.slm_background_cpus,
            args.slm_promoted_cpus,
            args.slm_high_watermark,
            args.slm_low_watermark,
            args.slm_max_wait_seconds,
            output_dir / "slm_queue.jsonl",
            args.slm_priority_use_sudo,
        )
        self.worker = threading.Thread(target=self.run_worker, name="slm-anonymiser", daemon=True)
        self.worker.start()

    # Prime both persistent llama.cpp slots before a benchmark's timed boundary.
    def warmup(self) -> None:
        text = "1: speaker_a: This fixed sentence warms the anonymisation service."
        profile = self.processor.profile.render()
        system, user = build_windowed_adversary_prompt(text, profile)
        adversary = self.processor.adversary.chat(
            system, user, self.args.temperature, self.args.top_p,
            self.args.max_tokens, "warmup_adversary",
        )
        if adversary.error:
            raise RuntimeError("SLM adversary warm-up failed: " + adversary.error)
        system, user = build_anonymiser_rag_prompt(text, profile, "[]")
        anonymiser = self.processor.anonymiser.chat(
            system, user, self.args.temperature, self.args.top_p,
            self.args.max_tokens, "warmup_anonymiser",
        )
        if anonymiser.error:
            raise RuntimeError("SLM anonymiser warm-up failed: " + anonymiser.error)
        self.log_end_to_end_event({"stage": "slm_warmup_completed"})

    # Attach wall-clock source availability to transcript and SLM timing events.
    def set_input_started_unix(self, started_unix: float) -> None:
        self.input_started_unix = started_unix

    # Append one durable timing event shared by the live-pacing analysis tools.
    def log_end_to_end_event(self, event: dict) -> None:
        event = {"time_unix": time.time(), **event}
        with self.lock:
            with self.end_to_end_timing_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    # Return outstanding SLM-window count and the age of the oldest uncompleted job.
    def queue_state(self) -> tuple[int, float | None]:
        with self.lock:
            times = list(self.queued_enqueued_at)
            if self.active_enqueued_at is not None:
                times.append(self.active_enqueued_at)
        if not times:
            return 0, None
        return len(times), max(0.0, time.monotonic() - min(times))

    # Record queue telemetry and change priority only when a watermark policy requires it.
    def observe_queue(self) -> None:
        depth, oldest_age = self.queue_state()
        self.priority.observe(depth, oldest_age)

    # Process queued windows serially so disclosure profile state remains ordered.
    def run_worker(self) -> None:
        while True:
            job = self.jobs.get()
            if job is STOP:
                self.jobs.task_done()
                return
            with self.lock:
                self.active_enqueued_at = self.queued_enqueued_at.pop(0)
            self.log_end_to_end_event({
                "stage": "slm_window_started",
                "window": job["window"],
                "line_numbers": job["line_numbers"],
                "enqueued_unix": job["enqueued_unix"],
            })
            self.observe_queue()
            try:
                self.process_window(job)
            except Exception as error:
                self.worker_error = error
            finally:
                with self.lock:
                    self.active_enqueued_at = None
                self.jobs.task_done()
                self.observe_queue()

    # Stop after all queued work is complete and surface a worker failure to the caller.
    def finish(self) -> None:
        self.jobs.join()
        self.jobs.put(STOP)
        self.worker.join()
        self.observe_queue()
        if self.worker_error is not None:
            raise RuntimeError("SLM worker failed") from self.worker_error

    # Convert one timestamped ASR record to the Harness's numbered line format.
    def add_record(self, record: dict) -> None:
        speaker = str(record.get("speaker", "speaker_a")).strip()
        text = str(record.get("text", "")).strip()
        if not text:
            return
        with self.lock:
            number = len(self.raw_lines) + 1
            line = f"{number}: {speaker}: {text}"
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False))
            if self.pending_lines and self.pending_tokens + token_count > self.token_budget:
                self.enqueue_window(final=False)
            self.raw_lines.append(line)
            self.raw_timing_records[number] = {
                "speaker": speaker,
                "start_seconds": record.get("start_seconds"),
                "end_seconds": record.get("end_seconds"),
                "committed_seconds": record.get("committed_seconds"),
                "boundary": record.get("boundary"),
            }
            source_end = record.get("end_seconds")
            source_available_unix = (
                self.input_started_unix + float(source_end)
                if self.input_started_unix is not None and isinstance(source_end, (int, float))
                else None
            )
            self.log_end_to_end_event({
                "stage": "transcript_commit_observed",
                "line_number": number,
                "speaker": speaker,
                "source_start_seconds": record.get("start_seconds"),
                "source_end_seconds": source_end,
                "source_available_unix": source_available_unix,
                "asr_committed_seconds": record.get("committed_seconds"),
                "boundary": record.get("boundary"),
            })
            self.pending_lines.append(line)
            self.pending_numbers.append(number)
            self.pending_tokens += token_count
            if self.pending_since is None:
                self.pending_since = time.monotonic()
        self.write_outputs()

    # Optionally release a partial window once its oldest committed line has waited long enough.
    def flush_if_due(self) -> None:
        limit = self.args.slm_max_buffer_seconds
        with self.lock:
            if limit and self.pending_since is not None:
                if time.monotonic() - self.pending_since >= limit:
                    self.enqueue_window(final=False, reason="buffer_age")

    # Enqueue one immutable window snapshot without blocking source transcription.
    def enqueue_window(self, final: bool, reason: str = "token_budget") -> None:
        if not self.pending_lines:
            return
        job = {
            "window": self.next_window_id,
            "line_numbers": list(self.pending_numbers),
            "original": "\n".join(self.pending_lines),
            "text_tokens": self.pending_tokens,
            "final_partial_window": final,
            "enqueued_unix": time.time(),
            "dispatch_reason": "end_of_input" if final else reason,
        }
        self.next_window_id += 1
        source_ends = [
            self.raw_timing_records[number].get("end_seconds")
            for number in job["line_numbers"]
            if isinstance(self.raw_timing_records[number].get("end_seconds"), (int, float))
        ]
        job["latest_source_end_seconds"] = max(source_ends) if source_ends else None
        job["latest_source_available_unix"] = (
            self.input_started_unix + job["latest_source_end_seconds"]
            if self.input_started_unix is not None and job["latest_source_end_seconds"] is not None
            else None
        )
        self.pending_lines = []
        self.pending_numbers = []
        self.pending_tokens = 0
        self.pending_since = None
        self.queued_enqueued_at.append(time.monotonic())
        self.jobs.put(job)
        self.log_end_to_end_event({
            "stage": "slm_window_enqueued",
            "window": job["window"],
            "line_numbers": job["line_numbers"],
            "text_tokens": job["text_tokens"],
            "final_partial_window": final,
            "latest_source_end_seconds": job["latest_source_end_seconds"],
            "latest_source_available_unix": job["latest_source_available_unix"],
            "enqueued_unix": job["enqueued_unix"],
            "queue_depth_after_enqueue": self.jobs.qsize(),
            "dispatch_reason": job["dispatch_reason"],
        })
        self.observe_queue()

    # Preserve the historical public method name while making submission asynchronous.
    def flush_window(self, final: bool) -> None:
        with self.lock:
            self.enqueue_window(final)

    # Run one queued adversary/anonymiser exchange in source-window order.
    def process_window(self, job: dict) -> None:
        index = job["window"]
        line_numbers = job["line_numbers"]
        print(
            f"[SLM window {index}] lines {line_numbers[0]}-{line_numbers[-1]}, "
            f"{job['text_tokens']}/{self.token_budget} text tokens"
            + (" (final)" if job["final_partial_window"] else ""),
            flush=True,
        )
        model_input = job["original"]
        model_numbers = line_numbers
        selected_fixture = None
        selected_lines = None
        fixture_active = None
        if self.benchmark_slm_fixture is not None:
            # Prefix-balanced rounding keeps every partial run close to the
            # requested active-window fraction without relying on random luck.
            previous_active = int(index * self.args.benchmark_slm_active_fraction + 0.5)
            current_active = int((index + 1) * self.args.benchmark_slm_active_fraction + 0.5)
            fixture_active = current_active > previous_active
            if fixture_active:
                selected_fixture = self.args.benchmark_slm_fixture
                selected_lines = self.fixture_lines
            else:
                selected_fixture = self.args.benchmark_slm_empty_fixture
                selected_lines = self.empty_fixture_lines
            # Synthetic IDs belong to the fixture, independently of ASR turn IDs.
            first = index * len(selected_lines) + 1
            if self.args.benchmark_slm_state == "fresh":
                first = 1
                # Reset only disclosure/RAG contents; the model, embedder and slots remain resident.
                old = self.processor
                memory = EditMemory(
                    old.memory.embedder, threshold=self.args.rag_threshold,
                    candidates=self.args.rag_candidates, top_k=self.args.rag_top_k,
                    context_lines=self.args.context_lines,
                )
                self.processor = WindowProcessor(
                    old.adversary, old.anonymiser, memory,
                    passes_per_window=self.args.passes_per_window,
                    temperature=self.args.temperature, top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                )
                self.processor.exchanges = old.exchanges
                self.processor.trace = old.trace
                self.processor.errors = old.errors
            model_numbers = list(range(first, first + len(selected_lines)))
            model_input = "\n".join(
                f"{number}: {line}" for number, line in zip(model_numbers, selected_lines)
            )
        anonymised = self.processor.process(index, model_numbers, model_input)
        if self.args.benchmark_report is not None:
            for role in ("adversary", "anonymiser"):
                exchange = self.processor.exchanges[f"{role}_w{index}p0"]
                finish_reason = ((exchange.response or {}).get("choices") or [{}])[0].get("finish_reason")
                if exchange.error or finish_reason == "length":
                    raise RuntimeError(f"Incomplete benchmark {role} call: {exchange.error or finish_reason}")
        completed_unix = time.time()
        with self.lock:
            if self.benchmark_slm_fixture is None:
                self.anonymised_windows.append(anonymised)
            else:
                self.benchmark_stress_outputs.append({
                    "window": index,
                    "fixture": str(selected_fixture),
                    "fixture_active": fixture_active,
                    "model_input": model_input,
                    "model_line_numbers": model_numbers,
                    "state_policy": self.args.benchmark_slm_state,
                    "anonymised": anonymised,
                })
            self.window_records.append({
                "window": index,
                "line_numbers": line_numbers,
                "model_line_numbers": model_numbers,
                "text_tokens": job["text_tokens"],
                "token_budget": self.token_budget,
                "final_partial_window": job["final_partial_window"],
                "original": job["original"],
                "anonymised": anonymised,
                "benchmark_slm_fixture": str(self.args.benchmark_slm_fixture)
                if self.benchmark_slm_fixture is not None else None,
                "benchmark_slm_selected_fixture": str(selected_fixture)
                if selected_fixture is not None else None,
                "benchmark_slm_fixture_active": fixture_active,
            })
        self.log_end_to_end_event({
            "stage": "slm_window_completed",
            "window": index,
            "benchmark_slm_fixture_active": fixture_active,
            "line_numbers": line_numbers,
            "latest_source_end_seconds": job["latest_source_end_seconds"],
            "latest_source_available_unix": job["latest_source_available_unix"],
            "enqueued_unix": job["enqueued_unix"],
            "completed_unix": completed_unix,
            "end_to_end_lag_seconds": (
                completed_unix - job["latest_source_available_unix"]
                if job["latest_source_available_unix"] is not None else None
            ),
        })
        self.write_outputs()

    # Attach original ASR timing to each surviving anonymized line without exposing raw text.
    def anonymised_timing_records(self) -> list[dict]:
        records = []
        for window in self.anonymised_windows:
            for raw_line in window.splitlines():
                match = NUMBERED_OUTPUT_RE.match(raw_line)
                if match is None:
                    continue
                line_number = int(match.group(1))
                source = self.raw_timing_records.get(line_number)
                if source is None:
                    continue
                text = match.group(2).strip()
                speaker_prefix = SPEAKER_PREFIX_RE.match(text)
                if speaker_prefix is not None:
                    text = speaker_prefix.group(1).strip()
                if not text:
                    continue
                records.append({
                    "line_number": line_number,
                    **source,
                    "text": text,
                })
        return records

    # Persist both user-facing transcripts and the full auditable SLM trace.
    def write_outputs(self) -> None:
        with self.lock:
            raw = "\n".join(self.raw_lines)
            anonymised = "\n".join(self.anonymised_windows)
            write_atomic(self.output_dir / "raw_transcript.txt", raw + ("\n" if raw else ""))
            write_atomic(
                self.output_dir / "anonymised_transcript.txt",
                anonymised + ("\n" if anonymised else ""),
            )
            anonymised_timing = self.anonymised_timing_records()
            write_atomic(
                self.output_dir / "anonymised_timestamped_lines.jsonl",
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in anonymised_timing),
            )
            if self.benchmark_stress_outputs:
                write_atomic(
                    self.output_dir / "benchmark_stress_outputs.jsonl",
                    "".join(
                        json.dumps(record, ensure_ascii=False) + "\n"
                        for record in self.benchmark_stress_outputs
                    ),
                )
            exchanges = {
                name: exchange.__dict__
                for name, exchange in self.processor.exchanges.items()
            }
            queue_depth, oldest_age = self.queue_state()
            trace = {
                "tokenizer": self.args.tokenizer,
                "window_lines": self.args.window_lines,
                "token_budget": self.token_budget,
                "raw_line_count": len(self.raw_lines),
                "processed_windows": len(self.window_records),
                "pending_line_numbers": self.pending_numbers,
                "pending_tokens": self.pending_tokens,
                "slm_queue_depth": queue_depth,
                "slm_oldest_window_age_seconds": oldest_age,
                "slm_priority_state": self.priority.state,
                "profile": self.processor.profile.records,
                "windows": self.window_records,
                "pipeline_trace": self.processor.trace,
                "errors": self.processor.errors,
                "exchanges": exchanges,
            }
            write_atomic(
                self.output_dir / "slm_trace.json",
                json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
            )


# Read every complete JSONL record not consumed during the previous poll.
def read_new_records(path: Path, consumed: int) -> tuple[list[dict], int]:
    if not path.is_file():
        return [], consumed
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    next_consumed = consumed
    for index in range(consumed, len(lines)):
        try:
            records.append(json.loads(lines[index]))
        except json.JSONDecodeError:
            break
        next_consumed = index + 1
    return records, next_consumed


# Assemble the existing source-separation deployment command with optimized models.
def source_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    if args.live_pcm:
        command = [
            str(args.source_python),
            str(SOURCE_STREAM),
            "--output", str(output_dir),
            "--input-sample-rate", str(args.input_sample_rate),
            "--prewarm-speakers", str(args.prewarm_speakers),
        ]
    else:
        command = [
            str(args.source_python),
            str(SOURCE_PIPELINE),
            "--input", str(args.input),
            "--output", str(output_dir),
        ]
    command.extend([
        "--chunk-seconds", str(args.chunk_seconds),
        "--overlap-collar-seconds", str(args.overlap_collar_seconds),
        "--maximum-overlap-seconds", str(args.maximum_overlap_seconds),
        "--lseend-threads", str(args.lseend_threads),
        "--separation-threads", str(args.separation_threads),
        "--speaker-encoder-threads", str(args.speaker_encoder_threads),
        "--overlap-speaker-encoder-threads", str(args.overlap_speaker_encoder_threads),
        "--asr-threads", str(args.asr_threads),
        "--asr-backend", args.asr_backend,
        "--moonshine-model", str(args.moonshine_model),
        "--moonshine-segment-seconds", str(args.moonshine_segment_seconds),
        "--moonshine-max-tokens", str(args.moonshine_max_tokens),
        "--lseend-model", str(args.lseend_model),
        "--lseend-metadata", str(args.lseend_metadata),
        "--convtasnet-2", str(args.convtasnet_2),
        "--convtasnet-3", str(args.convtasnet_3),
        "--convtasnet-2-onnx", str(args.convtasnet_2_onnx),
        "--convtasnet-3-onnx", str(args.convtasnet_3_onnx),
        "--separator-backend", args.separator_backend,
        "--speaker-encoder-model", str(args.speaker_encoder_model),
        "--nemotron-encoder", str(args.nemotron_encoder),
        "--nemotron-decoder", str(args.nemotron_decoder),
        "--nemotron-joiner", str(args.nemotron_joiner),
        "--nemotron-tokens", str(args.nemotron_tokens),
    ])
    if args.separator_backend == "sandglasset":
        command.extend([
            "--sandglasset-2-onnx", str(args.sandglasset_2_onnx),
            "--sandglasset-3-onnx", str(args.sandglasset_3_onnx),
        ])
    if args.benchmark_force_speakers is not None:
        command.extend(["--always-separate-speakers", str(args.benchmark_force_speakers)])
    for track in args.benchmark_clean_track:
        command.extend(["--benchmark-clean-track", str(track)])
    if args.benchmark_report is not None:
        command.extend([
            "--benchmark-report", str(output_dir / "audio_benchmark.json"),
            "--detailed-timing-log", str(output_dir / "audio_timeline.json"),
        ])
    if args.pace and not args.live_pcm:
        command.append("--pace")
    if args.semi_paced:
        command.append("--semi-paced")
    if args.no_agc:
        command.append("--no-agc")
    if args.asr_batch_ready:
        command.append("--asr-batch-ready")
    if args.speaker_encoder_batch_overlap:
        command.append("--speaker-encoder-batch-overlap")
    if args.benchmark_warmup_slm:
        command.append("--warmup-models")
    return command


# Create the device/TTP handoff only after every SLM window has been anonymized.
def protect_output(args: argparse.Namespace, output_dir: Path, encoder, prepared) -> None:
    protection_args = argparse.Namespace(
        source_output=output_dir,
        output=args.protected_output,
        env_file=args.crypto_env_file,
        ttp_public_key=args.ttp_public_key,
        fingerprint_model=args.fingerprint_model,
        fingerprint_threads=args.fingerprint_threads,
    )
    print("Creating encrypted device/TTP handoff", flush=True)
    protect_tracks_run(protection_args, encoder=encoder, prepared=prepared)


# Derive privacy-safe word timings before device-visible artifacts are copied.
def generate_anonymised_word_timings(output_dir: Path) -> None:
    command = [
        sys.executable,
        str(WORD_TIMING_ANONYMISER),
        "--source-output", str(output_dir),
    ]
    print("Aligning anonymized words to Nemotron timings", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


# Keep a small machine-readable status document for the browser and CLI users.
def write_status(output_dir: Path, phase: str, anonymiser: TranscriptAnonymiser | None,
                 source_running: bool, error: str | None = None) -> None:
    source_status = None
    source_status_path = output_dir / "source_status.json"
    if source_status_path.is_file():
        try:
            source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if anonymiser is not None:
        anonymiser.observe_queue()
        queue_depth, oldest_age = anonymiser.queue_state()
    else:
        queue_depth, oldest_age = 0, None
    payload = {
        "phase": phase,
        "source_pipeline_running": source_running,
        "raw_lines": len(anonymiser.raw_lines) if anonymiser else 0,
        "processed_windows": len(anonymiser.window_records) if anonymiser else 0,
        "pending_tokens": anonymiser.pending_tokens if anonymiser else 0,
        "token_budget": anonymiser.token_budget if anonymiser else None,
        "slm_queue_depth": queue_depth,
        "slm_oldest_window_age_seconds": oldest_age,
        "slm_priority_state": anonymiser.priority.state if anonymiser else None,
        "source": source_status,
        "error": error,
    }
    write_atomic(
        output_dir / "orchestrator_status.json",
        json.dumps(payload, indent=2) + "\n",
    )


# Run the source pipeline and consume committed ASR lines while it is still active.
def run(args: argparse.Namespace) -> None:
    startup_started = time.perf_counter()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    server = LlamaServer(args, output_dir)
    anonymiser = None
    source_process = None
    fingerprint_encoder = None
    fingerprint_thread = None
    fingerprint_prepared = None
    fingerprint_error = None
    benchmark_started = None
    benchmark_started_unix = None
    try:
        write_status(output_dir, "starting_llama_server", None, False)
        server.start()
        if (args.slm_priority_controller and args.slm_priority_mode == "affinity"
                and args.slm_priority_pid is None and server.process is not None):
            args.slm_priority_pid = server.process.pid
        write_status(output_dir, "loading_slm_context", None, False)
        anonymiser = TranscriptAnonymiser(args, output_dir)
        anonymiser.write_outputs()
        if not args.skip_protection:
            write_status(output_dir, "preloading_final_fingerprint_model", anonymiser, False)
            fingerprint_encoder = ReDimNetEncoder(
                args.fingerprint_model,
                cpu_threads=args.fingerprint_threads,
                onnx_optimization="all",
                memory_pattern=False,
                allow_spinning=False,
            )
        if args.benchmark_warmup_slm:
            write_status(output_dir, "warming_slm", anonymiser, False)
            anonymiser.warmup()

        command = source_command(args, output_dir)
        source_log_path = output_dir / "source_pipeline.log"
        source_log = source_log_path.open("w", encoding="utf-8")
        source_log.write("command=" + " ".join(command) + "\n")
        source_log.flush()
        source_process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=sys.stdin.buffer if args.live_pcm else None,
            stdout=source_log,
            stderr=subprocess.STDOUT,
        )
        source_log.close()

        # The source publishes this after its model setup, immediately before capture starts.
        clock_path = output_dir / "source_clock.json"
        while not clock_path.exists():
            if source_process.poll() is not None:
                raise RuntimeError("Source exited before publishing its capture clock; see source_pipeline.log")
            time.sleep(args.poll_seconds)
        source_clock = json.loads(clock_path.read_text(encoding="utf-8"))
        benchmark_started = source_clock["started_monotonic"]
        benchmark_started_unix = source_clock["started_unix"]
        anonymiser.set_input_started_unix(benchmark_started_unix)
        anonymiser.log_end_to_end_event({"stage": "source_capture_started", **source_clock})

        transcript_path = output_dir / "timestamped_lines.jsonl"
        consumed = 0
        while source_process.poll() is None:
            records, consumed = read_new_records(transcript_path, consumed)
            for record in records:
                anonymiser.add_record(record)
            anonymiser.flush_if_due()
            if args.semi_paced:
                request = read_state(output_dir / "semi_paced_request.json")
                approval_path = output_dir / "semi_paced_approval.json"
                previous = read_state(approval_path)
                if (request and not request.get("expired")
                        and previous.get("chunk") != request.get("chunk")):
                    depth, _ = anonymiser.queue_state()
                    if depth or request.get("audio_idle"):
                        # Transcript records were consumed above. Queued jobs
                        # count as busy even before the serial worker starts.
                        write_state(approval_path, {
                            "chunk": request["chunk"], "skip": depth == 0,
                        })
            write_status(output_dir, "streaming", anonymiser, True)
            time.sleep(args.poll_seconds)

        records, consumed = read_new_records(transcript_path, consumed)
        for record in records:
            anonymiser.add_record(record)
        if source_process.returncode != 0:
            tail = source_log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"source pipeline exited with {source_process.returncode}:\n{tail}")

        # Source tracks are final now. Run B6 during serial SLM draining instead
        # of adding its work to the post-drain completion latency.
        if fingerprint_encoder is not None:
            def prepare_final_fingerprints() -> None:
                nonlocal fingerprint_prepared, fingerprint_error
                try:
                    fingerprint_prepared = prepare_fingerprints(output_dir, fingerprint_encoder)
                except Exception as error:
                    fingerprint_error = error
            fingerprint_thread = threading.Thread(
                target=prepare_final_fingerprints,
                name="final-fingerprint-b6",
            )
            fingerprint_thread.start()

        write_status(output_dir, "flushing_final_window", anonymiser, False)
        anonymiser.flush_window(final=True)
        write_status(output_dir, "waiting_for_slm_queue", anonymiser, False)
        anonymiser.finish()
        anonymiser.write_outputs()
        write_status(output_dir, "aligning_anonymised_word_timings", anonymiser, False)
        if args.benchmark_slm_fixture is None:
            generate_anonymised_word_timings(output_dir)
        if not args.skip_protection:
            write_status(output_dir, "protecting_tracks", anonymiser, False)
            if fingerprint_thread is not None:
                fingerprint_thread.join()
            if fingerprint_error is not None:
                raise RuntimeError(f"Final ReDimNet2-B6 fingerprinting failed: {fingerprint_error}")
            protect_output(args, output_dir, fingerprint_encoder, fingerprint_prepared)
        write_status(output_dir, "complete", anonymiser, False)
        if args.benchmark_report is not None:
            elapsed = time.perf_counter() - benchmark_started
            skips = skipped_intervals(output_dir) if args.semi_paced else []
            removed = sum(event["skipped_seconds"] for event in skips)
            paced_elapsed = elapsed + removed
            duration = audio_duration_seconds(args.input)
            synthetic_records = [
                record for record in anonymiser.window_records
                if record["benchmark_slm_fixture_active"] is not None
            ]
            active_synthetic_windows = sum(
                record["benchmark_slm_fixture_active"] is True
                for record in synthetic_records
            )
            report = {
                "audio_duration_seconds": round(duration, 6),
                "elapsed_seconds": round(paced_elapsed if args.semi_paced else elapsed, 6),
                "started_unix": benchmark_started_unix,
                "started_monotonic": benchmark_started,
                "startup_seconds": round(benchmark_started - startup_started, 6),
                "final_drain_seconds": round((paced_elapsed if args.semi_paced else elapsed) - duration, 6),
                "throughput_audio_seconds_per_wall_second": round(duration / elapsed, 6),
                "rtf": round((paced_elapsed if args.semi_paced else elapsed) / duration, 6),
                "metric_clock": "reconstructed_paced" if args.semi_paced else "measured_wall",
                "measured_wall_rtf": round(elapsed / duration, 6),
                "evaluation_mode": "semi-paced" if args.semi_paced else ("paced" if args.pace else "unpaced"),
                "measured_wall_seconds": round(elapsed, 6),
                "skipped_idle_seconds": round(removed, 6),
                "reconstructed_paced_seconds": round(paced_elapsed, 6) if args.semi_paced else None,
                "reconstructed_paced_rtf": round(paced_elapsed / duration, 6) if args.semi_paced else None,
                "reconstructed_paced_throughput": round(duration / paced_elapsed, 6) if args.semi_paced else None,
                "reconstructed_final_drain_seconds": round(max(0.0, paced_elapsed - duration), 6) if args.semi_paced else None,
                "slm_windows": len(anonymiser.window_records),
                "raw_transcript_lines": len(anonymiser.raw_lines),
                "source_command": command,
                "llama_threads": args.llama_threads,
                "llama_batch_threads": args.llama_batch_threads,
                "slm_nice": args.slm_nice,
                "slm_scheduling": {
                    "mechanism": "nice" if not args.slm_scope_unit else "systemd_cpu_weight",
                    "priority_class": "background" if args.slm_nice > 0 else "normal",
                    "cpu_weight": args.slm_cpu_weight if args.slm_scope_unit else None,
                },
                "benchmark_force_speakers": args.benchmark_force_speakers,
                "benchmark_clean_tracks": [str(path) for path in args.benchmark_clean_track],
                "benchmark_slm_fixture_sha256": (
                    hashlib.sha256(args.benchmark_slm_fixture.read_bytes()).hexdigest()
                    if args.benchmark_slm_fixture is not None else None
                ),
                "benchmark_slm_empty_fixture_sha256": (
                    hashlib.sha256(args.benchmark_slm_empty_fixture.read_bytes()).hexdigest()
                    if args.benchmark_slm_empty_fixture is not None else None
                ),
                "benchmark_slm_active_fraction_target": args.benchmark_slm_active_fraction,
                "benchmark_slm_active_windows": active_synthetic_windows,
                "benchmark_slm_empty_windows": len(synthetic_records) - active_synthetic_windows,
                "benchmark_slm_active_fraction_realized": (
                    active_synthetic_windows / len(synthetic_records)
                    if synthetic_records else None
                ),
                "benchmark_slm_state": args.benchmark_slm_state,
                "slm_max_buffer_seconds": args.slm_max_buffer_seconds,
                "embed_model": args.embed_model,
                "embed_threads": args.embed_threads,
                "asr_batch_ready": args.asr_batch_ready,
                "speaker_encoder_batch_overlap": args.speaker_encoder_batch_overlap,
                "model_sha256": {
                    name: file_sha256(path) if path.is_file() else None for name, path in {
                        "slm": args.slm_model,
                        "lseend": args.lseend_model,
                        "lseend_metadata": args.lseend_metadata,
                        "convtasnet_2": args.convtasnet_2_onnx,
                        "convtasnet_3": args.convtasnet_3_onnx,
                        "speaker_encoder_b1": args.speaker_encoder_model,
                        "fingerprint_redimnet2_b6": args.fingerprint_model,
                        "nemotron_encoder": args.nemotron_encoder,
                        "nemotron_decoder": args.nemotron_decoder,
                        "nemotron_joiner": args.nemotron_joiner,
                        "nemotron_tokens": args.nemotron_tokens,
                    }.items()
                },
            }
            write_atomic(args.benchmark_report, json.dumps(report, indent=2) + "\n")
            if args.semi_paced:
                reconstruct_audio_timeline(output_dir, benchmark_started_unix, skips)
        print(f"Raw transcript: {output_dir / 'raw_transcript.txt'}", flush=True)
        print(f"Anonymised transcript: {output_dir / 'anonymised_transcript.txt'}", flush=True)
        if not args.skip_protection:
            print(f"Encrypted handoff: {args.protected_output}", flush=True)
    except Exception as error:
        write_status(output_dir, "failed", anonymiser, bool(source_process and source_process.poll() is None), str(error))
        if source_process is not None and source_process.poll() is None:
            source_process.terminate()
        raise
    finally:
        if args.stop_server_after_run:
            server.stop()


# Parse deployment, SLM, and runtime settings in one discoverable command.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live source separation, ASR, and stateful SLM anonymisation."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--live-pcm", action="store_true",
                             help="Read mono little-endian PCM16 continuously from stdin.")
    parser.add_argument("--input-sample-rate", type=int, default=48000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pace", action="store_true")
    parser.add_argument("--semi-paced", action="store_true",
                        help="Skip fully idle arrival waits; keep real pacing in SLM-active intervals. Logs contain measured and reconstructed clocks.")
    parser.add_argument("--no-agc", action="store_true")
    parser.add_argument("--source-python", type=Path, default=Path(os.getenv("SOURCE_SEPARATION_PYTHON", UI_PYTHON)))
    parser.add_argument("--chunk-seconds", type=float, default=5.0)
    parser.add_argument("--overlap-collar-seconds", type=float, default=0.0)
    parser.add_argument("--maximum-overlap-seconds", type=float, default=3.0)
    parser.add_argument("--lseend-threads", type=int, default=1)
    parser.add_argument("--separation-threads", type=int, default=2)
    parser.add_argument("--speaker-encoder-threads", "--campplus-threads", dest="speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--overlap-speaker-encoder-threads", "--overlap-cam-threads", dest="overlap_speaker_encoder_threads", type=int, default=1)
    parser.add_argument("--asr-threads", type=int, default=3)
    parser.add_argument("--asr-backend", choices=("moonshine", "nemotron"), default="moonshine")
    parser.add_argument("--moonshine-model", type=Path,
                        default=RAW_MODELS / "moonshine_streaming_medium_onnx")
    parser.add_argument("--moonshine-segment-seconds", type=float, default=10.0)
    parser.add_argument("--moonshine-max-tokens", type=int, default=256)
    parser.add_argument("--asr-batch-ready", action="store_true")
    parser.add_argument("--speaker-encoder-batch-overlap", "--cam-batch-overlap", dest="speaker_encoder_batch_overlap", action="store_true")
    parser.add_argument("--prewarm-speakers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--lseend-model", type=Path,
                        default=DEPLOYMENT_MODELS / "lseend" / "ls_eend_dih3_step_ort_optimized.onnx")
    parser.add_argument("--lseend-metadata", type=Path,
                        default=DEPLOYMENT_MODELS / "lseend" / "ls_eend_dih3_step.json")
    parser.add_argument("--convtasnet-2", type=Path,
                        default=RAW_MODELS / "convtasnet_2" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-3", type=Path,
                        default=RAW_MODELS / "convtasnet_3" / "pytorch_model.bin")
    parser.add_argument("--convtasnet-2-onnx", type=Path,
                        default=DEPLOYMENT_MODELS / "convtasnet_2" / "convtasnet_2_sepnoisy_ort_optimized.onnx")
    parser.add_argument("--convtasnet-3-onnx", type=Path,
                        default=DEPLOYMENT_MODELS / "convtasnet_3" / "convtasnet_3_sepnoisy_ort_optimized.onnx")
    parser.add_argument("--separator-backend", choices=("convtasnet", "sandglasset"), default="convtasnet")
    parser.add_argument("--sandglasset-2-onnx", type=Path,
                        default=PROJECT_ROOT / "models" / "deployment" / "sandglasset_2spk_5s_ort_optimized.onnx")
    parser.add_argument("--sandglasset-3-onnx", type=Path,
                        default=PROJECT_ROOT / "models" / "deployment" / "sandglasset_3spk_5s_ort_optimized.onnx")
    parser.add_argument("--speaker-encoder-model", "--campplus-model", dest="speaker_encoder_model", type=Path,
                        default=DEPLOYMENT_MODELS / "speaker_encoder_b1" / "redimnet_b1_5s_ort_enable_all.onnx")
    parser.add_argument("--nemotron-encoder", type=Path,
                        default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "encoder.int8.onnx")
    parser.add_argument("--nemotron-decoder", type=Path,
                        default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "decoder.int8.onnx")
    parser.add_argument("--nemotron-joiner", type=Path,
                        default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "joiner.int8.onnx")
    parser.add_argument("--nemotron-tokens", type=Path,
                        default=DEPLOYMENT_MODELS / "nemotron_1120ms_int8" / "tokens.txt")

    parser.add_argument("--window-lines", type=int, default=12)
    parser.add_argument("--passes-per-window", type=int, default=1)
    parser.add_argument("--rag-candidates", type=int, default=20)
    parser.add_argument("--rag-top-k", type=int, default=8)
    parser.add_argument("--rag-threshold", type=float, default=0.774)
    parser.add_argument("--context-lines", type=int, default=2)
    parser.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--embed-threads", type=int, default=1,
                        help="CPU threads for Nomic per-line RAG embeddings.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--stream-slm-tokens", action="store_true")

    parser.add_argument("--slm-model", type=Path, default=SLM_MODEL)
    parser.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server")
    parser.add_argument("--llama-host", default="127.0.0.1")
    parser.add_argument("--llama-port", type=int, default=8080)
    parser.add_argument("--llama-total-context", type=int, default=65536,
                        help="Total KV context across two slots; 65536 gives 32768 per role.")
    parser.add_argument("--llama-threads", type=int, default=4)
    parser.add_argument("--llama-batch-threads", type=int, default=10)
    parser.add_argument("--slm-cpu-weight", type=int, default=1)
    parser.add_argument("--slm-nice", type=int, default=19)
    parser.add_argument("--llama-start-timeout", type=float, default=120.0)
    parser.add_argument("--stop-server-after-run", action="store_true")
    parser.add_argument("--slm-scope-unit",
                        help="Transient systemd scope for llama-server, such as slm-qwen.scope.")
    parser.add_argument("--slm-priority-controller", action="store_true",
                        help="Use the two-watermark policy to reserve or promote SLM CPU resources.")
    parser.add_argument("--slm-priority-mode", choices=("systemd", "affinity"), default="affinity",
                        help="Use systemd CPUWeight or no-sudo Linux CPU affinity control.")
    parser.add_argument("--slm-priority-use-sudo", action="store_true",
                        help="Use passwordless sudo for a root-owned Pi SLM scope.")
    parser.add_argument("--slm-priority-pid", type=int,
                        help="User-owned llama-server PID for affinity mode; inferred when started here.")
    parser.add_argument("--slm-background-cpus", default="0",
                        help="SLM CPU set while the queue is below the watermark.")
    parser.add_argument("--slm-promoted-cpus", default="0-3",
                        help="SLM CPU set while the queue is promoted.")
    parser.add_argument("--slm-high-watermark", type=int, default=8)
    parser.add_argument("--slm-low-watermark", type=int, default=3)
    parser.add_argument("--slm-max-wait-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--slm-max-buffer-seconds", type=float, default=0.0,
                        help="Optional maximum wait of committed text before partial-window dispatch; zero disables it.")
    parser.add_argument("--benchmark-force-speakers", type=int, choices=(1, 2, 3),
                        help="Benchmark-only continuous routing mode.")
    parser.add_argument("--benchmark-clean-track", type=Path, action="append", default=[],
                        help="Benchmark-only aligned clean track passed downstream after separation.")
    parser.add_argument("--benchmark-report", type=Path,
                        help="Write end-to-end capacity metrics after successful anonymisation.")
    parser.add_argument("--benchmark-warmup-slm", action="store_true",
                        help="Warm both persistent SLM role slots before the benchmark timer starts.")
    parser.add_argument("--benchmark-slm-fixture", type=Path,
                        help="Benchmark-only synthetic SLM input; its outputs are saved separately.")
    parser.add_argument("--benchmark-slm-empty-fixture", type=Path,
                        help="Benign synthetic input used for inactive SFT-like benchmark windows.")
    parser.add_argument("--benchmark-slm-active-fraction", type=float, default=1.0,
                        help="Deterministic fraction of benchmark SLM windows using the stress fixture.")
    parser.add_argument("--benchmark-slm-state", choices=("carry", "fresh"), default="carry",
                        help="Carry disclosure state, or reset it for a repeated independent synthetic stress exchange.")
    parser.add_argument("--skip-protection", action="store_true",
                        help="Do not create the device/TTP encrypted handoff after anonymisation.")
    parser.add_argument("--protected-output", type=Path,
                        help="Defaults to outputs/bundles/<orchestrator output directory name>.")
    parser.add_argument("--crypto-env-file", type=Path, default=PROJECT_ROOT / ".env",
                        help="TTP public key environment file used by the protection pass.")
    parser.add_argument("--ttp-public-key", type=Path,
                        help="Optional PEM override for the TTP public key.")
    parser.add_argument("--fingerprint-model", type=Path,
                        default=DEPLOYMENT_MODELS / "fingerprint_redimnet2_b6" / "redimnet2_b6_5s_ort_enable_all.onnx")
    parser.add_argument("--fingerprint-threads", type=int, default=4)
    args = parser.parse_args()

    if args.input is not None:
        args.input = args.input.resolve()
    if args.output is None:
        if args.live_pcm:
            parser.error("--output is required with --live-pcm")
        args.output = OUTPUT_ROOT / args.input.stem
    args.output = args.output.resolve()
    args.slm_model = args.slm_model.resolve()
    for name in ("lseend_model", "lseend_metadata", "convtasnet_2", "convtasnet_3",
                 "convtasnet_2_onnx", "convtasnet_3_onnx", "sandglasset_2_onnx", "sandglasset_3_onnx",
                 "speaker_encoder_model", "fingerprint_model",
                 "nemotron_encoder", "nemotron_decoder", "nemotron_joiner", "nemotron_tokens", "moonshine_model"):
        setattr(args, name, getattr(args, name).resolve())
    if args.protected_output is None:
        args.protected_output = OUTPUT_BUNDLE_ROOT / args.output.name
    args.protected_output = args.protected_output.resolve()
    args.crypto_env_file = args.crypto_env_file.resolve()
    if args.ttp_public_key is not None:
        args.ttp_public_key = args.ttp_public_key.resolve()
    args.benchmark_clean_track = [path.resolve() for path in args.benchmark_clean_track]
    if args.benchmark_report is not None:
        args.benchmark_report = args.benchmark_report.resolve()
    if args.benchmark_slm_fixture is not None:
        args.benchmark_slm_fixture = args.benchmark_slm_fixture.resolve()
        if not args.benchmark_slm_fixture.is_file():
            parser.error("Missing --benchmark-slm-fixture: " + str(args.benchmark_slm_fixture))
        if not args.skip_protection:
            parser.error("Synthetic SLM benchmarking requires --skip-protection")
    if args.benchmark_slm_empty_fixture is not None:
        args.benchmark_slm_empty_fixture = args.benchmark_slm_empty_fixture.resolve()
        if not args.benchmark_slm_empty_fixture.is_file():
            parser.error(
                "Missing --benchmark-slm-empty-fixture: "
                + str(args.benchmark_slm_empty_fixture)
            )
    if not 0.0 <= args.benchmark_slm_active_fraction <= 1.0:
        parser.error("--benchmark-slm-active-fraction must be in [0, 1]")
    if args.benchmark_slm_active_fraction < 1.0:
        if args.benchmark_slm_fixture is None:
            parser.error("A partial active fraction requires --benchmark-slm-fixture")
        if args.benchmark_slm_empty_fixture is None:
            parser.error("A partial active fraction requires --benchmark-slm-empty-fixture")
    if args.slm_max_buffer_seconds < 0:
        parser.error("--slm-max-buffer-seconds cannot be negative")
    if args.semi_paced and (args.live_pcm or args.pace):
        parser.error("--semi-paced is a file replay mode and replaces --pace")
    if args.semi_paced and args.slm_max_buffer_seconds:
        parser.error("--semi-paced currently requires token-budget dispatch (--slm-max-buffer-seconds 0)")
    if args.embed_threads < 1:
        parser.error("--embed-threads must be positive")
    if not args.source_python.is_absolute():
        args.source_python = (PROJECT_ROOT / args.source_python).absolute()
    args.llama_base_url = f"http://{args.llama_host}:{args.llama_port}/v1"

    if args.input is not None and not args.input.is_file():
        parser.error(f"Input audio does not exist: {args.input}")
    if args.live_pcm and args.input_sample_rate < 8000:
        parser.error("--input-sample-rate must be at least 8000")
    if not args.source_python.is_file():
        parser.error(f"Source pipeline Python does not exist: {args.source_python}")
    if args.window_lines < 1 or args.max_tokens < 1:
        parser.error("window-lines and max-tokens must be positive")
    if args.poll_seconds <= 0:
        parser.error("poll-seconds must be positive")
    if args.slm_low_watermark >= args.slm_high_watermark:
        parser.error("--slm-low-watermark must be smaller than --slm-high-watermark")
    if args.slm_max_wait_seconds <= 0:
        parser.error("--slm-max-wait-seconds must be positive")
    if not 1 <= args.slm_cpu_weight <= 10000:
        parser.error("--slm-cpu-weight must be in [1, 10000]")
    if not -20 <= args.slm_nice <= 19:
        parser.error("--slm-nice must be in [-20, 19]")
    if args.benchmark_clean_track:
        if args.benchmark_force_speakers is None:
            parser.error("--benchmark-clean-track requires --benchmark-force-speakers")
        if len(args.benchmark_clean_track) != args.benchmark_force_speakers:
            parser.error("Provide one --benchmark-clean-track per forced speaker")
        missing_tracks = [str(path) for path in args.benchmark_clean_track if not path.is_file()]
        if missing_tracks:
            parser.error("Missing benchmark clean tracks:\n" + "\n".join(missing_tracks))
    if args.slm_priority_controller and args.slm_priority_mode == "systemd" and not args.slm_scope_unit:
        parser.error("systemd SLM priority control requires --slm-scope-unit")
    if args.slm_priority_controller and args.slm_priority_mode == "affinity":
        try:
            SlmPriorityController.parse_cpus(args.slm_background_cpus)
            SlmPriorityController.parse_cpus(args.slm_promoted_cpus)
        except ValueError as error:
            parser.error(str(error))
    if args.fingerprint_threads < 1:
        parser.error("fingerprint-threads must be positive")
    if not WORD_TIMING_ANONYMISER.is_file():
        parser.error(f"Word-timing anonymizer is missing: {WORD_TIMING_ANONYMISER}")
    if not args.skip_protection:
        if not CRYPTO_PROTECTOR.is_file():
            parser.error(f"Crypto protector is missing: {CRYPTO_PROTECTOR}")
        if args.ttp_public_key is not None and not args.ttp_public_key.is_file():
            parser.error("--ttp-public-key must exist when supplied")
        if args.ttp_public_key is None and not args.crypto_env_file.is_file():
            parser.error("Protection requires --ttp-public-key or a .env file with TTP_RSA_PUBLIC_KEY_B64")
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
