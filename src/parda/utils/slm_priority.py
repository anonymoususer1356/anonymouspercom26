#!/usr/bin/env python3
# Apply a two-watermark, timeout-backed resource policy to the local SLM.

import json
import os
import subprocess
import time
from pathlib import Path


class SlmPriorityController:
    # Keep the SLM in the background unless the queued work needs to catch up.
    def __init__(self, enabled: bool, mode: str, scope_unit: str | None,
                 process_id: int | None, background_cpus: str, promoted_cpus: str,
                 high_watermark: int,
                 low_watermark: int, maximum_wait_seconds: float, log_path: Path,
                 use_sudo: bool = False):
        if low_watermark >= high_watermark:
            raise ValueError("SLM low watermark must be smaller than the high watermark")
        if maximum_wait_seconds <= 0:
            raise ValueError("SLM maximum wait must be positive")
        if enabled and mode == "systemd" and not scope_unit:
            raise ValueError("systemd priority control requires --slm-scope-unit")
        if enabled and mode == "affinity" and process_id is None:
            raise ValueError("affinity priority control requires --slm-priority-pid")
        self.enabled = enabled
        self.mode = mode
        self.scope_unit = scope_unit
        self.process_id = process_id
        self.background_cpus = self.parse_cpus(background_cpus)
        self.promoted_cpus = self.parse_cpus(promoted_cpus)
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self.maximum_wait_seconds = maximum_wait_seconds
        self.log_path = log_path
        self.use_sudo = use_sudo
        self.state = "background"
        if self.enabled and self.mode == "affinity":
            try:
                os.sched_setaffinity(self.process_id, self.background_cpus)
            except OSError as error:
                raise RuntimeError(f"Could not set initial SLM CPU affinity: {error}") from error

    # Parse a portable CPU-set argument such as "0" or "0-3,5".
    @staticmethod
    def parse_cpus(value: str) -> set[int]:
        cpus = set()
        for part in value.split(","):
            start, separator, end = part.strip().partition("-")
            if not start:
                raise ValueError("CPU sets cannot contain an empty item")
            if separator:
                if not end:
                    raise ValueError(f"Invalid CPU range: {part}")
                cpus.update(range(int(start), int(end) + 1))
            else:
                cpus.add(int(start))
        if not cpus or min(cpus) < 0:
            raise ValueError("CPU sets must contain non-negative CPU indexes")
        return cpus

    # Append one state sample so queue behaviour is auditable after a run.
    def record(self, depth: int, oldest_age_seconds: float | None,
               reason: str | None = None) -> None:
        record = {
            "time_unix": time.time(),
            "queue_depth": depth,
            "oldest_age_seconds": oldest_age_seconds,
            "state": self.state,
            "mode": self.mode,
            "process_id": self.process_id,
            "high_watermark": self.high_watermark,
            "low_watermark": self.low_watermark,
            "maximum_wait_seconds": self.maximum_wait_seconds,
            "reason": reason,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    # Return the desired state, preserving hysteresis while the queue drains.
    def desired_state(self, depth: int, oldest_age_seconds: float | None) -> tuple[str, str | None]:
        timed_out = oldest_age_seconds is not None and oldest_age_seconds >= self.maximum_wait_seconds
        if self.state == "background" and (depth > self.high_watermark or timed_out):
            return "promoted", "timeout" if timed_out else "high_watermark"
        if self.state == "promoted" and depth < self.low_watermark and not timed_out:
            return "background", "low_watermark"
        return self.state, None

    # Change the SLM allocation only when the desired state changes.
    def observe(self, depth: int, oldest_age_seconds: float | None) -> None:
        desired, reason = self.desired_state(depth, oldest_age_seconds)
        if desired == self.state:
            self.record(depth, oldest_age_seconds)
            return
        if self.enabled:
            if self.mode == "affinity":
                # Linux permits a process owner to alter its own process affinity.
                # This avoids sudo and reserves the remaining cores for audio work.
                cpus = self.promoted_cpus if desired == "promoted" else self.background_cpus
                try:
                    os.sched_setaffinity(self.process_id, cpus)
                except OSError as error:
                    self.record(depth, oldest_age_seconds, f"priority_change_failed: {error}")
                    raise RuntimeError(f"Could not update SLM CPU affinity: {error}") from error
            else:
                # On this Pi, systemd accepts CPUWeight at runtime but rejects Nice
                # as a scope property. Nice=19 is applied when the scope starts.
                properties = ("CPUWeight=100",) if desired == "promoted" else ("CPUWeight=1",)
                command = ["systemctl", "set-property", self.scope_unit, *properties]
                if self.use_sudo:
                    command = ["sudo", "-n", *command]
                result = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode:
                    self.record(depth, oldest_age_seconds, "priority_change_failed: " + result.stdout.strip())
                    raise RuntimeError("Could not update SLM priority: " + result.stdout.strip())
        self.state = desired
        self.record(depth, oldest_age_seconds, reason)
