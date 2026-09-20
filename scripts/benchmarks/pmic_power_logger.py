#!/usr/bin/env python3
"""Low-rate Raspberry Pi 5 PMIC rail-power logger.

The aggregate is the sum of V*I for rails that expose both values through
``vcgencmd pmic_read_adc``. Voltage-only rails, notably EXT5V, are retained in
the raw command output boundary but are deliberately not added to the sum.
"""

from __future__ import annotations

import argparse
import csv
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path


PATTERN = re.compile(r"^\s*(.+?)_(A|V)\s+(?:current|volt)\(\d+\)=([0-9.]+)")
STOP = False


def stop_handler(_signum, _frame) -> None:
    global STOP
    STOP = True


def telemetry() -> tuple[dict[str, float], float, float]:
    raw = subprocess.check_output(["/usr/bin/vcgencmd", "pmic_read_adc"], text=True)
    rails: dict[str, dict[str, float]] = {}
    for line in raw.splitlines():
        match = PATTERN.match(line)
        if match:
            rails.setdefault(match.group(1).strip(), {})[match.group(2)] = float(match.group(3))
    watts = {
        name: values["A"] * values["V"]
        for name, values in rails.items()
        if "A" in values and "V" in values
    }
    return watts, sum(watts.values()), watts.get("VDD_CORE", 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=0,
                        help="Stop after N samples; zero logs until SIGTERM/SIGINT.")
    args = parser.parse_args()
    if args.rate_hz <= 0:
        parser.error("--rate-hz must be positive")
    if args.samples < 0:
        parser.error("--samples cannot be negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    period = 1.0 / args.rate_hz
    deadline = time.monotonic()
    count = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "unix_seconds", "monotonic_seconds", "aggregate_pmic_w", "vdd_core_w",
            "sample_duration_seconds", "rail_watts",
        ])
        writer.writeheader()
        while not STOP and (not args.samples or count < args.samples):
            started_monotonic = time.monotonic()
            watts, aggregate, core = telemetry()
            finished_monotonic = time.monotonic()
            writer.writerow({
                "unix_seconds": f"{time.time():.6f}",
                "monotonic_seconds": f"{started_monotonic:.6f}",
                "aggregate_pmic_w": f"{aggregate:.8f}",
                "vdd_core_w": f"{core:.8f}",
                "sample_duration_seconds": f"{finished_monotonic - started_monotonic:.6f}",
                "rail_watts": ";".join(f"{name}={value:.8f}" for name, value in sorted(watts.items())),
            })
            stream.flush()
            count += 1
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()

    durations = []
    with args.output.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            durations.append(float(row["sample_duration_seconds"]))
    if durations:
        print(
            f"wrote {len(durations)} samples to {args.output}; "
            f"mean command duration {statistics.mean(durations) * 1000:.1f} ms"
        )


if __name__ == "__main__":
    main()
