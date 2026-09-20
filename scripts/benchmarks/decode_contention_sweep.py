#!/usr/bin/env python3
"""Run Moonshine pipeline thread allocations under fixed SLM decode load."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def wait_http(url: str, process: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited before becoming ready")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("llama-server did not become ready")


def terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def nice_preexec(value: int):
    def apply() -> None:
        os.setpriority(os.PRIO_PROCESS, 0, value)
    return apply


def read_temperature() -> float | None:
    try:
        output = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, timeout=3)
        return float(output.split("=", 1)[1].split("'", 1)[0])
    except Exception:
        return None


def wait_temperature(maximum: float) -> float | None:
    while True:
        value = read_temperature()
        if value is None or value <= maximum:
            return value
        time.sleep(5)


def power_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    watts = [float(row["aggregate_pmic_w"]) for row in rows]
    stamps = [float(row["unix_seconds"]) for row in rows]
    ordered = sorted(watts)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    energy = sum(
        (watts[index - 1] + watts[index]) * 0.5 * (stamps[index] - stamps[index - 1])
        for index in range(1, len(rows))
    )
    return {
        "samples": len(rows), "duration_seconds": stamps[-1] - stamps[0],
        "mean_w": sum(watts) / len(watts), "p95_w": p95,
        "peak_w": max(watts), "energy_j": energy,
    }


def configurations(args: argparse.Namespace) -> list[dict]:
    if args.configurations_file:
        return json.loads(args.configurations_file.read_text(encoding="utf-8"))
    result = []
    for speakers in args.speakers:
        if speakers == 1:
            iterator = itertools.product(args.thread_values, repeat=3)
            for lse, b1, asr in iterator:
                result.append({"speakers": 1, "lseend": lse, "separator": 1,
                               "b1": b1, "overlap": 1, "asr": asr})
        else:
            iterator = itertools.product(args.thread_values, repeat=5)
            for lse, sep, b1, overlap, asr in iterator:
                result.append({"speakers": speakers, "lseend": lse, "separator": sep,
                               "b1": b1, "overlap": overlap, "asr": asr})
    return result


def case_name(config: dict) -> str:
    return (f"sp{config['speakers']}_l{config['lseend']}_s{config['separator']}_"
            f"b{config['b1']}_o{config['overlap']}_a{config['asr']}")


def pipeline_command(args: argparse.Namespace, config: dict, output: Path) -> list[str]:
    speakers = config["speakers"]
    return [
        args.python, str(args.pipeline),
        "--input", str(args.inputs[speakers]), "--output", str(output),
        "--always-separate-speakers", str(speakers), "--chunk-seconds", "5",
        "--benchmark-warmup-seconds", str(args.benchmark_warmup_seconds),
        "--benchmark-report", str(output / "benchmark.json"),
        "--detailed-timing-log", str(output / "audio_timeline.json"),
        "--warmup-models", "--speaker-encoder-batch-overlap",
        "--lseend-threads", str(config["lseend"]),
        "--separation-threads", str(config["separator"]),
        "--speaker-encoder-threads", str(config["b1"]),
        "--overlap-speaker-encoder-threads", str(config["overlap"]),
        "--asr-threads", str(config["asr"]),
        "--asr-backend", "moonshine", "--moonshine-model", str(args.moonshine_model),
        "--separator-backend", "sandglasset",
        "--lseend-model", str(args.lseend_model), "--lseend-metadata", str(args.lseend_metadata),
        "--sandglasset-2-onnx", str(args.sandglasset_2),
        "--sandglasset-3-onnx", str(args.sandglasset_3),
        "--speaker-encoder-model", str(args.speaker_encoder_model),
    ]


def measured_rtf(output: Path, audio_seconds: float) -> float | None:
    timeline = output / "audio_timeline.json"
    if not timeline.is_file() or audio_seconds <= 0:
        return None
    events = json.loads(timeline.read_text(encoding="utf-8")).get("events", [])
    final = max((
        float(event.get("end_seconds", 0.0))
        for event in events if event.get("stage") != "source_capture"
    ), default=0.0)
    return final / audio_seconds


def start_services(args: argparse.Namespace) -> tuple[subprocess.Popen, subprocess.Popen, object, object]:
    service_dir = args.output / "slm_decode_load"
    service_dir.mkdir(parents=True, exist_ok=True)
    server_log = (service_dir / "llama_server.log").open("w", encoding="utf-8")
    server = subprocess.Popen([
        str(args.llama_server), "--model", str(args.slm_model),
        "--alias", "decode-load", "--host", "127.0.0.1", "--port", str(args.llama_port),
        "--ctx-size", str(args.slm_context), "--parallel", "1",
        "--threads", str(args.slm_threads), "--threads-batch", str(args.slm_batch_threads),
        "--batch-size", "512", "--ubatch-size", "256", "--cache-type-k", "f16",
        "--cache-type-v", "f16", "--flash-attn", "on", "--n-gpu-layers", "0",
    ], stdout=server_log, stderr=subprocess.STDOUT, text=True,
       preexec_fn=nice_preexec(args.slm_nice))
    wait_http(f"http://127.0.0.1:{args.llama_port}/health", server)
    loader_log = (service_dir / "decode_client.log").open("w", encoding="utf-8")
    ready = service_dir / "ready.json"
    ready.unlink(missing_ok=True)
    loader = subprocess.Popen([
        args.python, str(args.decode_loader), "--output", str(service_dir),
        "--ready-file", str(ready), "--base-url", f"http://127.0.0.1:{args.llama_port}",
        "--input-tokens", str(args.slm_input_tokens),
        "--output-tokens", str(args.slm_output_tokens),
    ], stdout=loader_log, stderr=subprocess.STDOUT, text=True,
       preexec_fn=nice_preexec(args.slm_nice))
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if ready.is_file():
            return server, loader, server_log, loader_log
        if loader.poll() is not None:
            raise RuntimeError("decode loader exited before first completed decode window")
        time.sleep(0.25)
    raise TimeoutError("decode loader did not become ready")


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = configurations(args)
    (args.output / "configurations.json").write_text(json.dumps(tasks, indent=2) + "\n")
    progress = args.output / "progress.jsonl"
    completed = set()
    if progress.exists():
        completed = {json.loads(line)["name"] for line in progress.read_text().splitlines()
                     if line.strip() and json.loads(line).get("returncode") == 0}
    server = loader = None
    server_log = loader_log = None
    try:
        server, loader, server_log, loader_log = start_services(args)
        for index, config in enumerate(tasks, 1):
            if loader.poll() is not None:
                raise RuntimeError("continuous SLM decode loader stopped unexpectedly")
            name = case_name(config)
            if name in completed:
                continue
            wait_temperature(args.maximum_start_temperature)
            output = args.output / "runs" / name
            output.mkdir(parents=True, exist_ok=True)
            baseline_samples = max(1, round(args.power_rate_hz * args.power_baseline_seconds))
            subprocess.run([
                args.python, str(args.power_logger), "--output", str(output / "pmic_slm_only_5hz.csv"),
                "--rate-hz", str(args.power_rate_hz), "--samples", str(baseline_samples),
            ], check=True, preexec_fn=nice_preexec(args.logger_nice))
            power = subprocess.Popen([
                args.python, str(args.power_logger), "--output", str(output / "pmic_combined_5hz.csv"),
                "--rate-hz", str(args.power_rate_hz),
            ], preexec_fn=nice_preexec(args.logger_nice))
            started = time.time()
            try:
                result = subprocess.run(
                    pipeline_command(args, config, output), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True,
                    preexec_fn=nice_preexec(args.pipeline_nice),
                )
            finally:
                terminate(power)
            (output / "console.log").write_text(result.stdout, encoding="utf-8")
            audio_seconds = args.audio_seconds[config["speakers"]]
            record = {
                "time_unix": time.time(), "index": index, "total": len(tasks),
                "name": name, "configuration": config, "returncode": result.returncode,
                "wall_seconds": time.time() - started, "audio_seconds": audio_seconds,
                "pipeline_rtf": measured_rtf(output, audio_seconds),
                "slm_only_power": power_summary(output / "pmic_slm_only_5hz.csv"),
                "combined_power": power_summary(output / "pmic_combined_5hz.csv"),
                "temperature_c": read_temperature(),
            }
            append_jsonl(progress, record)
            print(json.dumps(record), flush=True)
            if result.returncode and args.stop_on_error:
                raise RuntimeError(f"{name} failed")
    finally:
        terminate(loader)
        terminate(server)
        if loader_log:
            loader_log.close()
        if server_log:
            server_log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configurations-file", type=Path)
    parser.add_argument("--speakers", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--thread-values", type=int, nargs="+", default=(1, 2, 3, 4))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--decode-loader", type=Path, required=True)
    parser.add_argument("--power-logger", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--slm-model", type=Path, required=True)
    parser.add_argument("--llama-port", type=int, default=8080)
    parser.add_argument("--slm-context", type=int, default=1024)
    parser.add_argument("--slm-input-tokens", type=int, default=951)
    parser.add_argument("--slm-output-tokens", type=int, default=41)
    parser.add_argument("--slm-threads", type=int, default=3)
    parser.add_argument("--slm-batch-threads", type=int, default=4)
    parser.add_argument("--slm-nice", type=int, default=19)
    parser.add_argument("--pipeline-nice", type=int, default=10)
    parser.add_argument("--logger-nice", type=int, default=19)
    parser.add_argument("--power-rate-hz", type=float, default=5.0)
    parser.add_argument("--power-baseline-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-start-temperature", type=float, default=65.0)
    parser.add_argument("--benchmark-warmup-seconds", type=float, default=5.0)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--input-1", type=Path, required=True)
    parser.add_argument("--input-2", type=Path, required=True)
    parser.add_argument("--input-3", type=Path, required=True)
    parser.add_argument("--audio-seconds-1", type=float, default=20.0)
    parser.add_argument("--audio-seconds-2", type=float, default=20.0)
    parser.add_argument("--audio-seconds-3", type=float, default=20.0)
    parser.add_argument("--moonshine-model", type=Path, required=True)
    parser.add_argument("--lseend-model", type=Path, required=True)
    parser.add_argument("--lseend-metadata", type=Path, required=True)
    parser.add_argument("--sandglasset-2", type=Path, required=True)
    parser.add_argument("--sandglasset-3", type=Path, required=True)
    parser.add_argument("--speaker-encoder-model", type=Path, required=True)
    args = parser.parse_args()
    args.inputs = {1: args.input_1, 2: args.input_2, 3: args.input_3}
    args.audio_seconds = {1: args.audio_seconds_1, 2: args.audio_seconds_2, 3: args.audio_seconds_3}
    return args


if __name__ == "__main__":
    run(parse_args())
