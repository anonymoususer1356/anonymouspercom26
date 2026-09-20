#!/usr/bin/env python3
"""Maintain bounded, decode-only llama.cpp load and discard generated text."""

from __future__ import annotations

import argparse
import json
import signal
import time
import urllib.request
from pathlib import Path


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def token_count(base_url: str, text: str) -> int:
    response = post(base_url + "/tokenize", {"content": text, "add_special": True}, 60.0)
    return len(response["tokens"])


def representative_prompt(base_url: str, target_tokens: int) -> tuple[str, int]:
    sentence = (
        "speaker_a discussed an ordinary personal detail and speaker_b replied; "
        "rewrite the passage while preserving meaning and removing identifying information. "
    )
    words = (sentence * max(2, target_tokens // 10)).split()
    low, high = 1, len(words)
    best_text, best_count = words[0], token_count(base_url, words[0])
    while low <= high:
        middle = (low + high) // 2
        candidate = " ".join(words[:middle])
        count = token_count(base_url, candidate)
        if count <= target_tokens:
            best_text, best_count = candidate, count
            low = middle + 1
        else:
            high = middle - 1
    return best_text, best_count


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / "decode_load.jsonl"
    prompt, prompt_tokens = representative_prompt(args.base_url, args.input_tokens)
    stop = {"requested": False}

    def request_stop(*_unused) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    request_index = 0
    while not stop["requested"]:
        started = time.time()
        try:
            response = post(args.base_url + "/completion", {
                "prompt": prompt,
                "n_predict": args.output_tokens,
                "ignore_eos": True,
                "cache_prompt": True,
                "id_slot": 0,
                "temperature": args.temperature,
                "seed": args.seed + request_index,
                "stream": False,
            })
            timings = response.get("timings") or {}
            record = {
                "time_unix": time.time(), "event": "decode_window",
                "request": request_index, "wall_seconds": time.time() - started,
                "tokenized_input_tokens": prompt_tokens,
                "requested_output_tokens": args.output_tokens,
                "tokens_evaluated": response.get("tokens_evaluated", timings.get("prompt_n")),
                "tokens_predicted": response.get("tokens_predicted", timings.get("predicted_n")),
                "prompt_ms": timings.get("prompt_ms"),
                "predicted_ms": timings.get("predicted_ms"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "context_policy": "cached_prompt_checkpoint",
            }
            append_jsonl(events, record)
            # Request zero creates the prompt checkpoint. Request one proves
            # that the checkpoint can be restored before measurements begin.
            if request_index == 1 and not args.ready_file.exists():
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                args.ready_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
            request_index += 1
        except Exception as error:
            append_jsonl(events, {
                "time_unix": time.time(), "event": "error", "request": request_index,
                "error": str(error),
            })
            raise
    append_jsonl(events, {"time_unix": time.time(), "event": "stopped", "requests": request_index})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--input-tokens", type=int, default=951)
    parser.add_argument("--output-tokens", type=int, default=41)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.input_tokens < 1 or args.output_tokens < 1:
        parser.error("token counts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
