#!/usr/bin/env python3
"""Create the ORT_ENABLE_ALL Moonshine INT8 deployment graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort


NAMES = (
    "encoder_model_int8",
    "decoder_model_int8",
    "decoder_with_past_model_int8",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    for name in NAMES:
        source = model_dir / f"{name}.onnx"
        target = model_dir / f"{name}.ort"
        if target.exists() and target.stat().st_size > 0:
            print(f"exists {target}")
            continue
        if not source.is_file():
            raise FileNotFoundError(source)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = str(target)
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = True
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        print(f"optimising {source} -> {target}", flush=True)
        session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
        del session
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"ORT did not create {target}")
    print(f"ready {model_dir}")


if __name__ == "__main__":
    main()
