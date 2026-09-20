"""Run the windowed adversary/anonymiser pipeline on one transcript."""

import argparse
import json
import os
from pathlib import Path

from backends import get_backend
from pipeline import run_streaming
from rag_store import EditMemory, Embedder
from transcript_utils import load_records
from windowing import get_tokenizer


ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--backend", default="llama.cpp", choices=["llama.cpp", "openrouter"])
    parser.add_argument("--adversary-backend")
    parser.add_argument("--anonymiser-backend")
    parser.add_argument("--model")
    parser.add_argument("--adversary-model")
    parser.add_argument("--anonymiser-model")
    parser.add_argument("--providers", default="deepseek")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--window-lines", type=int, default=12)
    parser.add_argument("--passes-per-window", type=int, default=1)
    parser.add_argument("--rag-candidates", type=int, default=20)
    parser.add_argument("--rag-top-k", type=int, default=8)
    parser.add_argument("--rag-threshold", type=float, default=.774)
    parser.add_argument("--context-lines", type=int, default=2)
    parser.add_argument("--embed-model", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--stream-tokens", action="store_true")
    parser.add_argument("--llama-role-slots", action="store_true")
    parser.add_argument("--timing-log", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--raw-outputs", type=Path, default=ROOT / "outputs" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "transcripts")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    load_env(args.env_file)

    def backend(role: str):
        kind = getattr(args, role + "_backend") or args.backend
        model = getattr(args, role + "_model") or args.model
        options = {"model": model, "stream": args.stream_tokens,
                   "no_think": args.no_think and kind == "llama.cpp",
                   "slot": {"adversary": 0, "anonymiser": 1}.get(role)
                   if args.llama_role_slots else None,
                   "timing_log": args.timing_log, "seed": args.seed}
        if kind == "openrouter":
            options["providers"] = args.providers.split(",")
            options["reasoning_effort"] = args.reasoning_effort
        return get_backend(kind, **options)

    turns = load_records(args.transcript)[0]
    memory = EditMemory(
        Embedder(args.embed_model, args.embed_device),
        threshold=args.rag_threshold,
        candidates=args.rag_candidates,
        top_k=args.rag_top_k,
        context_lines=args.context_lines,
    )
    result = run_streaming(turns, backend("adversary"), backend("anonymiser"), memory,
                           get_tokenizer(), window_lines=args.window_lines,
                           passes_per_window=args.passes_per_window, temperature=args.temperature,
                           top_p=1.0, max_tokens=args.max_tokens)
    name = args.name or Path(args.transcript).stem
    raw_dir = args.raw_outputs / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "original": result.original, "anonymised": result.anonymised,
               "inferences": result.profile, "windows": result.windows, "errors": result.errors,
               "exchanges": {key: value.__dict__ for key, value in result.exchanges.items()}}
    (raw_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    output_dir = args.output_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input.txt").write_text(result.original)
    (output_dir / "anonymised.txt").write_text(result.anonymised)
    print(f"wrote {raw_dir / (name + '.json')} and {output_dir / 'anonymised.txt'}")


if __name__ == "__main__":
    main()
