"""Independently run an adversary and score an original/anonymised transcript pair.

The script is deliberately separate from ``run.py``. It performs three
measurements after anonymisation has finished:

1. a fresh adversary inference on the anonymised transcript;
2. privacy grading of those guesses against silver labels;
3. utility grading against the original transcript.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from backends import get_backend
from Prompts.prompt_generator import (
    build_adversary_prompt,
    build_privacy_judge_prompt,
    build_utility_judge_prompt,
)
from transcript_utils import extract_json, speaker_key


def load_env(path: Path) -> None:
    # Load simple KEY=value settings without printing their values.
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def load_labels(path: Path, minimum_certainty: float) -> dict:
    # Group silver-label items by speaker and attribute.
    grouped = defaultdict(list)
    data = json.loads(path.read_text(encoding="utf-8"))

    # Older label files use {"items": [{"speaker", "category", ...}]}.
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        items = data["items"]
        for item in items:
            if (item.get("certainty") or 0) >= minimum_certainty:
                key = (speaker_key(item["speaker"]), item["category"])
                grouped[key].append(item["value"])
        return grouped

    # The current silver-label generator uses
    # {"speakers": [{"speaker": ..., "identity": [...], ...}]}.
    for speaker_block in data.get("speakers", []):
        speaker = speaker_key(speaker_block.get("speaker"))
        for attribute, values in speaker_block.items():
            if attribute == "speaker" or not isinstance(values, list):
                continue
            for item in values:
                if (item.get("certainty") or 0) >= minimum_certainty:
                    grouped[(speaker, attribute)].append(item["value"])
    return grouped


def render_privacy_items(grouped: dict, guesses: object) -> str:
    # Create the reference/guess block expected by the privacy judge.
    by_key = defaultdict(list)
    records = guesses if isinstance(guesses, list) else []
    for record in records:
        if not isinstance(record, dict) or not record.get("guess"):
            continue
        key = (speaker_key(record.get("speaker")), record.get("attribute"))
        by_key[key].append(str(record["guess"]))

    blocks = []
    for (speaker, attribute), values in grouped.items():
        guesses_text = "\n".join(f"    {value}" for value in by_key[(speaker, attribute)])
        if not guesses_text:
            guesses_text = "    none"
        references = "\n".join(f"    [{index}] {value}" for index, value in enumerate(values, 1))
        blocks.append(f"{speaker} | {attribute}\n  attacker guesses:\n{guesses_text}\n"
                      f"  reference items:\n{references}")
    return "\n\n".join(blocks)


def collect_privacy_scores(result: object, grouped: dict) -> tuple[list[dict], list[str]]:
    # Read every returned score; missing items count as zero.
    scores = {}
    errors = []
    if isinstance(result, dict):
        for speaker_block in result.get("speakers", []):
            if not isinstance(speaker_block, dict):
                continue
            speaker = speaker_key(speaker_block.get("speaker"))
            for attribute, entries in speaker_block.items():
                if attribute == "speaker" or not isinstance(entries, list):
                    continue
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(entry.get("index"), int):
                        scores[(speaker, attribute, entry["index"])] = entry.get("score", 0)

    output = []
    for (speaker, attribute), values in grouped.items():
        for index, value in enumerate(values, 1):
            score = scores.get((speaker, attribute, index), 0)
            if score not in (0, 0.5, 1):
                errors.append(f"{speaker}.{attribute}[{index}] returned an invalid score")
                score = 0
            if (speaker, attribute, index) not in scores:
                errors.append(f"{speaker}.{attribute}[{index}] was not scored; counted as 0")
            output.append({"speaker": speaker, "attribute": attribute, "index": index,
                           "value": value, "score": score})
    return output, errors


def utility_score(result: object) -> tuple[float, list[str]]:
    # Convert the utility judge's three fields into the project score.
    if not isinstance(result, dict):
        return 0.0, ["utility judge output was not a JSON object"]
    required = ("readability", "meaning", "hallucinations")
    if any(not isinstance(result.get(key), dict) for key in required):
        return 0.0, ["utility judge output was missing a required field"]
    try:
        score = (result["readability"]["score"] / 10
                 + result["meaning"]["score"] / 10
                 + result["hallucinations"]["score"]) / 3
    except (KeyError, TypeError):
        return 0.0, ["utility judge scores were malformed"]
    return score, []


def make_backend(args: argparse.Namespace, role: str):
    # Use role-specific settings when present; otherwise retain the old shared settings.
    backend_name = getattr(args, f"{role}_backend") or args.backend
    model = getattr(args, f"{role}_model") or args.model
    providers = getattr(args, f"{role}_providers") or args.providers
    reasoning_effort = getattr(args, f"{role}_reasoning_effort") or args.reasoning_effort
    options = {"model": model, "stream": args.stream_tokens, "no_think": args.no_think,
               "seed": args.seed}
    if backend_name == "openrouter":
        options.update({"providers": providers.split(","), "reasoning_effort": reasoning_effort})
    return get_backend(backend_name, **options)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--anonymised", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--backend", default="llama.cpp", choices=["llama.cpp", "openrouter"])
    parser.add_argument("--model")
    parser.add_argument("--providers", default="deepseek")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--adversary-backend", choices=["llama.cpp", "openrouter"])
    parser.add_argument("--judge-backend", choices=["llama.cpp", "openrouter"])
    parser.add_argument("--adversary-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--adversary-providers")
    parser.add_argument("--judge-providers")
    parser.add_argument("--adversary-reasoning-effort")
    parser.add_argument("--judge-reasoning-effort")
    parser.add_argument("--temperature", type=float, default=.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--stream-tokens", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--min-certainty", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("Judge Outputs"))
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--env-file", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    load_env(args.env_file)

    adversary = make_backend(args, "adversary")
    judge = make_backend(args, "judge")
    original = args.original.read_text(encoding="utf-8")
    anonymised = args.anonymised.read_text(encoding="utf-8")

    adversary_system, adversary_user = build_adversary_prompt(anonymised)
    adversary_exchange = adversary.chat(adversary_system, adversary_user, args.temperature, 1.0,
                                        args.max_tokens, "judge_adversary")
    inferences = extract_json(adversary_exchange.content) if adversary_exchange.content else []

    grouped = load_labels(args.labels, args.min_certainty)
    privacy_system, privacy_user = build_privacy_judge_prompt(
        render_privacy_items(grouped, inferences))
    privacy_exchange = judge.chat(privacy_system, privacy_user, args.temperature, 1.0,
                                  args.max_tokens, "privacy_judge")
    privacy_result = extract_json(privacy_exchange.content) if privacy_exchange.content else None
    privacy_items, privacy_errors = collect_privacy_scores(privacy_result, grouped)
    total = len(privacy_items)
    privacy_fraction = sum(item["score"] for item in privacy_items) / total if total else 0.0

    utility_system, utility_user = build_utility_judge_prompt(original, anonymised)
    utility_exchange = judge.chat(utility_system, utility_user, args.temperature, 1.0,
                                  args.max_tokens, "utility_judge")
    utility_result = extract_json(utility_exchange.content) if utility_exchange.content else None
    utility, utility_errors = utility_score(utility_result)

    output = {"original": str(args.original), "anonymised": str(args.anonymised),
              "adversary_backend": adversary.name, "adversary_model": adversary.model,
              "judge_backend": judge.name, "judge_model": judge.model,
              "privacy_fraction": privacy_fraction,
              "privacy_items": privacy_items, "privacy_errors": privacy_errors,
              "utility": utility, "utility_judge": utility_result,
              "utility_errors": utility_errors, "adversary_inferences": inferences,
              "exchanges": {"adversary": adversary_exchange.__dict__, "privacy": privacy_exchange.__dict__,
                            "utility": utility_exchange.__dict__}}
    destination = args.output_dir / args.tag
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "judges.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"privacy={privacy_fraction:.3f} utility={utility:.3f}")
    print(f"wrote {destination / 'judges.json'}")


if __name__ == "__main__":
    main()
