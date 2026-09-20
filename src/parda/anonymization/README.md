# Runtime harness

This directory contains the publishable runtime core. The previous implementation,
experiments, plots, data builders, and Pi tools are preserved unchanged in
`../../Scripts old/`.

| File | Responsibility |
| --- | --- |
| `run.py` | Command-line entry point; writes raw trajectory JSON and final text. |
| `pipeline.py` | One-pass streaming adversary/anonymiser loop. |
| `backends.py` | OpenAI-compatible llama.cpp and OpenRouter clients. |
| `rag_store.py` | Embedding retrieval of prior edits and the accumulated attacker profile. |
| `validation.py` | JSON repair, profile schema validation, normalisation, rejection reports, and exact deduplication. |
| `windowing.py` | Token-budgeted whole-line windows. |
| `transcript_utils.py` | Input loading, output parsing, and safe edit application. |
| `Prompts/` | The exact reviewed model instructions. |

The old framework's standalone silver-label and privacy/utility rescoring
tools are intentionally archived for review rather than mixed into this small
runtime. They can be brought back only after their schemas are consolidated.

Adversary responses are passed through `validation.py` before they are added to
the running profile. The raw run trace records whether JSON was valid, repaired,
valid-but-empty, or malformed, together with every rejected-record reason.

Example:

```sh
python Scripts/Harness/run.py \
  --transcript "Dataset/Postprocessed/<transcript>.txt" \
  --backend llama.cpp --no-think --stream-tokens --llama-role-slots \
  --window-lines 12 --passes-per-window 1 \
  --rag-candidates 20 --rag-top-k 8 --rag-threshold 0.774 \
  --context-lines 2 \
  --embed-model nomic-ai/nomic-embed-text-v1.5 --embed-device cpu
```

Retrieval treats every line in the current window as an independent embedding
query. Up to `rag-top-k` qualifying edits are retained for each line, duplicate
edit IDs are merged across lines, and no second window-wide top-k limit is
applied. Each retrieved edit is shown with up to `context-lines` original lines
before it and the same number after it. The defaults reproduce the selected
configuration: eight results per line and two context lines on each side.
Deduplicated results are ordered from weaker to stronger matches so the most
relevant examples sit closest to the anonymiser instruction.
