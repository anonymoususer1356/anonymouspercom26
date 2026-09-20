# Dataset release

PARDA was developed with 717 silver-labeled conversations from CANDOR. CANDOR is a third-party corpus and is not redistributed here. This directory contains PARDA's parsed annotation layer and the IDs needed to reconstruct inputs from a separately obtained CANDOR copy.

## Current split status

| Split | Paper count | Public status |
| --- | ---: | --- |
| Train | 515 | Membership placeholder pending recovery from frozen training artifacts |
| Validation | 59 | Membership placeholder pending recovery from frozen training artifacts |
| Test | 143 | IDs verified against the frozen evaluation manifest and result tables |

The 574 unresolved development labels are in `silver_labels/pending_assignment/`. They must not be treated as training data until the original train/validation membership is recovered. No new random split is substituted for the paper split.

`candor_manifest.csv` is the authoritative inventory. Each row gives the CANDOR conversation ID, expected processed transcript filename, current split, label path, membership status, and annotation checksum. The per-split CSVs provide convenient subsets.

## Reconstruct transcripts

Obtain CANDOR from its maintainers and place each conversation under:

```text
candor_raw/<conversation_id>/transcription/transcript_cliffhanger.csv
```

Then run:

```bash
python scripts/convert_candor.py \
  --candor-root dataset/candor_raw \
  --output dataset/processed
```

The converter preserves CANDOR's cliffhanger order and writes:

```text
1: speaker_a: First utterance
2: speaker_b: Second utterance
```

Speaker aliases follow first occurrence. Fillers, disfluencies, punctuation, and ASR artifacts are retained.

## Label schema

Each `labels.json` contains a `speakers` array. Every speaker has nine protected categories: identity, location, demographic, health, legal, economic, relational, lifestyle, and appearance. Each annotation contains:

- `value`: the inferred private attribute;
- `reasoning`: why the cited evidence supports it;
- `lines`: one or more transcript line numbers;
- `certainty`: integer confidence from 1 to 5.

Only parsed annotations are released. Raw provider requests/responses and copied CANDOR transcripts are excluded.

Validate the release with `python scripts/validate_dataset.py`. Maintainers can regenerate the manifest with `--write` after a verified split update.
