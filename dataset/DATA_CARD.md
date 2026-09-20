# PARDA silver-label data card

## Purpose

The annotations support research on causal semantic anonymization of natural conversation. They identify explicit and inferred private attributes and the transcript lines supporting each inference.

## Source and creation

Source conversations come from the CANDOR corpus of naturalistic dyadic video calls. PARDA linearizes CANDOR's cliffhanger transcripts into speaker-addressed lines. DeepSeek V4 Pro with high reasoning generated the silver labels under a nine-category privacy taxonomy. The paper reports subsequent review by two annotators.

## Composition

The release contains 717 parsed label sets. The paper uses 515 conversations for training, 59 for checkpoint validation, and 143 for held-out evaluation. The held-out IDs are included; development membership remains explicitly pending until the original split manifest is recovered.

## Limitations

These are model-generated silver labels and may contain false inferences, omissions, and subjective certainty judgments. CANDOR is English, dyadic, and recorded over video calls, so it does not represent all languages, cultures, acoustic conditions, or wearer-view interactions. Labels can themselves contain sensitive inferred attributes and must be handled accordingly.

## Distribution

CANDOR audio, video, and transcripts are not included. Users must obtain CANDOR separately and comply with its license and access terms. This repository currently grants no separate open-source or data license.
