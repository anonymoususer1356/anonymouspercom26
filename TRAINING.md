# Training and model export

The paper trains a Qwen3.5 2B student in two stages: supervised fine-tuning on teacher adversary/anonymizer completions, followed by targeted DPO over same-state anonymizer actions. The release currently provides the runtime prompts and validation logic, but the final training launch scripts and trained adapters are not yet included.

Reported preparation:

- 515 training conversations produce 15,480 windows and 30,960 balanced role examples.
- The selected SFT adapter is checkpoint 3,870.
- Preference construction samples 200 conversations and yields 6,241 pairs from 564 states across 190 conversations.
- DPO uses a frozen SFT reference, one epoch, BF16, and an NVIDIA A100 80 GB.
- Export merges the adapter into the BF16 base, removes the auxiliary MTP head, emits f16 GGUF, and applies llama.cpp Q4_0 quantization.

Until the frozen launch manifests are curated, these numbers document the paper rather than define a complete one-command reproduction. Do not infer omitted optimizer or scheduler settings.
