# Benchmarks

PARDA reports several workloads with different pacing and contention semantics. Keep them separate when comparing results.

| Workload | Input delivery | Purpose |
| --- | --- | --- |
| Audio scheduling search | Unpaced 20-second forced-speaker audio | Saturate stages and select thread cohorts |
| Semi-paced replay | Audio released according to recording time | Measure deployment-like lag, drain, and power |
| Concurrent SLM sweep | Unpaced forced-speaker audio plus continuous low-priority decode | Measure audio scheduling under SLM contention |

The scheduling search prewarms model sessions. One-speaker runs bypass source separation. Forced two- and three-speaker runs separate every five-second block, so their RTF values are stress workloads rather than estimates of ordinary overlap frequency.

## Paper scheduling results

| Workload | LSE | Separation | Embedding | Overlap | ASR | RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 speaker | 1 | - | 4 | - | 3 | 0.6405 |
| 2 speakers | 1 | 2 | 4 | 3 | 3 | 1.2471 |
| 3 speakers | 1 | 2 | 1 | 3 | 3 | 1.5734 |
| Weighted policy | 1 | 2 | 3 | 2 | 3 | 0.7703 |

The weighted policy uses `0.81 RTF1 + 0.15 RTF2 + 0.04 RTF3`.

## Included tools

- `scripts/benchmarks/decode_contention_sweep.py`: exhaustive forced-speaker thread search under continuous SLM decode.
- `scripts/benchmarks/continuous_decode_load.py`: bounded cached-prompt decode load.
- `scripts/benchmarks/pmic_power_logger.py`: Raspberry Pi PMIC sampling.

Benchmark records should include model identity, audio hash and duration, warm-up policy, pacing mode, thread allocation, SLM tokens/s, RTF, 5 Hz power, temperature, and throttling state.
