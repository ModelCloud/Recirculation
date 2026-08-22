# Llama 3.2 1B MMLU shared-prefix batching

- Date: 2026-08-22
- Implementation commit: `1046cb7`
- Pre-optimization commit: `ef3921b`
- Model: `/local-models/Llama-3.2-1B-Instruct`
- Device: CUDA, 96 GB VRAM
- Precision: FP16
- Recirculation: `10→1`, `α=0.04`
- Benchmark: Evalution MMLU-STEM, five-shot multiple-choice likelihood
- Attention request: paged FlashAttention 2; recirculation likelihoods use the repository CUDA scorer

## Result

Evalution represents every MMLU row as four one-token choice requests. The old
adapter replayed the identical five-shot prompt four times. The optimized adapter
consolidates identical contexts, evaluates each prompt once, and gathers the A/B/C/D
losses from that shared state.

| Implementation | Unique prompt batch | Rows | Wall time | Throughput | Speedup vs old |
|---|---:|---:|---:|---:|---:|
| Old duplicated-choice scorer | 8 effective | 32 | 52.01 s | 0.615 rows/s | 1.00× |
| Shared-prefix scorer | 32 | 32 | 19.10 s | 1.675 rows/s | 2.72× |
| Shared-prefix scorer | 64 | 64 | 19.58 s | 3.268 rows/s | 5.31× |
| Shared-prefix scorer | 128 | 128 | 20.40 s | 6.274 rows/s | 10.20× |
| Shared-prefix scorer | 256 | 256 | 33.93 s | 7.545 rows/s | 12.27× |
| Shared-prefix scorer | 512 | 512 | 51.24 s | 9.991 rows/s | 16.25× |

The CUDA auto policy chooses 512 unique prompts on this device from the post-load
free-VRAM measurement. Smaller devices step down through 256, 128, 64, and 32.
An explicitly supplied `--batch-size` remains authoritative.

## Accuracy gate

The fixed 32-row before/after comparison produced the same 21.88% accuracy, exact
A/B/C/D log-likelihoods at width 32, and zero prediction mismatches. A real-model
forward sweep compared vocabulary logits for the same 338-token prompt against the
batch-32 reference:

| Batch | Mean absolute logit error | Maximum error | Greedy-token parity | Peak allocated VRAM |
|---:|---:|---:|---:|---:|
| 32 | 0 | 0 | exact | 2.85 GB |
| 64 | 0.001949 | 0.015625 | exact | 3.26 GB |
| 128 | 0.002182 | 0.015625 | exact | 3.97 GB |
| 256 | 0.002157 | 0.015625 | exact | 5.45 GB |
| 512 | 0.002027 | 0.015625 | exact | 8.46 GB |

Batch 512 passes the repository fused/batched forward gate of mean absolute error
≤`4e-3` and preserves the greedy token exactly. The gate intentionally uses mean
forward-logit error, not the more punitive maximum error and not an aggregate eval score.

## Validation

- Full tests: 132 passed, 25 skipped
- Ruff: passed
- `git diff --check`: passed
- Automatic 96 GB selection smoke test: 512 unique prompts / 2,048 choice requests

## Long-context token-budgeted cohorts

- Implementation commit: `17b2e17`
- Automatic padded-token budget on this device: 524,288
- Ordering: stable token-length sort with outputs restored to their original row/choice indices
- Telemetry: cohort count, `rows×max_tokens` layout, and active cohort are logged

The first 512 Humanities prompts average 1,960 tokens and reach 3,082 tokens,
compared with 436 average and 664 maximum for the first 512 STEM prompts. A
fixed 512-row cohort therefore made the evaluator appear stalled and grew to
73 GB VRAM without returning after more than 12 minutes.

| Humanities configuration | Cohort layout | Padded tokens | Result |
|---|---|---:|---:|
| Fixed width 512 | `512×3082` | 1,577,984 | incomplete after >12 min; 73 GB VRAM |
| Length sorted, budget 524,288 | `238×2194, 190×2758, 84×3082` | 1,305,080 | 532.41 s; 0.962 rows/s; ~25.7 GB VRAM |

The completed 512-row result was 248/512 (`acc,ll=48.44%`) with zero invalid
predictions. The new path completed at least 1.35× sooner than the interrupted
fixed-width lower bound and reduced observed VRAM by approximately 65%. Short
STEM prompts remain in one 512-row cohort because their padded work fits the
same token budget.

Validation after the token-budget change: 134 passed, 25 skipped; Ruff and
`git diff --check` passed.
