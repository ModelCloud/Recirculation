# Llama 3.2 1B full GSM8K/MMLU recirculation evaluation

- Model: `/local-models/Llama-3.2-1B-Instruct`
- Precision: FP16
- Device: CUDA
- Recirculation: `10→1`, `α=0.04`, `β=0.96`, norm-matched, no ramp
- GSM8K decoding: natural generation, maximum 256 new tokens
- MMLU scoring: five-shot multiple-choice likelihood
- MMLU optimization commits: `1046cb7`, `17b2e17`
- Humanities execution commit: `7700e5b3674a5005fc5d6d3a3f78b707761cbdfb`

## Completed results

| Suite | Correct | Rows | Accuracy | Invalid | Runtime | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K Platinum | 597 | 1,209 | 49.38% | 0 | 701.00 s | 1.725 rows/s |
| MMLU-STEM | 1,268 | 3,153 | 40.22% | 0 | completed checkpoint | — |
| MMLU-Humanities | 2,034 | 4,705 | 43.23% | 0 | 1,752.91 s | 2.684 rows/s |

The suites use different scoring contracts, so their correct counts and accuracies
must not be combined into a single benchmark score.

## MMLU execution configuration

- Shared-prefix A/B/C/D scoring enabled
- Automatic unique-prompt width: 512
- Padded-token cohort budget: 524,288
- Stable token-length sorting with output-order restoration
- FlashAttention 2 requested
- Continuous batching enabled
- CUDA graph disabled
- Allocator: `expandable_segments:True,garbage_collection_threshold:0.8`
- Batched/fused forward gate: mean absolute error ≤`4e-3`

Humanities completed with length-sorted token-budgeted cohorts. The first 512-row
request was split as `238×2194`, `190×2758`, and `84×3082`, reducing observed
VRAM from approximately 73 GB in the interrupted fixed-width run to approximately
25.7 GB. The completed JSON is
`results/evaluations/llama32_1b_mmlu_humanities_recirc_10_1_alpha004_fp16_length_bucketed.json`.

