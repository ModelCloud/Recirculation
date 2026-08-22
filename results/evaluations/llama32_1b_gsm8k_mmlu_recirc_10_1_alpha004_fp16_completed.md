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

## Exact commands

These are the literal commands used for the completed suite checkpoints. They use
the same unified `scripts/evaluate.py run` entry point. All commands were executed
from the repository root on CPython 3.14t with free threading enabled.

### GSM8K Platinum

```bash
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --device cuda \
  --benchmark gsm8k_platinum \
  --benchmark mmlu_stem \
  --benchmark mmlu_humanities \
  --benchmark-arg gsm8k_platinum.variant=cot_llama \
  --benchmark-arg gsm8k_platinum.apply_chat_template=true \
  --path 10:1 \
  --alpha 0.04 \
  --report-every-seconds 60 \
  --output results/evaluations/llama32_1b_full_gsm8k_platinum_mmlu_stem_humanities_recirc_10_1_alpha004_fp16.json
```

The combined process was stopped after GSM8K completed, so the durable GSM8K
checkpoint is Markdown rather than a final combined JSON.

### MMLU-STEM

```bash
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --device cuda \
  --benchmark mmlu_stem \
  --benchmark mmlu_humanities \
  --path 10:1 \
  --alpha 0.04 \
  --report-every-seconds 30 \
  --output results/evaluations/llama32_1b_mmlu_stem_humanities_recirc_10_1_alpha004_fp16.json
```

The process was stopped during the original oversized Humanities cohort after
STEM completed. Its durable status file preserves the completed STEM aggregate.

### MMLU-Humanities

```bash
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --device cuda \
  --benchmark mmlu_humanities \
  --path 10:1 \
  --alpha 0.04 \
  --report-every-seconds 30 \
  --output results/evaluations/llama32_1b_mmlu_humanities_recirc_10_1_alpha004_fp16_length_bucketed.json
```

The completed JSON records the resolved automatic configuration: engine batch 32,
512 unique MMLU prompts, 2,048 A/B/C/D requests, a 524,288 padded-token scoring
budget, paged FlashAttention 2, continuous batching, FP16, and CUDA graphs disabled.
