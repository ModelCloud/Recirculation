# Transformers paged recirculation benchmark

Date: 2026-08-21
Validation base commit: `57a17ed5366784ed9a54dfa8699b7f04dc85e31e`
Implementation state: pre-commit validation snapshot
Model: `/local-models/Llama-3.2-1B-Instruct`
Device: NVIDIA CUDA, 96 GiB
Precision: FP16
Path: 10→1, alpha 0.05

## Numerical and behavioral gates

| Gate | Result |
|---|---:|
| Torch CUDA vs. paged raw-logit max absolute error | 0.0 |
| Allowed forward-error rate | ≤ 0.002 |
| One-step greedy token parity | Exact |
| 32-row generated-text parity vs. established runner | 32/32 exact |
| 32-row numeric-answer parity | 32/32 exact |
| 64-row generated-text parity vs. established runner | 64/64 exact |
| 32-row numeric flips | 1 wrong→correct, 0 correct→wrong, net +1 |
| 64-row numeric flips | 4 wrong→correct, 1 correct→wrong, net +3 |

The raw gate is stored in `results/transformers_paged_recirc_forward_gate.json`. The 32-row behavioral result is
stored in `results/dense_baselines/llama32_1b_gsm8k_recirc_10_1_alpha005_paged_cb_final_batch32.json`.

## Width scaling before paged scheduling

Identical 128-token prompts and 32 generated tokens were run through the CUDA concurrent runner. Every width produced
the same continuation tokens as width 1.

| Batch width | Aggregate generated tok/s | Median seconds | Peak allocated GiB |
|---:|---:|---:|---:|
| 1 | 7.61 | 4.204 | 2.35 |
| 2 | 14.78 | 4.330 | 2.36 |
| 4 | 30.25 | 4.231 | 2.38 |
| 8 | 58.28 | 4.393 | 2.42 |
| 16 | 118.93 | 4.305 | 2.50 |
| 32 | 240.57 | 4.257 | 2.66 |

Source artifact: `results/cuda_batch_widths_llama32_1b_10_1_alpha005_fp16.json`.

## Real GSM8K throughput

The prompts share 1,078 tokens. Four complete 256-token blocks (1,024 tokens) were precomputed once and imported into
the Transformers paged cache with their recurrent state.

| Configuration | Rows | Total seconds | Generated tokens | End-to-end tok/s | Scheduler tok/s |
|---|---:|---:|---:|---:|---:|
| Dynamic SDPA runner | 8 | 62.46 | — | — | — |
| Seeded paged FlashAttention | 8 | 46.25 | — | — | — |
| Seeded paged FlashAttention | 32 | 58.72 | 3,296 | 56.13 | 143.28 |
| Native immediate one-row refill | 64 | 202.95 | — | — | — |
| Cohort refill, width 32 | 64 | 73.70 | 6,346 | 86.10 | — |

The 32-row run spent 35.71 seconds constructing the recurrent prefix once and 23.00 seconds inside paged scheduling.
Immediate refill was pathological for these long few-shot prompts: every newly free slot admitted and prefetched one
request almost alone. Coherent 32-request admission made the 64-row workload 2.75× faster and preserved the established
outputs exactly. Prefix cost is paid once per candidate, so larger evaluations approach the measured 143 tok/s
scheduler rate.

## Reproduction

```bash
PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
python scripts/sweep_gsm8k.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --dataset /local-models/datasets/gsm8k-platinum/test.arrow \
  --device cuda \
  --row-start 0 \
  --rows 32 \
  --max-new-tokens 256 \
  --skip-baseline \
  --baseline-results results/dense_baselines/llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json \
  --candidate 10:1:0.05 \
  --cuda-paged-continuous \
  --cuda-batch-size 32 \
  --cuda-paged-num-blocks 256 \
  --output results/dense_baselines/llama32_1b_gsm8k_recirc_10_1_alpha005_paged_cb_final_batch32.json
```
