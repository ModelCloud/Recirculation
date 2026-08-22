# Llama 3.2 1B Instruct GSM8K-Platinum evaluation

## Completed result

| Suite | Path | Alpha | Correct | Accuracy | Invalid | State |
|---|---:|---:|---:|---:|---:|---|
| GSM8K-Platinum `cot_llama` | 10→1 | 0.04 | 597 / 1,209 | 49.38% | 0 | Complete |

This is a completed suite-level checkpoint. MMLU STEM and MMLU Humanities had not completed and are intentionally
excluded; they are being evaluated separately so GSM8K is not repeated.

## Performance

| Metric | Value |
|---|---:|
| Generation wall time | 701.000 seconds |
| Generation throughput | 1.725 rows/s |
| Generation throughput | 103.48 rows/min |
| Dataset load | 0.010 seconds |
| Scoring | 0.095 seconds |

## Configuration

| Setting | Value |
|---|---|
| Implementation commit | `b982db37c4313cc30e432c2e8aee3c7584325aa0` |
| Model | `/local-models/Llama-3.2-1B-Instruct` |
| Dtype | FP16 |
| Device | CUDA |
| Prompt variant | Evalution `cot_llama` |
| Chat template | Enabled |
| Maximum new tokens | 256 |
| Recirculation | source 10 → destination 1, α=0.04, β=0.96 |
| Attention | Paged FlashAttention 2 |
| Continuous batching | Enabled, coherent cohorts |
| Cohort width | 32 |
| Scheduler token budget | 65,536 |
| Paged cache | 256 blocks × 256 tokens |
| Shared recurrent prefix | 1,024 tokens / 4 blocks |
| CUDA graphs | Disabled; graph-compatible paged recirculation remains a TODO |
| Python | CPython 3.14t, GIL disabled |
| Forward gate | Fused/batched mean absolute error ≤4e-3 |

The suite completed inside the original combined evaluation process. That process was stopped during MMLU after the
GSM8K result had been recorded in its durable status and terminal telemetry, so it did not produce the final combined
JSON. This Markdown checkpoint preserves the completed aggregate, configuration, and timing without claiming that
unfinished MMLU results exist.

## Command

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
