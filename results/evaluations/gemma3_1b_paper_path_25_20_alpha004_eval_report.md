# Gemma 3 1B paper-path evaluation

Model: `/local-models/gemma-3-1b-it` (FP16). The path/alpha search selected the
robust paper-aligned arm `25->20, alpha=0.04` from
`results/gemma3_1b_paper_exact_1472x1024_path_alpha/race/`.

## MMLU results

| Suite | Rows | Recirculation | Dense | Delta |
|---|---:|---:|---:|---:|
| MMLU-STEM | 3153 | 34.35% | 34.38% | -0.03 pp |
| MMLU-Humanities | 4705 | 36.92% | 36.90% | +0.02 pp |

Both runs completed all rows with zero invalid predictions. Recirculation took
6080.5 s (1.292 rows/s); dense took 143.6 s (54.734 rows/s). The backend was
FP16, FlashAttention-2, paged attention, continuous batching, batch size 128,
and a 16384-token batch budget.

Recirculation command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
/root/venv-py3.14t-gil0/bin/python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/gemma-3-1b-it --device cuda \
  --max-batch-tokens 16384 --batch-size 128 \
  --benchmark mmlu_stem --benchmark mmlu_humanities \
  --path 25:20 --alpha 0.04 --report-every-seconds 30 \
  --output results/evaluations/gemma3_1b_mmlu_recirc_best_fp16.json
```

Dense command (identical settings, recurrence disabled):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
/root/venv-py3.14t-gil0/bin/python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/gemma-3-1b-it --device cuda \
  --max-batch-tokens 16384 --batch-size 128 \
  --benchmark mmlu_stem --benchmark mmlu_humanities --report-every-seconds 30 \
  --output results/evaluations/gemma3_1b_mmlu_dense_fp16.json
```

## GSM8K-Platinum results

| Suite | Rows | Recirculation | Dense correctness baseline | Delta |
|---|---:|---:|---:|---:|
| GSM8K-Platinum | 1209 | 554/1209 (45.82%) | 540/1209 (44.67%) | +14 / +1.16 pp |

Both full runs used the same model, FP16 dtype, GSM8K-Platinum split, eight-shot
chat template, row order, and 256-token generation limit. Recirculation used
`25->20, alpha=0.04`, while dense omitted recurrence. Four row partitions were
run in parallel and merged with exact global coverage `0..1208`; both have zero
invalid predictions. Recirculation wall time was 6666.5 s (0.181 rows/s across
the four shards); the dense eager baseline wall time was 624.9 s (1.935 rows/s).

The dense correctness baseline deliberately uses eager, non-paged,
non-continuous generation. A native dense run with paged FlashAttention-2 and
continuous batching completed but produced corrupted/repetitive outputs (623
invalid rows and 11/1209 correct), so it is retained only as a diagnostic and
is not used for the score comparison. This backend limitation is recorded here
so the result is reproducible rather than silently presenting an invalid
baseline.

Both runs used four explicit `row_indices` partitions: `0..302`, `303..605`,
`606..908`, and `909..1208`. The following command template was run once per
partition, with `ROW_INDICES_JSON` set to the corresponding JSON array and
`SHARD` set to `0`, `1`, `2`, or `3`:

Recirculation shard command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 BLIS_NUM_THREADS=16 \
PYTHONUNBUFFERED=1 /root/venv-py3.14t-gil0/bin/python3 scripts/evaluate.py run \
  --model /local-models/gemma-3-1b-it --device cuda \
  --attention-backend flash_attention_2 --continuous-batching --paged-attention \
  --benchmark gsm8k_platinum --benchmark-arg gsm8k_platinum.row_indices="$ROW_INDICES_JSON" \
  --path 25:20 --alpha 0.04 --batch-size 8 --max-batch-tokens 16384 \
  --report-every-seconds 30 \
  --output results/evaluations/gemma3_gsm_recirc_shard${SHARD}.json
```

Dense baseline shard command (the only semantic change is recurrence disabled;
eager/non-paged/non-continuous is required for valid Gemma generation):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16 BLIS_NUM_THREADS=16 \
PYTHONUNBUFFERED=1 /root/venv-py3.14t-gil0/bin/python3 scripts/evaluate.py run \
  --model /local-models/gemma-3-1b-it --device cuda \
  --attention-backend eager --no-paged-attention --no-continuous-batching \
  --benchmark gsm8k_platinum --benchmark-arg gsm8k_platinum.row_indices="$ROW_INDICES_JSON" \
  --batch-size 8 --max-batch-tokens 16384 \
  --report-every-seconds 30 \
  --output results/evaluations/gemma3_gsm_dense_eager_shard${SHARD}.json
```

Merged artifacts are `gemma3_1b_gsm8k_recirc_best_fp16.json` and
`gemma3_1b_gsm8k_dense_fp16.json`, with matching `.status.json` files. The
per-shard JSON and logs are retained alongside them.

With GIL=1, the serial `CUDAPrefillRunner` fallback is used;
`CUDAConcurrentRunner` Python workers are reserved for GIL=0.
