# Gemma 3 1B paper-path evaluation

Model: `/local-models/gemma-3-1b-it` (FP16). The path/alpha search selected the
robust paper-aligned arm `25->20, alpha=0.04` from
`results/gemma3_1b_paper_exact_1472x1024_path_alpha/race/`. The MMLU runs below
use the same CUDA settings; the dense run differs only by omitting
`--path/--alpha`.

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

## GSM8K status

The full GSM8K-Platinum score is **not recorded**. A one-row natural-generation
probe with the same path, chat template, and paged FA2 configuration failed at
runtime with a CUDA device-side assertion after 326.4 s (`gemma3_gsm_probe_serial_v2.status.json`). Earlier paged runs that emitted gibberish are intentionally not treated as scores. The generation path must be fixed and revalidated before launching a 1209-row comparison.

The evaluator did confirm the intended runtime selection (`paged|flash_attention_2`,
continuous batching). With GIL=1, the new serial `CUDAPrefillRunner` fallback is
used; `CUDAConcurrentRunner` Python workers are reserved for GIL=0.
