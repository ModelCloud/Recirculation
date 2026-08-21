# Llama 3.2 1B Instruct: GSM8K-Platinum FP16 evaluation

## Result

`10→1, α=0.05` completed all 1,209 GSM8K-Platinum rows and improved numeric accuracy by 1.08 percentage
points over the paired dense baseline.

| Arm | Correct | Accuracy | Wrong→correct | Correct→wrong | Net correct |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 591 / 1,209 | 48.88% | — | — | — |
| Recirculation `10→1, α=0.05` | 604 / 1,209 | 49.96% | 52 | 39 | +13 |

The primary metric is Evalution's format-insensitive numeric GSM8K score. The dense predictions came from the
committed full result
[`llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json`](llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json).
Both arms used the same local rows, FP16 model, eight-shot `cot_llama` prompt, chat template, greedy decoding, and
256-token generation limit. Because greedy FP16 output is batch-shape-sensitive, comparisons outside this exact
protocol should not be treated as bitwise reproductions.

## Provenance

- Evaluation implementation: `2aec78b` (`Reuse dense baselines in batched recirculation evals`)
- Dense artifact commit: `3e8e256`
- Path-search implementation recorded by stage 3: `949e5e3f3d8d32ad3d29bd081de8a3d42e617535`
- Model: `/local-models/Llama-3.2-1B-Instruct`
- Dataset: `/local-models/datasets/gsm8k-platinum/test.arrow`, rows `0:1209`
- Device: NVIDIA PG506-230, CUDA
- Dtype: FP16 for both arms
- Scorer: Evalution 0.0.12 `acc,num`

`10→1` was the stage-3 pure language-modeling-perplexity leader on 256 C4 plus 256 PG-19 windows. It was not
selected using these GSM8K evaluation rows. Its stage-3 target perplexity was 19.2861928 with ΔNLL −0.00185280322.

## Dense configuration

| Setting | Value |
|---|---|
| Engine | Evalution `Transformers` |
| Transformers | 5.15.1 |
| Attention | SDPA |
| Batch size | 128 fixed-shape requests |
| Continuous batching | Disabled; paged SDPA has no usable decode fast path in this environment |
| Variant | `cot_llama` |
| Few-shot layout | 8 examples, multiturn |
| Chat template | Enabled |
| Sampling | Greedy |
| Maximum new tokens | 256 |
| End-to-end time | 194.439 seconds |
| Throughput | 6.218 rows/s |

Dense command:

```bash
PYTHON_GIL=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=true \
PYTHONUNBUFFERED=1 \
/root/venv-py3.14t-gil0/bin/python scripts/eval_dense_gsm8k_evalution.py \
  --batch-size 128 \
  --output results/dense_baselines/llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json
```

## Recirculation configuration

| Setting | Value |
|---|---|
| Source → destination | `10→1` |
| α / β | `0.05 / 0.95` |
| Source normalization | Enabled, norm matched |
| Additional iterations | 1 |
| Ramp tokens | 0 |
| Runner | `CUDAConcurrentRunner` |
| Python | 3.14.7 free-threaded, GIL disabled |
| CUDA Python workers | Enabled |
| Candidate prompt batching | Exact-length batches, maximum width 32 |
| Shared-prefix serial replay | Disabled |
| Attention | SDPA |
| Compiled runner | Automatic, enabled; Inductor CUDA graph trees disabled |
| Maximum new tokens | 256 |
| Dense baseline inference | Reused from committed JSON; not repeated |

Recirculation command (the same invocation also evaluates the pending `11→7` robust-NLL leader):

```bash
PYTHON_GIL=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=true \
PYTHONUNBUFFERED=1 \
/root/venv-py3.14t-gil0/bin/python scripts/sweep_gsm8k.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --dataset /local-models/datasets/gsm8k-platinum/test.arrow \
  --device cuda \
  --row-start 0 \
  --rows 1209 \
  --max-new-tokens 256 \
  --skip-baseline \
  --baseline-results results/dense_baselines/llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json \
  --candidate 10:1:0.05 \
  --candidate 11:7:0.05 \
  --no-cuda-shared-prefix \
  --cuda-batch-size 32 \
  --cuda-python-threads \
  --status-every 60 \
  --output results/dense_baselines/llama32_1b_instruct_gsm8k_platinum_recirculation_10_1_11_7_alpha005_fp16.json
```

## Interpretation

The net gain is positive, but 39 baseline-correct rows regressed. This is evidence for this path under this exact
evaluation protocol, not a claim that recirculation uniformly improves every prompt. The separate `11→7` arm was
still running when this checkpoint was recorded and is intentionally excluded from the completed result table.
