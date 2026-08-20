# Recirculation reproduction

## Progress

- 2026-08-20 — Graphed CUDA fused prefill ran **3.987x faster** over 128 tokens within the forward-error gate.
- 2026-08-20 — Exact shared-prefix reuse made real eight-shot GSM8K prefill **5.24x faster** with zero measured error.
- 2026-08-20 — MLX prefill reached **1.30x speedup** over 128 tokens with zero measured forward error.
- 2026-08-20 — Shared-prefix reuse reached **3.34x speedup** for repeated prompts.
- 2026-08-20 — Dense Llama 3.2 1B improved from 46.09% to 53.91% on the 128-row confirmation.

This repository contains an independent, inference-only implementation and validation of
[Recirculation](https://arxiv.org/abs/2608.17981). It is not the authors' official implementation.

The implementation adds a delayed deep-to-shallow residual path. For token step `t`, the source activation from a
deeper layer is norm-matched and mixed into the output of a shallower destination layer at step `t+1`:

```text
s_hat = ||d|| / ||s|| * s
d_next = beta * d + alpha * s_hat
```

Serial prefill is required because prompt token `t+1` consumes feedback produced by prompt token `t`. Decode remains
cached and autoregressive. The intervention changes neither model weights nor checkpoint files.

## Reproduction result

We first screened four middle-stack layer pairs on 16 GSM8K-Platinum rows. The selected `12 -> 5` pair was then locked
and evaluated on a disjoint 128-row confirmation slice using dense `meta-llama/Llama-3.2-1B-Instruct`.

| Metric | Baseline | Recirculation | Change |
|---|---:|---:|---:|
| Correct answers | 59/128 | **69/128** | **+10** |
| Accuracy | 46.09% | **53.91%** | **+7.81 points** |
| Relative accuracy | — | — | **+16.95%** |
| Wrong to correct | — | 15 | — |
| Correct to wrong | — | 5 | — |

The paired discordances are 15 gains and 5 losses (exact two-sided binomial/McNemar `p ~= 0.0414`). This is a focused
reproduction result, not evidence that the configuration generalizes to other models, datasets, or prompt formats.

### Locked confirmation configuration

| Setting | Value |
|---|---|
| Model | `meta-llama/Llama-3.2-1B-Instruct` (dense) |
| Dataset | `madrylab/gsm8k-platinum`, test rows 144–271 |
| Source -> destination | `12 -> 5` on the next token |
| Alpha / beta | `0.10 / 0.90` |
| Source normalization | L2 norm matched to destination |
| Ramp | Disabled |
| Decode | Greedy, at most 256 new tokens |
| Prompt | Fixed eight-shot chat prompt in `configs/` |

Aggregate artifacts are under [`results/`](results/). They intentionally exclude dataset questions and model outputs.

## Install and test

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[eval,dev]'
pytest -q
```

The unit tests verify that feedback comes from the previous token, is mixed after the destination layer, and is removed
cleanly when the controller detaches.

## Run the confirmation evaluation

Use `mps` on Apple Silicon, `cuda` on NVIDIA, or `cpu`:

```bash
python scripts/eval_gsm8k_platinum.py \
  --device mps \
  --row-start 144 \
  --rows 128 \
  --max-new-tokens 256 \
  --source-layer 12 \
  --destination-layer 5 \
  --alpha 0.10 \
  --output results/reproduction.json
```

The baseline uses ordinary Hugging Face generation. The intervention arm uses the same model, tokenizer, prompts,
greedy decoding settings, and row order, but performs paper-faithful serial prefill.

## Screen layer pairs

Source and destination layers are model-dependent hyperparameters. The screen used here can be reproduced with:

```bash
python scripts/sweep_gsm8k.py \
  --device mps \
  --row-start 128 \
  --rows 16 \
  --max-new-tokens 128 \
  --output results/screen.json
```

The controller currently supports batch size 1 and the one-path, one-iteration variant. Adaptive alpha, multiple paths,
and multiple recirculation iterations are outside this reproduction.

## MLX prefill

An MLX-LM prefill path and reusable shared-prefix state are included for Apple Silicon.

Every faster forward is checked against the reference implementation. The maximum permitted accumulated forward-error
rate is `2e-3`; changes above that limit are rejected.

Install the MLX backend on Apple Silicon with `python -m pip install -e '.[mlx,dev]'`.

Detailed benchmark outputs and settings are available under [`results/`](results/).

## CUDA prefill

The CUDA backend preserves token-serial KV-cache updates while fusing the two L2 reductions, source normalization, and
residual mixture into one Triton kernel. It also skips the vocabulary projection for every non-final prompt token. For
repeated inference at a known prompt length, `CUDAGraphedPrefill` captures the complete serial loop and replays it with
new token and mask values, removing per-token CPU dispatch. Fused results are gated against the ordinary PyTorch
expression with the same `2e-3` maximum forward-error rate.

```bash
python -m pip install -e '.[cuda,eval,dev]'
python scripts/benchmark_cuda_prefill.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --tokens 128 \
  --output results/cuda_fused_prefill_128_tokens.json
```

## Citation

```bibtex
@article{mozer2026recirculation,
  title={Recirculation},
  author={Mozer, Michael C. and Siddiqui, Shoaib Ahmed and Sawyer, Danny and Sanyal, Sunny and Liu, Rosanne},
  journal={arXiv preprint arXiv:2608.17981},
  year={2026}
}
```

Licensed under Apache-2.0.
