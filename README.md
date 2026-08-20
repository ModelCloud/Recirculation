# Recirculation reproduction

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

## MLX optimization policy

`recirculation.mlx_backend.MLXRecirculator` is the exact MLX-LM reference path. It walks a loaded decoder one token at
a time, updates the ordinary per-layer KV caches, mixes the previous token's source after the destination block, and
retains the current source for the next token.

Every optimized forward must be compared against this unfused reference over the full accumulated sequence. The gate
is `max(relative L2 error, normalized maximum error) <= 2e-3`. A faster result that exceeds this limit must not be
promoted or documented as an accepted optimization.

### Accepted optimization 1: compiled exact norm/mix

`CompiledNormMix` uses `mx.compile` on the unchanged reference expression. On the test Apple M4, the isolated
2,048-element operation improved from 191.75 us to 108.04 us median (`1.77x`). A warmed 64-token dense Llama 3.2 1B
prefill improved from 375.94 ms to 368.08 ms median (`1.021x`). The accumulated full-logit trace error was exactly zero
under the metrics above. See `results/mlx_compiled_mix_m4_64_tokens.json` and reproduce with
`scripts/benchmark_mlx_prefill.py`.

### Accepted optimization 2: exact recirculated prefix snapshots

`MLXRecirculator.snapshot` stores both the per-layer KV state and the pending deep-layer source activation. Restoring
only the KV tensors would be incorrect because the first suffix token also depends on the final prefix source. Across
eight requests sharing 64 prefix tokens and carrying 16 unique suffix tokens, warmed median time improved from
2,954.94 ms to 885.35 ms (`3.338x`) on the test Apple M4. Final logits were bit-identical (error rate `0.0`). See
`results/mlx_prefix_cache_m4_8x64_plus_16.json` and `scripts/benchmark_mlx_prefix_cache.py`.

### Accepted optimization 3: project only final prefill logits

Intermediate prompt logits do not feed attention, KV state, or recirculation. The optimized prefill therefore runs the
final RMSNorm and vocabulary projection only for the last prompt token. For 64 tokens, warmed median time improved from
369.72 ms to 289.62 ms (`1.277x`) on the test Apple M4. Final logits were bit-identical (error rate `0.0`). Diagnostic
callers can retain the slower all-token trace with `collect_logits=True`. See
`results/mlx_final_projection_m4_64_tokens.json` and `scripts/benchmark_mlx_final_projection.py`.

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
