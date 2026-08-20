# Recirculation

## Paper and authors

This independent reproduction is based on
[Recirculation](https://arxiv.org/html/2608.17981v1) by **Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer,
Sunny Sanyal, and Rosanne Liu**. Their central contribution is a training-free inference method that feeds a small,
norm-matched amount of each token's deeper representation back into a shallower layer, giving the model a recurrent
state across token steps while retaining first-pass readout. The implementation, validation, and optimization in this
repository all build on that method. This is not the authors' official implementation.

## Progress

- 2026-08-20 — Locked disjoint evaluation completed at 59/128 baseline and 67/128 with recirculation.
- 2026-08-20 — CUDA screening jointly evaluates historical final-answer and full-solution perplexity, then applies
  the real correct-to-wrong penalty during paired generation.
- 2026-08-20 — Two-stack CUDA execution supports both GIL-enabled and free-threaded Python.
- 2026-08-20 — The paper's zero-based ten-step ramp is now available with `ramp_tokens=10`.
- 2026-08-20 — Corrected CUDA same-token replay reached **4.533x prefill speedup** while meeting accuracy standards.
- 2026-08-20 — Corrected recirculation to replay each token's own upper stack and replace its upper-layer KV state.
- 2026-08-20 — With zero feedback, replay now matches ordinary serial inference exactly in Torch and MLX.

The code aims to remain as faithful to the upstream method as possible. Backend implementations may differ slightly,
especially during inference, where MLX, CUDA, ROCm, and their underlying hardware impose different execution and
kernel constraints. Such backend-specific differences should preserve the published mathematical and state behavior
and need to meet accuracy standards.

For token `t`, let `d_t` and `s_t` be its first-pass residual-stream outputs at the destination and deeper source
decoder blocks. Decoder-block indices are zero-based. The default implementation applies the paper's per-token L2
norm matching and convex mixture:

```text
s_hat_t = ||d_t||_2 / max(||s_t||_2, epsilon) * s_t
alpha_t = alpha                                  # ramp disabled
alpha_t = min(t / ramp_tokens, 1) * alpha       # ramp enabled
beta_t = 1 - alpha_t                            # default convex mixture
d_mix_t = beta_t * d_t + alpha_t * s_hat_t
```

Backend epsilon guards handle zero-norm numerical edge cases. If a non-convex `beta` is supplied explicitly, it
remains fixed while `alpha_t` ramps. The mixed state acts as token `t`'s replacement output at the destination and is
replayed from block `destination + 1` through the final decoder block. That replay replaces token `t`'s KV entries in
those upper blocks; readout still uses the token's first-pass logits, as specified by the paper.

An ordinary all-token parallel prefill is therefore not equivalent: token `t`'s corrected upper state must be ready
before token `t+1` enters block `destination + 1`. The reference Torch and MLX paths execute this dependency serially.
The concurrent CUDA path overlaps token `t`'s upper replay with token `t+1` through the destination, then joins the two
branches before `t+1` enters the upper stack. The intervention changes neither model weights nor checkpoint files.

## Evaluation status

Results produced before the same-token replay correction measured a different delayed cross-token intervention and
are withdrawn as recirculation evidence. The initial corrected candidate was selected by final-answer likelihood. On
128 disjoint rows, Evalution scored the baseline at 59 correct and recirculation at 67 correct; the positive estimate
is still statistically noise-consistent, so the candidate remains provisional. The CUDA screening funnel now scores
the historical final-answer continuation and the complete gold solution in the same candidate batch. It retains plain
perplexity and harmed-row-penalized rankings for both objectives, sends their union to paired generation, and only then
applies the actual correct-to-wrong flip penalty. See [`results/`](results/) for the complete record.

| Arm | Configuration | Correct | Accuracy | Change vs. baseline |
|---|---|---:|---:|---:|
| Dense baseline | No recirculation | 59/128 | 46.09% | — |
| Recirculation | 8→2, alpha 0.20 | **67/128** | **52.34%** | **+8 correct / +6.25 percentage points** |

This is a **13.56% relative accuracy increase**. The search rows and evaluation rows are disjoint.

| Evaluation detail | Value |
|---|---|
| Model | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), dense and unquantized |
| Dataset | [`madrylab/gsm8k-platinum`](https://huggingface.co/datasets/madrylab/gsm8k-platinum), configuration `main`, split `test` |
| Path search | Test rows 272–303 (32 rows) |
| Alpha search | Test rows 304–335 (32 rows) |
| Locked evaluation | Test rows 144–271 (128 rows) |
| Evaluation toolkit | [Evalution](https://github.com/ModelCloud/Evalution) `0.0.7`, GSM8K-Platinum `cot_llama`, primary metric `acc,num` |

The path search, alpha search, and locked evaluation ranges are pairwise disjoint.

## Install and test

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[eval,dev]'
pytest -q
```

The tests verify the published norm-ratio mixture, prohibit cross-token injection, and check replay state handling.

## Run the confirmation evaluation

Use `mps` on Apple Silicon, `cuda` on NVIDIA, or `cpu`:

```bash
python scripts/eval_gsm8k_platinum.py \
  --device mps \
  --row-start 144 \
  --rows 128 \
  --forbid-range 272:304 \
  --forbid-range 304:336 \
  --max-new-tokens 256 \
  --source-layer 8 \
  --destination-layer 2 \
  --alpha 0.20 \
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

The controller currently supports batch size 1 and the one-path, one-iteration variant. The fixed linear ramp is
supported; the paper's learned adaptive variant, multiple paths, and multiple recirculation iterations are outside
this reproduction.

## Accelerated backends

The repository includes semi-optimized MLX and CUDA paths that speed up path/alpha screening and recirculation
inference on Apple Silicon and NVIDIA hardware. Accelerated paths are checked against the reference behavior and need
to meet accuracy standards. CUDA screening evaluates all valid `source > destination` pairs by default, without a
layer-distance cap. Each candidate step jointly scores complete solutions and the historical final-answer-only
continuation; proxy regressions are kept distinct from actual accuracy flips, which require paired generation on the
disjoint evaluation stage. Multiprocess screens can use `--native-baseline PATH`: prepare it once with
`--baseline-only`, then every worker validates and reuses the same per-row baseline instead of recomputing alpha zero.

Install with `python -m pip install -e '.[mlx,dev]'` for MLX or `python -m pip install -e '.[cuda,eval,dev]'` for CUDA.
Reproduction commands and benchmark details are kept under [`results/`](results/).

## Citation

Paper: [Recirculation (Mozer et al., 2026), arXiv:2608.17981v1](https://arxiv.org/html/2608.17981v1)

```bibtex
@article{mozer2026recirculation,
  title={Recirculation},
  author={Mozer, Michael C. and Siddiqui, Shoaib Ahmed and Sawyer, Danny and Sanyal, Sunny and Liu, Rosanne},
  journal={arXiv preprint arXiv:2608.17981},
  year={2026},
  url={https://arxiv.org/abs/2608.17981v1}
}
```

Licensed under Apache-2.0.
