# Recirculation

## Paper and authors

This independent reproduction is based on
[Recirculation](https://arxiv.org/html/2608.17981v1) by **Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer,
Sunny Sanyal, and Rosanne Liu**. Their central contribution is a training-free inference method that feeds a small,
norm-matched amount of each token's deeper representation back into a shallower layer, giving the model a recurrent
state across token steps while retaining first-pass readout. The implementation, validation, and optimization in this
repository all build on that method. This is not the authors' official implementation.

## Progress

- 2026-08-20 — CUDA screening now supports full-solution paired perplexity, tail-regression penalties, and E2E harm gates.
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
are withdrawn as recirculation evidence. The initial corrected path/alpha screen used final-answer likelihood; its
candidate remains provisional because paired E2E evaluation found both wrong-to-correct and correct-to-wrong changes.
The robust full-solution screen and locked Evalution-scored confirmation use separate tuning and holdout ranges.

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
to meet accuracy standards.

Install with `python -m pip install -e '.[mlx,dev]'` for MLX or `python -m pip install -e '.[cuda,eval,dev]'` for CUDA.
Reproduction commands and benchmark details are kept under [`results/`](results/).

### Robust CUDA screening funnel

The CUDA screen searches source/destination pairs first at a conservative fixed alpha, then refines alpha only on the
best paths. It scores complete GSM8K rationales and final answers, compares every row with an exact `alpha=0` native
baseline, and ranks by native-relative NLL plus a worst-tail regression penalty. The historical final-answer-only
proxy remains available as `--target-mode final_answer`, but it should not select release settings.

Finalists undergo paired greedy generation. Ranking records both wrong-to-correct and correct-to-wrong transitions
and defaults to `wrong_to_correct - 2 * correct_to_wrong`; a hard regression cap is also available. Exact outcome
transitions cannot be inferred from teacher-forced perplexity, so a disjoint E2E holdout remains mandatory.

```bash
python scripts/run_cuda_screening_funnel.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --row-start 272 \
  --rows 32 \
  --holdout-row-start 304 \
  --holdout-rows 32 \
  --top-paths 8 \
  --e2e-top-k 5 \
  --harm-weight 2 \
  --output-dir results/cuda_screening_funnel
```

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
