# Recirculation

## Paper and authors

This best-effort independent reproduction is based on
[Recirculation](https://arxiv.org/html/2608.17981v1) by **Michael C. Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer,
Sunny Sanyal, and Rosanne Liu**. Their central contribution is a training-free inference method that feeds a small,
norm-matched amount of each token's deeper representation back into a shallower layer, giving the model a recurrent
state across token steps while retaining first-pass readout. The implementation, validation, and optimization in this
repository all build on that method.

This repository is **not** the authors' official implementation, a reference implementation, or an authoritative
statement of the method, and it does not claim to be correct. The code may contain bugs, misunderstandings, or
behavioral differences from the paper. Until the authors release source code, development here will continue as a
best-effort attempt to reproduce the publicly described method; results should be treated as provisional and
independently verified.

## Progress

- 2026-08-21 — Added Qwen3-8B support across Torch/CUDA inference and path/alpha screening, including its no-BOS
  tokenizer contract. Qwen3-8B recirculation runs faster on Torch/MPS and MLX (MLX preferred on Apple Silicon), with
  1.57x prefill speedup at matching outputs and 1.82x faster prompt processing from fixed-prefix reuse.
- 2026-08-20 — Corrected same-token replay so each token replays its own upper stack and replaces upper-layer KV
  state; with zero feedback, replay matches ordinary serial inference in Torch and MLX. CUDA same-token replay reaches
  **4.533x prefill speedup** while meeting accuracy. Added the paper's zero-based ten-step ramp (`ramp_tokens=10`)
  and two-stack CUDA execution for both GIL-enabled and free-threaded Python. CUDA screening now jointly evaluates
  final-answer and full-solution perplexity, then applies the correct-to-wrong penalty during paired generation.
  Locked disjoint evaluation finished at 59/128 baseline and 67/128 recirculation. Torch is the single reference for
  MLX and CUDA accuracy, evaluation auto-selects CUDA or MLX, and can compare same-destination candidates together.
  MLX ran the matched short Apple M4 evaluation 2.86x faster than Torch/MPS.

The Torch path is the absolute reference for the published mathematical and state behavior. MLX and CUDA may use
hardware-specific execution, but their outputs are checked against Torch and need to meet accuracy standards.

For token `t`, let `d_t` and `s_t` be its first-pass residual-stream outputs at the destination and deeper source
decoder blocks. Decoder-block indices are zero-based. The default implementation applies the paper's per-token L2
norm matching and convex mixture:

```text
s_hat_t = ||d_t||_2 / ||s_t||_2 * s_t          # when ||s_t||_2 > 0
s_hat_t = 0                                     # zero-source-norm edge case
alpha_t = alpha                                  # ramp disabled
alpha_t = min(t / ramp_tokens, 1) * alpha       # ramp enabled
beta_t = 1 - alpha_t                            # default convex mixture
d_mix_t = beta_t * d_t + alpha_t * s_hat_t
```

The paper does not define division by a zero source norm, so Torch defines that normalized source as zero and the
accelerated backends mirror it. If a non-convex `beta` is supplied explicitly, it remains fixed while `alpha_t` ramps.
The mixed state acts as token `t`'s replacement output at the destination and is replayed from block
`destination + 1` through the final decoder block. That replay replaces token `t`'s KV entries in those upper blocks;
readout still uses the token's first-pass logits, as specified by the paper.

An ordinary all-token parallel prefill is therefore not equivalent: token `t`'s corrected upper state must be ready
before token `t+1` enters block `destination + 1`. The reference Torch and MLX paths execute this dependency serially.
The concurrent CUDA path overlaps token `t`'s upper replay with token `t+1` through the destination, then joins the two
branches before `t+1` enters the upper stack. The intervention changes neither model weights nor checkpoint files.

## Evaluation status

Results produced before the same-token replay correction measured a different delayed cross-token intervention and
are withdrawn as recirculation evidence. The initial corrected candidate was selected by final-answer likelihood. On
128 disjoint rows, Evalution scored the baseline at 59 correct and recirculation at 67 correct; the positive estimate
is still statistically noise-consistent, so the candidate remains provisional. The CUDA screening funnel no longer
injects `The final answer is ` into the prompt. It uses complete-gold-solution likelihood only as a cheap shortlist
proxy, then lets the dense and recirculation arms generate autoregressively with no answer cue. Only naturally
generated paired accuracy can promote a candidate, and promotion requires both a positive net correction count and a
positive harmed-row-penalized score. See [`results/`](results/) for the complete record.

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

For a paper-style language-modeling shortlist, corpus mode streams fixed windows from C4 and PG-19 train. The example
below uses 256 qualifying documents per corpus and scores one 1024-token window from each. It does not add an answer
cue; GSM8K natural generation remains the downstream promotion gate.

The Torch and CUDA paths support Hugging Face Llama and Qwen3 causal-LM checkpoints. Tokenicer preserves each model's
special-token contract: Llama corpus windows begin after the checkpoint BOS, while Qwen3—which intentionally has no
BOS—starts from an empty KV cache and scores the second text token from the first. Recirculation does not substitute
EOS or PAD as a synthetic Qwen3 BOS. Chat evaluation uses the checkpoint's own chat template.

```bash
python scripts/screen_cuda_recirculation.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --corpus c4 \
  --corpus pg19 \
  --windows-per-corpus 256 \
  --window-tokens 1024 \
  --corpus-artifact results/c4_pg19_256x1024_windows.json \
  --row-batch-size 256 \
  --alpha 0.10 \
  --output results/cuda_c4_pg19_paths.json
```

For the locally downloaded Qwen3-8B checkpoint, the same search is selected with:

```bash
python scripts/screen_cuda_recirculation.py \
  --model /local-models/Qwen3-8B \
  --corpus c4 \
  --corpus pg19 \
  --windows-per-corpus 256 \
  --window-tokens 1024 \
  --corpus-artifact results/qwen3_c4_pg19_256x1024_windows.json \
  --alpha 0.05 \
  --output results/qwen3_cuda_paths.json
```

Qwen3-8B has 36 decoder layers, so an unrestricted scan contains 630 ordered `source > destination` paths per alpha.
Use repeated `--path SOURCE:DESTINATION` arguments to smoke-test or refine a shortlist before launching the full grid.
Candidate execution is deterministically randomized by default so partial runs sample the layer space instead of always
starting at 1→0, 2→0, 3→0. JSON and Markdown ledgers store the full scan-index→path/alpha schedule and each result's
scan index. Set `--scan-seed` to choose a different reproducible order or `--scan-order sequential` for legacy order.

## Install and test

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[eval,dev]'
pytest -q
```

The tests verify the published norm-ratio mixture, prohibit cross-token injection, and check replay state handling.

## Run the confirmation evaluation

The evaluator defaults to `--backend auto`: it selects the accelerated CUDA path when CUDA and Triton are available,
MLX on Apple Silicon, then Torch on MPS or CPU. Use `--backend` and `--device` only when an explicit override is
needed.

```bash
python scripts/eval_gsm8k_platinum.py \
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

The baseline uses ordinary backend-native generation. The intervention arm uses the same model, tokenizer, prompts,
greedy decoding settings, and row order, but performs paper-faithful serial prefill.

On MLX, evaluation automatically snapshots the exact token prefix shared by all selected prompts and restores it for
each row. This avoids repeatedly processing fixed few-shot examples without changing generated outputs.

Multiple same-destination candidates can share one baseline and their common lower-layer work:

```bash
python scripts/eval_gsm8k_platinum.py \
  --row-start 144 \
  --rows 128 \
  --forbid-range 272:304 \
  --forbid-range 304:336 \
  --path 8:2 \
  --path 4:2 \
  --alpha 0.10 \
  --output results/comparison.json
```

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
When a screen is split across processes, merge every shard before promotion with
`scripts/aggregate_cuda_screening.py`. The aggregate validates commit and scoring settings, rejects duplicate
candidates, recalculates all four rankings over the full completed population, and atomically maintains both a full
JSON record and a human-readable Markdown ledger. Downstream alpha or E2E promotion must consume this aggregate,
never an individual shard.

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
