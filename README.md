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

- 2026-08-22 — Completed full-split FP16 evaluation of Llama 3.2 1B Instruct with the paper-corpus-selected
  `10→1`, `alpha=0.04` intervention. Recirculation improved GSM8K Platinum from 588/1,209 to 597/1,209,
  MMLU-STEM from 1,263/3,153 to 1,268/3,153, and MMLU-Humanities from 2,027/4,705 to 2,034/4,705.
  The matched dense baseline, exact commands, resolved CUDA settings, artifact checksums, and paired Humanities flips
  are recorded in the [full evaluation report](results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_vs_recirc_10_1_alpha004_fp16.md).
- 2026-08-21 — Added Qwen3-8B support across Torch/CUDA inference and path/alpha screening, including its no-BOS
  tokenizer contract. Qwen3-8B recirculation runs faster on Torch/MPS and MLX (MLX preferred on Apple Silicon), with
  1.57x prefill speedup at matching outputs and 1.82x faster prompt processing from fixed-prefix reuse.
- 2026-08-20 — Corrected same-token replay so each token replays its own upper stack and replaces upper-layer KV
  state; with zero feedback, replay matches ordinary serial inference in Torch and MLX. CUDA same-token replay reaches
  **4.533x prefill speedup** while meeting accuracy. Added the paper's zero-based ten-step ramp (`ramp_tokens=10`)
  and two-stack CUDA execution for both GIL-enabled and free-threaded Python. CUDA screening now jointly evaluates
  final-answer and full-solution perplexity, then applies the correct-to-wrong penalty during paired generation.
  Torch is the single reference for MLX and CUDA accuracy, evaluation auto-selects CUDA or MLX, and can compare
  same-destination candidates together. MLX ran the matched short Apple M4 evaluation 2.86x faster than Torch/MPS.

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
are withdrawn as recirculation evidence. The current result uses Llama 3.2 1B Instruct in FP16 and evaluates the full
reported test splits with the paper-corpus-selected `10→1`, `alpha=0.04`, `beta=0.96` intervention. The dense and
recirculation arms use the same model, prompts, scoring contracts, engine batch sizes, paged FlashAttention 2 settings,
and MMLU token-budgeted cohorts. Full commands, environment versions, resolved settings, checksums, and row-level
Humanities flip accounting are in the
[detailed reproducibility report](results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_vs_recirc_10_1_alpha004_fp16.md).

### Llama 3.2 1B Instruct

| Benchmark | Dense | Dense acc | Recirc | Recirc acc | Delta | Rel |
|---|---:|---:|---:|---:|---:|---:|
| <span style="white-space:nowrap">GSM8K Platinum</span> | <span style="white-space:nowrap">588/1,209</span> | <span style="white-space:nowrap">48.64%</span> | <span style="white-space:nowrap"><strong>597/1,209</strong></span> | <span style="white-space:nowrap"><strong>49.38%</strong></span> | <span style="white-space:nowrap"><strong>+9 / +0.74 pp</strong></span> | <span style="white-space:nowrap"><strong>+1.53%</strong></span> |
| <span style="white-space:nowrap">MMLU-STEM</span> | <span style="white-space:nowrap">1,263/3,153</span> | <span style="white-space:nowrap">40.06%</span> | <span style="white-space:nowrap"><strong>1,268/3,153</strong></span> | <span style="white-space:nowrap"><strong>40.22%</strong></span> | <span style="white-space:nowrap"><strong>+5 / +0.16 pp</strong></span> | <span style="white-space:nowrap"><strong>+0.40%</strong></span> |
| <span style="white-space:nowrap">MMLU-Humanities</span> | <span style="white-space:nowrap">2,027/4,705</span> | <span style="white-space:nowrap">43.08%</span> | <span style="white-space:nowrap"><strong>2,034/4,705</strong></span> | <span style="white-space:nowrap"><strong>43.23%</strong></span> | <span style="white-space:nowrap"><strong>+7 / +0.15 pp</strong></span> | <span style="white-space:nowrap"><strong>+0.35%</strong></span> |

Relative change is `(recirculation accuracy - dense accuracy) / dense accuracy`. The three suites have different
scoring contracts, so their correct counts and accuracies must not be combined. On the 4,705 aligned Humanities rows,
recirculation produced 42 wrong→correct and 35 correct→wrong flips, a paired net of +7.

| Evaluation detail | Value |
|---|---|
| Model | [`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct), dense and unquantized |
| Evaluation datasets | [`madrylab/gsm8k-platinum`](https://huggingface.co/datasets/madrylab/gsm8k-platinum) full `main/test` split; [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) full STEM and Humanities test groups |
| Evaluation rows | 1,209 GSM8K Platinum; 3,153 MMLU-STEM; 4,705 MMLU-Humanities |
| Path/alpha search data | arXiv, C4, and PG-19 language-modeling windows; disjoint from GSM8K and MMLU evaluation data |
| Evaluation toolkit | [Evalution](https://github.com/ModelCloud/Evalution) `0.0.12`; GSM8K `cot_llama` natural generation; MMLU five-shot A/B/C/D likelihood |
| CUDA execution | FP16, paged FlashAttention 2, continuous batching, engine batch 32, MMLU suite batch 128, 512-prompt scoring cohorts |

The GSM8K delta is provisional: two unseeded dense runs differed on five row scores and three aggregate correct
answers. MMLU reproduced exactly; see the [detailed report](results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_vs_recirc_10_1_alpha004_fp16.md).

### Gemma 3 1B Instruct

The Gemma 3 1B Instruct evaluation uses the paper-corpus-selected `25→20`, `alpha=0.04` intervention. The complete
path/alpha funnel, commands, shard provenance, and backend limitation are recorded in the
[Gemma evaluation report](results/evaluations/gemma3_1b_paper_path_25_20_alpha004_eval_report.md).

| Benchmark | Dense | Dense acc | Recirc | Recirc acc | Delta | Rel |
|---|---:|---:|---:|---:|---:|---:|
| <span style="white-space:nowrap">GSM8K Platinum</span> | <span style="white-space:nowrap">540/1,209</span> | <span style="white-space:nowrap">44.67%</span> | <span style="white-space:nowrap"><strong>554/1,209</strong></span> | <span style="white-space:nowrap"><strong>45.82%</strong></span> | <span style="white-space:nowrap"><strong>+14 / +1.16 pp</strong></span> | <span style="white-space:nowrap"><strong>+2.59%</strong></span> |
| <span style="white-space:nowrap">MMLU-STEM</span> | <span style="white-space:nowrap">1,084/3,153</span> | <span style="white-space:nowrap">34.38%</span> | <span style="white-space:nowrap"><strong>1,083/3,153</strong></span> | <span style="white-space:nowrap"><strong>34.35%</strong></span> | <span style="white-space:nowrap"><strong>−1 / −0.03 pp</strong></span> | <span style="white-space:nowrap"><strong>−0.09%</strong></span> |
| <span style="white-space:nowrap">MMLU-Humanities</span> | <span style="white-space:nowrap">1,736/4,705</span> | <span style="white-space:nowrap">36.90%</span> | <span style="white-space:nowrap"><strong>1,737/4,705</strong></span> | <span style="white-space:nowrap"><strong>36.92%</strong></span> | <span style="white-space:nowrap"><strong>+1 / +0.02 pp</strong></span> | <span style="white-space:nowrap"><strong>+0.06%</strong></span> |

Gemma GSM8K recirculation and dense runs had zero invalid predictions. The dense GSM8K score is the
correctness-validated eager, non-paged baseline; the native dense paged FlashAttention 2 continuous path emitted
corrupted output and is not used as a score. See the
[full Gemma report](results/evaluations/gemma3_1b_paper_path_25_20_alpha004_eval_report.md) for the exact commands,
settings, and raw artifacts.

### Paper-style path screening

Corpus screening reads local arXiv, C4, and PG-19 shards, takes at most two complete 1,024-token windows per document,
and adds no answer cue. The paper reports 484, 488, and 500 windows respectively (about 1.5 million predicted tokens).
The arXiv shard is a documented best-effort choice because the paper does not identify its source dataset. Raw shards
default to `/local-models/datasets/recirculation-paper`; downloads require explicit `--allow-download`.

Llama, Qwen3, and Gemma 3 use the same Torch/CUDA screening interface. Tokenicer preserves checkpoint special-token
behavior: Llama and Gemma use their BOS contract, while Qwen3 starts without a synthetic BOS, EOS, or PAD token. Llama
and Qwen3 support fused candidate batches; Gemma widths above one currently use the accuracy-gated serial fallback.
Batched/fused forwards allow mean absolute error up to `4e-3`; unfused, unbatched forwards allow `2e-3`.

A paper-scale, unrestricted Llama path scan is:

```bash
python -X gil=0 scripts/screen_cuda_recirculation.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --dtype float16 \
  --corpus arxiv --corpus c4 --corpus pg19 \
  --corpus-window-count arxiv=484 \
  --corpus-window-count c4=488 \
  --corpus-window-count pg19=500 \
  --windows-per-document 2 \
  --window-tokens 1024 \
  --alpha 0.10 \
  --scheduler concurrent --dual-gemm \
  --candidate-batch-size 8 \
  --corpus-artifact results/llama32_1b_paper_windows.json \
  --output results/llama32_1b_paper_paths.json
```

Omitting `--max-distance` scans every `source > destination` pair. Candidate order is deterministically randomized;
use `--scan-seed` to change it or repeated `--path SOURCE:DESTINATION` options for a shortlist. The paper's final
cross-corpus comparison fixes `alpha=0.10`; its arXiv heatmap instead uses `0.04`, `0.07`, `0.10`, and `0.16`. Swap the
model path (and dtype when appropriate) to run the same search on Qwen3 or Gemma 3.

## Install and test

Python 3.10 or newer is required.

```bash
python -m pip install -e '.[eval,dev]'
pytest -q
```

The tests verify the published norm-ratio mixture, prohibit cross-token injection, and check replay state handling.

## Run the full evaluation

The evaluator defaults to `--backend auto`: it selects the accelerated CUDA path when CUDA and Triton are available,
MLX on Apple Silicon, then Torch on MPS or CPU. Use `--backend` and `--device` only when an explicit override is
needed.

```bash
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --device cuda \
  --benchmark gsm8k_platinum \
  --benchmark mmlu_stem \
  --benchmark mmlu_humanities \
  --benchmark-arg gsm8k_platinum.variant=cot_llama \
  --benchmark-arg gsm8k_platinum.apply_chat_template=true \
  --benchmark-arg mmlu_stem.batch_size=128 \
  --benchmark-arg mmlu_humanities.batch_size=128 \
  --path 10:1 \
  --alpha 0.04 \
  --report-every-seconds 60 \
  --output results/evaluations/llama32_1b_full_recirc_10_1_alpha004_fp16.json
```

Remove only `--path 10:1` and `--alpha 0.04` to run the matched dense model. The committed detailed report records the
literal commands used for both arms and every automatically resolved inference setting.

On MLX, evaluation automatically snapshots the exact token prefix shared by all selected prompts and restores it for
each row. This avoids repeatedly processing fixed few-shot examples without changing generated outputs.

Multiple same-destination candidates can share one baseline and their common lower-layer work:

```bash
python scripts/evaluate.py paired-gsm8k \
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

The controller remains a one-path, one-iteration implementation. The fixed linear ramp is supported; the paper's
learned adaptive variant, multiple paths, and multiple recirculation iterations are outside this reproduction.

MLX also provides contiguous batched recirculation through `MLXBatchedRecirculator` and request admission/retirement
through `MLXContinuousBatch`. Torch/MPS supports synchronized equal-length dense batches. These paths use contiguous
KV storage; the experimental paged-attention path currently targets CUDA.

Batch accuracy is covered by the unit suite. Warmed speed comparisons are available with
`scripts/benchmark_mlx_batch_recirculation.py` and `scripts/benchmark_torch_batch_recirculation.py`; both report
scalar-versus-batched latency, throughput, speedup, and the forward-error gate.

## Run Evalution benchmarks

One benchmark-agnostic entrypoint runs every suite exposed by Evalution. Repeat `--benchmark` to evaluate several
suites with one model load, use `--suite-arg KEY=VALUE` for common constructor arguments, and use
`--benchmark-arg BENCHMARK.KEY=VALUE` for suite-specific settings. Values accept JSON, so lists, numbers, booleans,
and null can be passed without adding benchmark-specific CLI code.

```bash
python scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --benchmark gsm8k_platinum \
  --benchmark mmlu \
  --max-rows 128 \
  --benchmark-arg gsm8k_platinum.variant=cot_llama \
  --benchmark-arg gsm8k_platinum.apply_chat_template=true \
  --benchmark-arg mmlu.subsets=stem \
  --output results/llama32_1b_gsm8k_mmlu.json
```

Use `python scripts/evaluate.py run --list-benchmarks` to print the installed Evalution targets. A local Arrow file
can be selected with `--suite-arg dataset_path=/absolute/path/data.arrow`; it is loaded directly without a Hub lookup
or cache copy. The generic runner defaults to local-only model loading, FP16, and the validated CUDA inference
settings. Pass `--no-local-files-only` only when an intentional Hub download is required.

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

CUDA GSM8K evaluation also has an experimental Transformers continuous-batching path for models with one
full-attention cache group. `--cuda-paged-continuous` uses paged FlashAttention, stores recirculation residuals in
physical cache pages, copies them with shared-prefix cache blocks, and masks completed rows. A block-aligned recurrent
snapshot seeds the prompt prefix common to every row. Requests are admitted in coherent cohorts (the CUDA batch width
by default), because immediate one-row refill makes long few-shot prompts substantially slower; use
`--cuda-paged-admission-batch` to test another admission width. The path remains accuracy-gated against the Torch CUDA
runner and does not enable CUDA graphs.

TODO: make paged recirculation CUDA-graph compatible. The packed forward currently reads dynamic sequence metadata
from CUDA on the host, which graph capture rejects. A graph-safe implementation needs fixed device-resident metadata,
captured shape buckets, and forward-error validation before paged varlen/decode graphs can become an automatic default.

On CUDA, the sweep/evaluation scripts now choose the validated settings automatically: FP16, FlashAttention 2, paged
continuous batching, coherent request/admission width 32, 256 cache blocks of 256 tokens, a cache-derived 65,536-token
scheduler budget, and the expandable-segments allocator. CUDA graphs remain off until the graph-compatible paged TODO
above is accuracy-gated. Dense Evalution and recirculation evaluation log the resolved values and save them in result
provenance. Explicit `--no-cuda-paged-continuous`, `--no-paged-attention`, and `--no-continuous-batching` switches remain
available for controlled comparisons and systems without the required kernels.

```bash
python scripts/sweep_gsm8k.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --dataset /local-models/datasets/gsm8k-platinum/test.arrow \
  --device cuda \
  --rows 64 \
  --candidate 10:1:0.05 \
  --cuda-paged-continuous \
  --cuda-batch-size 32 \
  --output results/paged_recirculation.json
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
