# Recirculation reproduction

## Progress

- 2026-08-20 — Two-stack CUDA execution supports both GIL-enabled and free-threaded Python.
- 2026-08-20 — Ramping now follows the paper's zero-based 10-step schedule exactly.
- 2026-08-20 — Corrected CUDA same-token replay reached **4.533x prefill speedup** within the `2e-3` error gate.
- 2026-08-20 — Corrected recirculation to replay each token's own upper stack and replace its upper-layer KV state.
- 2026-08-20 — With zero feedback, replay now matches ordinary serial inference exactly in Torch and MLX.

This repository contains an independent, inference-only implementation and validation of
[Recirculation](https://arxiv.org/abs/2608.17981). It is not the authors' official implementation.
The code aims to remain as faithful to the upstream method as possible. Backend implementations may differ slightly,
especially during inference, where MLX, CUDA, ROCm, and their underlying hardware impose different execution and
kernel constraints. Such backend-specific differences should preserve the published mathematical and state behavior
and must be validated against the repository's accuracy gates.

For token `t`, its first-pass source activation is norm-matched and mixed with its own first-pass destination
activation. The mixed state is then replayed through layers after the destination, replacing token `t`'s upper-layer
KV entries before token `t+1` is processed:

```text
s_hat = ||d|| / ||s|| * s
d_second_pass = beta * d_first_pass + alpha * s_hat_first_pass
```

Serial prefill is required because token `t`'s corrected upper-layer state must be available before token `t+1`
reaches those layers. The intervention changes neither model weights nor checkpoint files.

## Evaluation status

Results produced before the same-token replay correction measured a different delayed cross-token intervention and
are withdrawn as recirculation evidence. Path/alpha tuning and the locked GSM8K confirmation must be rerun.

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
residual mixture into one Triton kernel. Fixed-length CUDA Graph replay includes the corrected same-token upper-stack
pass and KV replacement. On Llama 3.2 1B, the corrected 128-token benchmark measured `4.533x` speedup with accumulated
forward error `0.001206`, below the `0.002` release gate. Ramped coefficients are supported by the eager fused CUDA
runner; CUDA Graph replay rejects ramped configurations because changed-input error exceeds the gate. Earlier CUDA
measurements used the withdrawn scheduler.

```bash
python -m pip install -e '.[cuda,eval,dev]'
python scripts/benchmark_cuda_prefill.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --tokens 128 \
  --output results/cuda_fused_prefill_128_tokens.json
```

### Concurrent CUDA decode schedule

`CUDAConcurrentRunner` implements the paper's two-stack schedule: token `t-1` replays the upper stack on one CUDA
stream while token `t` runs through the destination layer on another, and the streams join before token `t` enters
its upper stack. Persistent Python worker threads enqueue the disjoint branches concurrently. The runner is enabled
with both `GIL=1` and `GIL=0`: PyTorch releases the GIL around the CUDA operations used by the worker paths. LogBar
records the detected mode. A free-threaded build such as CPython 3.14t with `-X gil=0` or `PYTHON_GIL=0` may still
reduce host scheduling overhead. Readout remains on the token's first pass, as specified by the paper. The current API
supports batch-one, unpadded inference.

On a GIL-enabled Python 3.14.6 runtime, the current 128-token benchmark measured eager dual-stream execution at
`1.054x` faster than sequential recirculation. Capturing the full dual-stream schedule measured `5.260x` speedup over
sequential and `4.990x` over eager dual-stream execution, with zero measured logits or pending-state error, including
changed-token graph replay. See
[`results/cuda_concurrent_graph_gil1_128_tokens.md`](results/cuda_concurrent_graph_gil1_128_tokens.md). Benchmark
artifacts record the detected runtime state and implementation commit.

`CUDAGraphedConcurrentPrefill` captures the lower and replay streams, their event dependencies, and the joined upper
stack as one fixed-shape CUDA Graph. A process-wide lock covers all warmups and capture, preventing two threads from
capturing or warming CUDA graphs on the same device concurrently. Capture uses CUDA's global safety mode; replay is
not locked. The benchmark gates both the original and changed-token outputs at accumulated error `<=2e-3`.

A sweep of capture safety modes, manual/automatic instantiation, and stream priorities found no material steady-state
replay difference. High-priority (`-3`) lower/replay streams were nominally fastest and are the default, while capture
retains the safer global mode and automatic instantiation. See
[`results/cuda_graph_mode_sweep_gil1_128_tokens.md`](results/cuda_graph_mode_sweep_gil1_128_tokens.md).

```bash
python scripts/benchmark_cuda_concurrent.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --tokens 128 \
  --output results/cuda_concurrent_128_tokens.json
```

### CUDA path and alpha screening

`screen_cuda_recirculation.py` ports the MLX tuning screen to CUDA. It scores teacher-forced answer NLL, computes each
candidate's 1,078-token shared prefix once, restores an immutable candidate-specific KV snapshot for every row, and
loads the model and dataset only once. The hook-free dual-stream scheduler is the default. Its screening mode enqueues
both CUDA branches from one Python thread, avoiding futures overhead while preserving stream overlap; outer candidate
parallelism is available but defaults to one because four workers slowed the measured single-GPU workload.

```bash
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 \
NUMEXPR_NUM_THREADS=16 \
python scripts/screen_cuda_recirculation.py \
  --model /local-models/Llama-3.2-1B-Instruct \
  --row-start 272 \
  --rows 32 \
  --forbid-range 0:272 \
  --forbid-range 304:336 \
  --alpha 0.10 \
  --max-distance 12 \
  --output results/cuda_path_screen_same_token_rows272_303_alpha010.json
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
