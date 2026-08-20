# Recirculation reproduction

## Progress

- 2026-08-20 — Ramping now follows the paper's zero-based 10-step schedule exactly.
- 2026-08-20 — Corrected CUDA same-token replay reached **4.533x prefill speedup** within the `2e-3` error gate.
- 2026-08-20 — Corrected recirculation to replay each token's own upper stack and replace its upper-layer KV state.
- 2026-08-20 — With zero feedback, replay now matches ordinary serial inference exactly in Torch and MLX.

This repository contains an independent, inference-only implementation and validation of
[Recirculation](https://arxiv.org/abs/2608.17981). It is not the authors' official implementation.

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
