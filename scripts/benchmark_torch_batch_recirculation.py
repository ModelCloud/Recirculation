#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark dense Torch/MPS recirculation batching against scalar execution."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig, RecirculationController


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _prompt(tokenizer, token_count: int, device: torch.device) -> torch.Tensor:
    text = (
        "Explain how a recurrent state can improve transformer inference while preserving the ordinary readout. "
        * 64
    )
    encoded = tokenizer(text, return_tensors="pt").input_ids[0]
    if encoded.numel() < token_count:
        raise ValueError(f"tokenizer produced only {encoded.numel()} tokens; reduce --prompt-tokens")
    return encoded[:token_count].to(device=device, dtype=torch.long)


def _scalar_generation(model, config, prompts, max_new_tokens: int) -> torch.Tensor:
    return torch.cat(
        [
            RecirculationController(model, config).generate(
                prompt[None, :],
                max_new_tokens=max_new_tokens,
                eos_token_id=None,
            )
            for prompt in prompts
        ],
        dim=0,
    )


def _batched_generation(model, config, prompts, max_new_tokens: int) -> torch.Tensor:
    return RecirculationController(model, config).generate(
        torch.stack(prompts),
        max_new_tokens=max_new_tokens,
        attention_mask=torch.ones_like(torch.stack(prompts)),
        eos_token_id=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--path", default="12:5")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--width", action="append", type=int, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = args.width or [1, 2, 4, 8]
    if min(widths + [args.prompt_tokens, args.max_new_tokens, args.repetitions]) < 1 or args.warmups < 0:
        parser.error("widths, token counts, and repetitions must be positive; warmups must be non-negative")

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    source, destination = map(int, args.path.split(":"))
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    tokenizer = Tokenicer.load(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    ).eval().to(device)
    prompt = _prompt(tokenizer, args.prompt_tokens, device)
    config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=args.alpha)
    results = []
    for width in widths:
        prompts = [prompt] * width
        scalar_output = _scalar_generation(model, config, prompts, args.max_new_tokens)
        batched_output = _batched_generation(model, config, prompts, args.max_new_tokens)
        exact_match = bool(torch.equal(scalar_output, batched_output))
        if not exact_match:
            raise RuntimeError(f"dense batch token mismatch at width {width}")
        for _ in range(args.warmups):
            _scalar_generation(model, config, prompts, args.max_new_tokens)
            _batched_generation(model, config, prompts, args.max_new_tokens)
        _synchronize(device)
        scalar_samples = []
        batched_samples = []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            _scalar_generation(model, config, prompts, args.max_new_tokens)
            _synchronize(device)
            scalar_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            _batched_generation(model, config, prompts, args.max_new_tokens)
            _synchronize(device)
            batched_samples.append(time.perf_counter() - started)
        scalar_median = statistics.median(scalar_samples)
        batched_median = statistics.median(batched_samples)
        speedup = scalar_median / batched_median
        if speedup < args.min_speedup:
            raise RuntimeError(f"width {width} speedup {speedup:.3f} is below required {args.min_speedup:.3f}")
        results.append(
            {
                "width": width,
                "scalar_ms": [sample * 1000 for sample in scalar_samples],
                "batched_ms": [sample * 1000 for sample in batched_samples],
                "scalar_median_ms": scalar_median * 1000,
                "batched_median_ms": batched_median * 1000,
                "speedup": speedup,
                "scalar_tokens_per_second": width * args.max_new_tokens / scalar_median,
                "batched_tokens_per_second": width * args.max_new_tokens / batched_median,
                "exact_token_match": exact_match,
            }
        )

    report = {
        "backend": f"torch-{device.type}-dense-batch",
        "model": args.model,
        "dtype": str(dtype),
        "source_layer": source,
        "destination_layer": destination,
        "alpha": args.alpha,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
