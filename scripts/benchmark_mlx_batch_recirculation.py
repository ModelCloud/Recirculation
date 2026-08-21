#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark contiguous MLX recirculation batching against scalar execution."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig
from recirculation.mlx_backend import (
    CompiledNormMix,
    MLXBatchedRecirculator,
    MLXRecirculator,
    measure_forward_error,
)


def _synchronize() -> None:
    mx.synchronize()


def _prompt(tokenizer, token_count: int) -> list[int]:
    text = (
        "Explain how a recurrent state can improve transformer inference while preserving the ordinary readout. "
        * 64
    )
    tokens = list(tokenizer.encode(text))
    if len(tokens) < token_count:
        raise ValueError(f"tokenizer produced only {len(tokens)} tokens; reduce --prompt-tokens")
    return [int(token) for token in tokens[:token_count]]


def _scalar_generation(model, config, prompts, max_new_tokens: int) -> None:
    for prompt in prompts:
        runner = MLXRecirculator(model, config, CompiledNormMix(config))
        logits, cache, pending, _ = runner.prefill(prompt)
        token = int(mx.argmax(logits[0, -1]).item())
        for _ in range(max_new_tokens):
            logits, pending = runner.step(
                mx.array([[token]], dtype=mx.int32),
                cache,
                pending,
            )
            mx.eval(logits, pending.destination, pending.source)
            token = int(mx.argmax(logits[0, -1]).item())


def _batched_generation(model, config, prompts, max_new_tokens: int) -> None:
    runner = MLXBatchedRecirculator(model, config)
    logits, cache, pending, _ = runner.prefill(
        mx.array(prompts, dtype=mx.int32),
    )
    tokens = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
    mx.eval(tokens, pending.destination, pending.source)
    for _ in range(max_new_tokens):
        logits, pending = runner.step(tokens[:, None], cache, pending)
        mx.eval(logits, pending.destination, pending.source)
        tokens = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(tokens)


def _accuracy_gate(model, config, prompts, max_new_tokens: int) -> float:
    scalar_states = []
    for prompt in prompts:
        runner = MLXRecirculator(model, config, CompiledNormMix(config))
        logits, cache, pending, _ = runner.prefill(prompt)
        scalar_states.append(
            (runner, cache, pending, logits, int(mx.argmax(logits[0, -1]).item()))
        )

    batch_runner = MLXBatchedRecirculator(model, config)
    batch_logits, batch_cache, batch_pending, _ = batch_runner.prefill(
        mx.array(prompts, dtype=mx.int32),
    )
    errors = []
    for row, (_, _, pending, logits, _) in enumerate(scalar_states):
        errors.append(measure_forward_error(logits, batch_logits[row : row + 1]))
        errors.append(
            measure_forward_error(
                pending.destination,
                batch_pending.destination[row : row + 1],
            )
        )

    scalar_tokens = mx.array(
        [[state[4]] for state in scalar_states],
        dtype=mx.int32,
    )
    for _ in range(max_new_tokens):
        scalar_logits = []
        for row, (runner, cache, pending, _, token) in enumerate(scalar_states):
            logits, next_pending = runner.step(
                mx.array([[token]], dtype=mx.int32),
                cache,
                pending,
            )
            mx.eval(logits, next_pending.destination, next_pending.source)
            scalar_logits.append(logits)
            scalar_states[row] = (
                runner,
                cache,
                next_pending,
                logits,
                int(mx.argmax(logits[0, -1]).item()),
            )
        batch_logits, batch_pending = batch_runner.step(
            scalar_tokens,
            batch_cache,
            batch_pending,
        )
        mx.eval(batch_logits, batch_pending.destination, batch_pending.source)
        scalar_tokens = mx.array(
            [[state[4]] for state in scalar_states],
            dtype=mx.int32,
        )
        for row, logits in enumerate(scalar_logits):
            errors.append(measure_forward_error(logits, batch_logits[row : row + 1]))

    error_rate = max(error.rate for error in errors)
    for error in errors:
        error.require()
    return error_rate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
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
    source, destination = map(int, args.path.split(":"))

    model, tokenizer = load(args.model)
    prompt = _prompt(tokenizer, args.prompt_tokens)
    config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=args.alpha)
    results = []
    for width in widths:
        prompts = [prompt] * width
        forward_error_rate = _accuracy_gate(model, config, prompts, args.max_new_tokens)
        for _ in range(args.warmups):
            _scalar_generation(model, config, prompts, args.max_new_tokens)
            _batched_generation(model, config, prompts, args.max_new_tokens)
        _synchronize()
        scalar_samples = []
        batched_samples = []
        for _ in range(args.repetitions):
            started = time.perf_counter()
            _scalar_generation(model, config, prompts, args.max_new_tokens)
            _synchronize()
            scalar_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            _batched_generation(model, config, prompts, args.max_new_tokens)
            _synchronize()
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
                "forward_error_rate": forward_error_rate,
            }
        )

    report = {
        "backend": "mlx-contiguous-batch",
        "model": args.model,
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
