#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact reuse of a recirculated MLX prefix snapshot."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from recirculation import RecirculationConfig
from recirculation.mlx_backend import CompiledNormMix, MLXRecirculator, measure_forward_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--suffix-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    runner = MLXRecirculator(model, config, CompiledNormMix(config))
    prefix = (tokenizer.encode("Shared eight shot reasoning context. ") * args.prefix_tokens)[: args.prefix_tokens]
    suffixes = [
        (tokenizer.encode(f"Question {index}: calculate a unique answer. ") * args.suffix_tokens)[: args.suffix_tokens]
        for index in range(args.requests)
    ]

    def full_prefill():
        output = mx.concatenate([runner.prefill(prefix + suffix, collect_logits=True)[0] for suffix in suffixes])
        mx.eval(output)
        return output

    def cached_prefill():
        _, cache, pending_source, _ = runner.prefill(prefix, collect_logits=True)
        snapshot = runner.snapshot(cache, pending_source)
        output = mx.concatenate(
            [runner.prefill_from_snapshot(suffix, snapshot, collect_logits=True)[0] for suffix in suffixes]
        )
        mx.eval(output)
        return output

    full_prefill()
    cached_prefill()
    full_ms = []
    cached_ms = []
    reference = candidate = None
    for _ in range(args.repetitions):
        started = time.perf_counter_ns()
        reference = full_prefill()
        full_ms.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        candidate = cached_prefill()
        cached_ms.append((time.perf_counter_ns() - started) / 1e6)
    error = measure_forward_error(reference, candidate)
    error.require()
    result = {
        "requests": args.requests,
        "prefix_tokens": args.prefix_tokens,
        "suffix_tokens": args.suffix_tokens,
        "repetitions": args.repetitions,
        "full_ms": full_ms,
        "cached_ms": cached_ms,
        "full_median_ms": statistics.median(full_ms),
        "cached_median_ms": statistics.median(cached_ms),
        "speedup": statistics.median(full_ms) / statistics.median(cached_ms),
        "forward_error": error.__dict__,
        "forward_error_rate": error.rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
