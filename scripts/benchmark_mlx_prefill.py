#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact and compiled MLX recirculation prefill with an error gate."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from recirculation import RecirculationConfig
from recirculation.mlx_backend import CompiledNormMix, MLXRecirculator, measure_forward_error


def _time_prefill(runner, tokens, repetitions):
    times = []
    trace = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        *_, trace = runner.prefill(tokens, collect_logits=True)
        mx.eval(trace)
        times.append((time.perf_counter_ns() - started) / 1e6)
    return times, trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    token_ids = (tokenizer.encode("Recirculation tracks state across tokens. ") * args.tokens)[: args.tokens]
    reference = MLXRecirculator(model, config)
    compiled = MLXRecirculator(model, config, CompiledNormMix(config))
    reference.prefill(token_ids[: args.warmup_tokens])
    compiled.prefill(token_ids[: args.warmup_tokens])
    reference_ms, reference_trace = _time_prefill(reference, token_ids, args.repetitions)
    compiled_ms, compiled_trace = _time_prefill(compiled, token_ids, args.repetitions)
    error = measure_forward_error(reference_trace, compiled_trace)
    error.require()
    result = {
        "tokens": args.tokens,
        "repetitions": args.repetitions,
        "reference_ms": reference_ms,
        "compiled_ms": compiled_ms,
        "reference_median_ms": statistics.median(reference_ms),
        "compiled_median_ms": statistics.median(compiled_ms),
        "speedup": statistics.median(reference_ms) / statistics.median(compiled_ms),
        "forward_error": error.__dict__,
        "forward_error_rate": error.rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
