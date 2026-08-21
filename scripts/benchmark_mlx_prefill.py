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
from recirculation.mlx_backend import (
    CompiledNormMix,
    MLXQwen3DualGemvRecirculator,
    MLXRecirculator,
    measure_forward_error,
)


def _time_prefill(runner, tokens, repetitions):
    times = []
    result = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        result = runner.prefill(tokens, collect_logits=True)
        mx.eval(result[3])
        times.append((time.perf_counter_ns() - started) / 1e6)
    return times, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--dual-gemv", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    token_ids = (tokenizer.encode("Recirculation tracks state across tokens. ") * args.tokens)[: args.tokens]
    reference = MLXRecirculator(model, config)
    compiled = MLXRecirculator(model, config, CompiledNormMix(config))
    dual = MLXQwen3DualGemvRecirculator(model, config, CompiledNormMix(config)) if args.dual_gemv else None
    reference.prefill(token_ids[: args.warmup_tokens])
    compiled.prefill(token_ids[: args.warmup_tokens])
    if dual is not None:
        dual.prefill(token_ids[: args.warmup_tokens])
    reference_ms, reference_result = _time_prefill(reference, token_ids, args.repetitions)
    compiled_ms, compiled_result = _time_prefill(compiled, token_ids, args.repetitions)
    error = measure_forward_error(reference_result[3], compiled_result[3])
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
    if dual is not None:
        dual_ms, dual_result = _time_prefill(dual, token_ids, args.repetitions)
        dual_errors = [
            measure_forward_error(compiled_result[3], dual_result[3]),
            measure_forward_error(compiled_result[2].destination, dual_result[2].destination),
            measure_forward_error(compiled_result[2].source, dual_result[2].source),
        ]
        for reference_cache, candidate_cache in zip(compiled_result[1], dual_result[1]):
            dual_errors.extend(
                measure_forward_error(reference_value, candidate_value)
                for reference_value, candidate_value in zip(reference_cache.state, candidate_cache.state)
            )
        for dual_error in dual_errors:
            dual_error.require()
        result.update(
            {
                "dual_gemv_ms": dual_ms,
                "dual_gemv_median_ms": statistics.median(dual_ms),
                "dual_gemv_speedup": statistics.median(compiled_ms) / statistics.median(dual_ms),
                "dual_gemv_forward_error_rate": max(dual_error.rate for dual_error in dual_errors),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
