#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark projecting only the final serial-prefill token to logits."""

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
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, tokenizer = load(args.model)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    runner = MLXRecirculator(model, config, CompiledNormMix(config))
    tokens = (tokenizer.encode("Recirculation tracks state across tokens. ") * args.tokens)[: args.tokens]

    def run(collect_logits):
        started = time.perf_counter_ns()
        logits, _, _, trace = runner.prefill(tokens, collect_logits=collect_logits)
        mx.eval(trace)
        return (time.perf_counter_ns() - started) / 1e6, logits

    run(True)
    run(False)
    all_logits_ms = []
    final_only_ms = []
    reference = candidate = None
    for _ in range(args.repetitions):
        elapsed, reference = run(True)
        all_logits_ms.append(elapsed)
        elapsed, candidate = run(False)
        final_only_ms.append(elapsed)
    error = measure_forward_error(reference, candidate)
    error.require()
    result = {
        "tokens": args.tokens,
        "repetitions": args.repetitions,
        "all_logits_ms": all_logits_ms,
        "final_only_ms": final_only_ms,
        "all_logits_median_ms": statistics.median(all_logits_ms),
        "final_only_median_ms": statistics.median(final_only_ms),
        "speedup": statistics.median(all_logits_ms) / statistics.median(final_only_ms),
        "forward_error": error.__dict__,
        "forward_error_rate": error.rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
