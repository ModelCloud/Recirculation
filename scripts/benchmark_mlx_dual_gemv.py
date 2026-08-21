#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the recirculation dual-GEMV Metal kernel against two MLX GEMVs."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx

from recirculation.mlx_backend import measure_forward_error
from recirculation.mlx_kernels import DualGemvMetal


def _measure(operation, warmups, repetitions):
    for _ in range(warmups):
        mx.eval(operation())
    timings = []
    output = None
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        timings.append((time.perf_counter_ns() - started) / 1e6)
    return timings, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-features", type=int, required=True)
    parser.add_argument("--out-features", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mx.random.seed(19)
    weight = mx.random.normal((args.out_features, args.in_features)).astype(mx.bfloat16)
    input0 = mx.random.normal((1, 1, args.in_features)).astype(mx.bfloat16)
    input1 = mx.random.normal((1, 1, args.in_features)).astype(mx.bfloat16)
    mx.eval(weight, input0, input1)
    kernel = DualGemvMetal()

    def reference():
        return input0 @ weight.T, input1 @ weight.T

    def candidate():
        return kernel(weight, input0, input1)

    reference_ms, reference_outputs = _measure(reference, args.warmups, args.repetitions)
    candidate_ms, candidate_outputs = _measure(candidate, args.warmups, args.repetitions)
    errors = [
        measure_forward_error(expected, actual)
        for expected, actual in zip(reference_outputs, candidate_outputs)
    ]
    for error in errors:
        error.require()
    result = {
        "device": mx.device_info(),
        "dtype": "bfloat16",
        "in_features": args.in_features,
        "out_features": args.out_features,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "reference_ms": reference_ms,
        "candidate_ms": candidate_ms,
        "reference_median_ms": statistics.median(reference_ms),
        "candidate_median_ms": statistics.median(candidate_ms),
        "speedup": statistics.median(reference_ms) / statistics.median(candidate_ms),
        "forward_errors": [error.__dict__ for error in errors],
        "forward_error_rate": max(error.rate for error in errors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
