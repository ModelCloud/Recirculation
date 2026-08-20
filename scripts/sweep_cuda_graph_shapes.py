#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep CUDA Graph batch/prefix shapes and allocator/GC policies in isolated workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from logbar import LogBar

LOG = LogBar.shared()
REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOCATORS = {
    "native_default": "backend:native",
    "native_graph_reuse_gc08": (
        "backend:native,expandable_segments:True,garbage_collection_threshold:0.8,"
        "roundup_power2_divisions:4,graph_capture_record_stream_reuse:True"
    ),
    "native_fragment_guard_gc06": (
        "backend:native,garbage_collection_threshold:0.6,max_split_size_mb:512,"
        "max_non_split_rounding_mb:1024,roundup_power2_divisions:4,graph_capture_record_stream_reuse:True"
    ),
    "cuda_malloc_async": "backend:cudaMallocAsync",
}


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def implementation_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokens", type=parse_int_list, default=parse_int_list("128,256,512"))
    parser.add_argument("--batch-sizes", type=parse_int_list, default=parse_int_list("1,4,8,16,32"))
    parser.add_argument("--allocator", action="append", choices=tuple(ALLOCATORS), default=None)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocator_names = args.allocator or list(ALLOCATORS)
    combinations = [
        (allocator_name, tokens, batch_size)
        for allocator_name in allocator_names
        for tokens in args.tokens
        for batch_size in args.batch_sizes
    ]
    results = []
    started = time.perf_counter()
    worker = REPO_ROOT / "scripts/benchmark_cuda_graph_shape_worker.py"
    for index, (allocator_name, tokens, batch_size) in enumerate(combinations, start=1):
        environment = os.environ.copy()
        environment["PYTORCH_ALLOC_CONF"] = ALLOCATORS[allocator_name]
        environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        command = [
            sys.executable,
            str(worker),
            "--model",
            args.model,
            "--tokens",
            str(tokens),
            "--batch-size",
            str(batch_size),
            "--repetitions",
            str(args.repetitions),
        ]
        case_started = time.perf_counter()
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True, check=False)
        case = {
            "allocator": allocator_name,
            "allocator_config": ALLOCATORS[allocator_name],
            "tokens": tokens,
            "batch_size": batch_size,
            "wall_seconds": time.perf_counter() - case_started,
            "returncode": completed.returncode,
        }
        if completed.returncode == 0:
            try:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                case.update(payload)
                case["status"] = "passed"
            except (IndexError, json.JSONDecodeError) as error:
                case.update(status="invalid_output", error=str(error), stdout=completed.stdout[-2000:])
        else:
            case.update(
                status="signal" if completed.returncode < 0 else "failed",
                signal=-completed.returncode if completed.returncode < 0 else None,
                stderr=completed.stderr[-4000:],
                stdout=completed.stdout[-2000:],
            )
        results.append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = {
            "status": "running",
            "implementation_commit": implementation_commit(),
            "complete": len(results),
            "active": None,
            "pending": len(combinations) - len(results),
            "results": results,
        }
        args.output.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metric = f"{case.get('tokens_per_second', 0):.1f} tok/s" if case["status"] == "passed" else case["status"]
        LOG.info(
            f"complete={index}/{len(combinations)} active=0 pending={len(combinations) - index} "
            f"allocator={allocator_name} batch={batch_size} tokens={tokens} result={metric}"
        )
    successful = [case for case in results if case["status"] == "passed"]
    best = max(successful, key=lambda case: case["tokens_per_second"]) if successful else None
    report = {
        "status": "complete",
        "implementation_commit": implementation_commit(),
        "seconds": time.perf_counter() - started,
        "tokens": args.tokens,
        "batch_sizes": args.batch_sizes,
        "allocators": {name: ALLOCATORS[name] for name in allocator_names},
        "best": best,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
