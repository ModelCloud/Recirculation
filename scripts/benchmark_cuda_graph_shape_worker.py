#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run one crash-isolated CUDA Graph batch/prefix/allocator measurement."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAGraphedPrefill, CUDAPrefillRunner, measure_forward_error


def mib(value: int) -> float:
    return value / (1024**2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).eval().to("cuda")
    )
    prompt = "Explain why recirculation can improve transformer inference. "
    row = tokenizer(prompt * (args.tokens // 8 + 1), return_tensors="pt").input_ids[:, : args.tokens].cuda()
    token_ids = row.repeat(args.batch_size, 1)
    runner = CUDAPrefillRunner(
        model,
        RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1),
    )
    reference = runner.prefill(token_ids)[0]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    capture_started = time.perf_counter()
    graphed = CUDAGraphedPrefill(runner, token_ids, warmups=1)
    capture_ms = (time.perf_counter() - capture_started) * 1000
    candidate = graphed.prefill(token_ids)[0]
    torch.cuda.synchronize()
    error = measure_forward_error(reference, candidate)
    error.require()
    samples = []
    for _ in range(args.repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        graphed.prefill(token_ids)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    stats = torch.cuda.memory_stats()
    median_ms = statistics.median(samples)
    result = {
        "allocator_config": os.environ.get("PYTORCH_ALLOC_CONF", ""),
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "batch_size": args.batch_size,
        "tokens": args.tokens,
        "total_tokens": args.batch_size * args.tokens,
        "repetitions": args.repetitions,
        "capture_ms": capture_ms,
        "samples_ms": samples,
        "median_ms": median_ms,
        "tokens_per_second": args.batch_size * args.tokens * 1000 / median_ms,
        "forward_error": error.__dict__,
        "forward_error_rate": error.rate,
        "baseline_allocated_mib": mib(baseline_allocated),
        "baseline_reserved_mib": mib(baseline_reserved),
        "current_allocated_mib": mib(torch.cuda.memory_allocated()),
        "current_reserved_mib": mib(torch.cuda.memory_reserved()),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        "inactive_split_current_mib": mib(stats.get("inactive_split_bytes.all.current", 0)),
        "inactive_split_peak_mib": mib(stats.get("inactive_split_bytes.all.peak", 0)),
        "device_free_mib": mib(torch.cuda.mem_get_info()[0]),
        "device_total_mib": mib(torch.cuda.mem_get_info()[1]),
    }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
