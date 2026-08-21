#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Measure recirculation throughput scaling for identical prompt batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAConcurrentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--path", default="10:1")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--width", action="append", type=int, default=None)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    widths = args.width or [1, 2, 4, 8, 16, 32]
    if min(widths + [args.prompt_tokens, args.new_tokens, args.repetitions]) < 1 or args.warmups < 0:
        parser.error("widths, token counts, and repetitions must be positive; warmups must be non-negative")
    source, destination = map(int, args.path.split(":"))

    tokenizer = Tokenicer.load(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    prompt = tokenizer(
        "Explain why recurrent state can improve transformer inference. " * 64,
        return_tensors="pt",
    ).input_ids[:, : args.prompt_tokens].to("cuda")
    runner = CUDAConcurrentRunner(
        model,
        RecirculationConfig(source_layer=source, destination_layer=destination, alpha=args.alpha),
        use_python_threads=True,
    )
    results = []
    reference_tokens = None
    try:
        for width in widths:
            batch = prompt.expand(width, -1).contiguous()
            for _ in range(args.warmups):
                runner.generate(batch, max_new_tokens=args.new_tokens, eos_token_id=None)
            torch.cuda.synchronize()
            samples = []
            generated = None
            for _ in range(args.repetitions):
                torch.cuda.synchronize()
                started = time.perf_counter()
                generated = runner.generate(batch, max_new_tokens=args.new_tokens, eos_token_id=None)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            continuation = generated[:, args.prompt_tokens:].detach().cpu()
            rows_identical = bool(torch.equal(continuation, continuation[:1].expand_as(continuation)))
            if reference_tokens is None:
                reference_tokens = continuation[0].clone()
            reference_match = bool(torch.equal(continuation[0], reference_tokens))
            median = statistics.median(samples)
            results.append(
                {
                    "width": width,
                    "seconds": samples,
                    "median_seconds": median,
                    "aggregate_generated_tokens_per_second": width * args.new_tokens / median,
                    "requests_per_second": width / median,
                    "rows_identical": rows_identical,
                    "width1_exact_token_match": reference_match,
                    "continuation_sha256": hashlib.sha256(continuation[0].numpy().tobytes()).hexdigest(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            torch.cuda.reset_peak_memory_stats()
    finally:
        runner.close()

    report = {
        "model": args.model,
        "dtype": "float16",
        "attention": "sdpa",
        "source_layer": source,
        "destination_layer": destination,
        "alpha": args.alpha,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
