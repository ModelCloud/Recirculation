#!/usr/bin/env python3
"""Warm CUDA autoregressive decode benchmark for the recirculation runner."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = Tokenicer.load(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to("cuda")
    prompt = tokenizer(
        "Explain why recirculation can improve transformer inference. " * 64,
        return_tensors="pt",
    ).input_ids[:, : args.prompt_tokens].cuda()
    config = RecirculationConfig(source_layer=8, destination_layer=2, alpha=0.2)
    results = {}
    for threaded in (True, False):
        runner = CUDAConcurrentRunner(model, config, use_python_threads=threaded)
        try:
            runner.generate(prompt, max_new_tokens=2, eos_token_id=None)
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repetitions):
                torch.cuda.synchronize()
                started = time.perf_counter()
                runner.generate(prompt, max_new_tokens=args.new_tokens, eos_token_id=None)
                torch.cuda.synchronize()
                samples.append(time.perf_counter() - started)
            median = statistics.median(samples)
            results["threaded" if threaded else "direct"] = {
                "seconds": samples,
                "median_seconds": median,
                "tokens_per_second": args.new_tokens / median,
            }
        finally:
            runner.close()
    report = {
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "repetitions": args.repetitions,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
