#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark paper-faithful two-stack CUDA concurrency against sequential CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM

from recirculation import RecirculationConfig
from recirculation.cuda_backend import (
    CUDAConcurrentRunner,
    CUDAGraphedConcurrentPrefill,
    CUDAPrefillRunner,
    log_concurrency_mode,
    measure_forward_error,
)


def time_prefill(runner, tokens, repetitions):
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        runner.prefill(tokens)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def implementation_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--source-layer", type=int, default=12)
    parser.add_argument("--destination-layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--ramp-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    log_concurrency_mode()

    tokenizer = Tokenicer.load(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).eval().to("cuda")
    )
    prompt = "Explain why recirculation can improve transformer inference. "
    token_ids = tokenizer(prompt * (args.tokens // 8 + 1), return_tensors="pt").input_ids[:, : args.tokens].cuda()
    config = RecirculationConfig(
        source_layer=args.source_layer,
        destination_layer=args.destination_layer,
        alpha=args.alpha,
        ramp_tokens=args.ramp_tokens,
    )
    sequential = CUDAPrefillRunner(model, config)
    concurrent = CUDAConcurrentRunner(model, config)
    try:
        sequential_output = sequential.prefill(token_ids, collect_logits=True)
        concurrent_output = concurrent.prefill(token_ids, collect_logits=True)
        logits_error = measure_forward_error(sequential_output[3], concurrent_output[3])
        logits_error.require()
        pending_reference = torch.cat((sequential_output[2].destination, sequential_output[2].source), dim=-1)
        pending_candidate = torch.cat((concurrent_output[2].destination, concurrent_output[2].source), dim=-1)
        pending_error = measure_forward_error(pending_reference, pending_candidate)
        pending_error.require()
        if sequential_output[2].token_position != concurrent_output[2].token_position:
            raise RuntimeError("concurrent pending token position differs from sequential CUDA")

        graphed = CUDAGraphedConcurrentPrefill(concurrent, token_ids)
        graphed_output = graphed.prefill(token_ids)
        graphed_logits_error = measure_forward_error(concurrent_output[0], graphed_output[0])
        graphed_logits_error.require()
        graphed_pending_reference = torch.cat(
            (concurrent_output[2].destination, concurrent_output[2].source), dim=-1
        )
        graphed_pending_candidate = torch.cat((graphed_output[2].destination, graphed_output[2].source), dim=-1)
        graphed_pending_error = measure_forward_error(graphed_pending_reference, graphed_pending_candidate)
        graphed_pending_error.require()
        replay_tokens = torch.roll(token_ids, shifts=1, dims=1)
        replay_reference = concurrent.prefill(replay_tokens)
        replay_candidate = graphed.prefill(replay_tokens)
        graphed_new_input_error = measure_forward_error(replay_reference[0], replay_candidate[0])
        graphed_new_input_error.require()
        replay_pending_reference = torch.cat(
            (replay_reference[2].destination, replay_reference[2].source), dim=-1
        )
        replay_pending_candidate = torch.cat(
            (replay_candidate[2].destination, replay_candidate[2].source), dim=-1
        )
        graphed_new_input_pending_error = measure_forward_error(
            replay_pending_reference, replay_pending_candidate
        )
        graphed_new_input_pending_error.require()

        sequential.prefill(token_ids)
        concurrent.prefill(token_ids)
        graphed.prefill(token_ids)
        sequential_ms = time_prefill(sequential, token_ids, args.repetitions)
        concurrent_ms = time_prefill(concurrent, token_ids, args.repetitions)
        graphed_ms = time_prefill(graphed, token_ids, args.repetitions)
        result = {
            "implementation_commit": implementation_commit(),
            "model": str(args.model),
            "tokens": token_ids.shape[1],
            "repetitions": args.repetitions,
            "source_layer": args.source_layer,
            "destination_layer": args.destination_layer,
            "alpha": args.alpha,
            "ramp_tokens": args.ramp_tokens,
            "python_gil_enabled": concurrent.gil_enabled,
            "python_worker_threads": 2,
            "cuda_streams": 2,
            "sequential_ms": sequential_ms,
            "concurrent_ms": concurrent_ms,
            "graphed_ms": graphed_ms,
            "sequential_median_ms": statistics.median(sequential_ms),
            "concurrent_median_ms": statistics.median(concurrent_ms),
            "graphed_median_ms": statistics.median(graphed_ms),
            "speedup": statistics.median(sequential_ms) / statistics.median(concurrent_ms),
            "graphed_speedup_vs_sequential": statistics.median(sequential_ms) / statistics.median(graphed_ms),
            "graphed_speedup_vs_eager_concurrent": statistics.median(concurrent_ms) / statistics.median(graphed_ms),
            "logits_error": logits_error.__dict__,
            "logits_error_rate": logits_error.rate,
            "pending_error": pending_error.__dict__,
            "pending_error_rate": pending_error.rate,
            "pending_token_position": concurrent_output[2].token_position,
            "graphed_logits_error": graphed_logits_error.__dict__,
            "graphed_logits_error_rate": graphed_logits_error.rate,
            "graphed_pending_error": graphed_pending_error.__dict__,
            "graphed_pending_error_rate": graphed_pending_error.rate,
            "graphed_new_input_error": graphed_new_input_error.__dict__,
            "graphed_new_input_error_rate": graphed_new_input_error.rate,
            "graphed_new_input_pending_error": graphed_new_input_pending_error.__dict__,
            "graphed_new_input_pending_error_rate": graphed_new_input_pending_error.rate,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        concurrent.close()


if __name__ == "__main__":
    main()
