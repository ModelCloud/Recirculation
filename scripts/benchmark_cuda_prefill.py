#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark reference and fused CUDA recirculation prefill with an error gate."""

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
from recirculation.cuda_backend import CUDAGraphedPrefill, CUDAPrefillRunner, measure_forward_error


def time_prefill(runner, tokens, repetitions):
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner.prefill(tokens)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--source-layer", type=int, default=12)
    parser.add_argument("--destination-layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
    )
    reference = CUDAPrefillRunner(model, config, fused=False, skip_intermediate_logits=False)
    fused = CUDAPrefillRunner(model, config, fused=True)

    reference_logits = reference.prefill(token_ids, collect_logits=True)[3]
    fused_logits = fused.prefill(token_ids, collect_logits=True)[3]
    error = measure_forward_error(reference_logits, fused_logits)
    error.require()
    graphed = CUDAGraphedPrefill(fused, token_ids)
    graph_logits = graphed.prefill(token_ids)[0]
    graph_error = measure_forward_error(reference_logits[:, -1:, :], graph_logits)
    graph_error.require()
    replay_tokens = torch.roll(token_ids, shifts=1, dims=1)
    replay_reference = fused.prefill(replay_tokens)
    replay_candidate = graphed.prefill(replay_tokens)
    replay_input_error = measure_forward_error(replay_reference[0], replay_candidate[0])
    replay_input_error.require()
    replay_pending_reference = torch.cat((replay_reference[2].destination, replay_reference[2].source), dim=-1)
    replay_pending_candidate = torch.cat((replay_candidate[2].destination, replay_candidate[2].source), dim=-1)
    replay_pending_error = measure_forward_error(replay_pending_reference, replay_pending_candidate)
    replay_pending_error.require()
    if replay_reference[2].token_position != replay_candidate[2].token_position:
        raise RuntimeError("graphed pending token position differs from the eager CUDA reference")
    reference.prefill(token_ids)
    reference_ms = time_prefill(reference, token_ids, args.repetitions)
    optimized_ms = time_prefill(graphed, token_ids, args.repetitions)
    result = {
        "model": str(args.model),
        "scheduler": "same-token replay with upper-layer KV replacement",
        "tokens": token_ids.shape[1],
        "repetitions": args.repetitions,
        "reference_ms": reference_ms,
        "optimized_ms": optimized_ms,
        "reference_median_ms": statistics.median(reference_ms),
        "optimized_median_ms": statistics.median(optimized_ms),
        "speedup": statistics.median(reference_ms) / statistics.median(optimized_ms),
        "forward_error": error.__dict__,
        "forward_error_limit": 2e-3,
        "forward_error_rate": error.rate,
        "graphed_final_error": graph_error.__dict__,
        "graphed_final_error_rate": graph_error.rate,
        "graphed_new_input_error": replay_input_error.__dict__,
        "graphed_new_input_error_rate": replay_input_error.rate,
        "graphed_new_input_pending_error": replay_pending_error.__dict__,
        "graphed_new_input_pending_error_rate": replay_pending_error.rate,
        "graphed_pending_token_position": replay_candidate[2].token_position,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
