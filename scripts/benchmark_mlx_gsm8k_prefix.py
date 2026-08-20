#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Benchmark exact recirculated prefix reuse on real GSM8K-Platinum prompts."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
from datasets import load_dataset
from mlx_lm import load
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig
from recirculation.mlx_backend import CompiledNormMix, MLXRecirculator, measure_forward_error
from scripts.eval_gsm8k_platinum import _prompt_ids, _task_contract


def _common_prefix_length(prompts):
    length = 0
    for values in zip(*prompts):
        if len(set(values)) != 1:
            break
        length += 1
    return length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, _ = load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fewshots, _ = _task_contract(REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    dataset = load_dataset("madrylab/gsm8k-platinum", name="main", split="test")
    prompts = [_prompt_ids(tokenizer, str(dataset[index]["question"]), fewshots) for index in range(args.rows)]
    common_length = _common_prefix_length(prompts)
    prefix = prompts[0][:common_length]
    suffixes = [prompt[common_length:] for prompt in prompts]
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    runner = MLXRecirculator(model, config, CompiledNormMix(config))

    def full_prefill():
        output = mx.concatenate([runner.prefill(prompt)[0] for prompt in prompts])
        mx.eval(output)
        return output

    def cached_prefill():
        _, cache, pending_source, _ = runner.prefill(prefix)
        snapshot = runner.snapshot(cache, pending_source)
        output = mx.concatenate([runner.prefill_from_snapshot(suffix, snapshot)[0] for suffix in suffixes])
        mx.eval(output)
        return output

    full_prefill()
    cached_prefill()
    full_ms = []
    cached_ms = []
    reference = candidate = None
    for _ in range(args.repetitions):
        started = time.perf_counter_ns()
        reference = full_prefill()
        full_ms.append((time.perf_counter_ns() - started) / 1e6)
        started = time.perf_counter_ns()
        candidate = cached_prefill()
        cached_ms.append((time.perf_counter_ns() - started) / 1e6)
    error = measure_forward_error(reference, candidate)
    error.require()
    result = {
        "rows": args.rows,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "common_prefix_tokens": common_length,
        "suffix_lengths": [len(suffix) for suffix in suffixes],
        "repetitions": args.repetitions,
        "full_ms": full_ms,
        "cached_ms": cached_ms,
        "full_median_ms": statistics.median(full_ms),
        "cached_median_ms": statistics.median(cached_ms),
        "speedup": statistics.median(full_ms) / statistics.median(cached_ms),
        "forward_error": error.__dict__,
        "forward_error_rate": error.rate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
