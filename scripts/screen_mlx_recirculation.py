#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Rank recirculation paths on a dedicated tuning split without a baseline run."""

import argparse
import json
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
from recirculation.mlx_backend import CompiledNormMix, MLXCandidateGroupRecirculator
from scripts.eval_gsm8k_platinum import _gold_answer, _prompt_ids, _task_contract


def _common_prefix_length(prompts):
    length = 0
    for values in zip(*prompts):
        if len(set(values)) != 1:
            break
        length += 1
    return length


def _parse_path(value):
    source, destination = value.split(":")
    return int(source), int(destination)


def _parse_range(value):
    start, stop = value.split(":")
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("ranges must be non-empty START:STOP intervals")
    return start, stop


def _overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _candidate_batches(candidates, batch_size):
    """Keep one destination per MLX batch while preserving candidate order."""

    by_destination = {}
    for candidate in candidates:
        by_destination.setdefault(candidate[1], []).append(candidate)
    return [
        group[start : start + batch_size]
        for group in by_destination.values()
        for start in range(0, len(group), batch_size)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--row-start", type=int, default=272)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--forbid-range",
        action="append",
        type=_parse_range,
        default=None,
        help="Reject a tuning range overlapping this START:STOP interval; repeat as needed.",
    )
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--path", action="append", type=_parse_path, default=None)
    parser.add_argument("--max-distance", type=int, default=12)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=8,
        help="Evaluate this many same-destination candidates in each exact MLX shared-lower group.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate_batch_size < 1:
        parser.error("candidate-batch-size must be positive")
    tuning_range = (args.row_start, args.row_start + args.rows)
    forbidden_ranges = args.forbid_range or []
    overlaps = [interval for interval in forbidden_ranges if _overlaps(tuning_range, interval)]
    if overlaps:
        parser.error(f"tuning range {tuning_range} overlaps forbidden evaluation range(s): {overlaps}")
    model, _ = load(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fewshots, _ = _task_contract(REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    dataset = load_dataset("madrylab/gsm8k-platinum", name="main", split="test")
    documents = [dataset[index] for index in range(args.row_start, args.row_start + args.rows)]
    prompts = [_prompt_ids(tokenizer, str(document["question"]), fewshots) for document in documents]
    answers = [_gold_answer(str(document["answer"])) for document in documents]
    common_length = _common_prefix_length(prompts)
    prefix = prompts[0][:common_length]
    contexts = []
    phrase = tokenizer("The final answer is ", add_special_tokens=False).input_ids
    for prompt, answer in zip(prompts, answers):
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        contexts.append((prompt[common_length:] + phrase, answer_ids))
    paths = args.path
    if paths is None:
        paths = [
            (source, destination)
            for destination in range(1, len(model.layers))
            for source in range(destination + 1, min(len(model.layers), destination + args.max_distance + 1))
        ]
    alphas = args.alpha or [0.1]
    candidates = [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    candidate_batches = _candidate_batches(candidates, args.candidate_batch_size)
    results = []
    started = time.perf_counter()
    completed = 0
    for candidate_batch in candidate_batches:
        configs = [
            RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
            for source, destination, alpha in candidate_batch
        ]
        runner = MLXCandidateGroupRecirculator(model, configs, [CompiledNormMix(config) for config in configs])
        _, caches, pendings, _ = runner.prefill(prefix)
        snapshot = runner.snapshot(caches, pendings)
        total_nll = [0.0] * len(candidate_batch)
        answer_tokens = [0] * len(candidate_batch)
        for context, answer_ids in contexts:
            logits, caches, pendings, _ = runner.prefill_from_snapshot(context, snapshot)
            for token_index, token in enumerate(answer_ids):
                losses = [
                    mx.logsumexp(value[0, -1].astype(mx.float32)) - value[0, -1, int(token)].astype(mx.float32)
                    for value in logits
                ]
                mx.eval(*losses)
                for candidate_index, loss in enumerate(losses):
                    total_nll[candidate_index] += float(loss.item())
                    answer_tokens[candidate_index] += 1
                if token_index + 1 < len(answer_ids):
                    logits, pendings = runner.step_shared(
                        mx.array([[int(token)]], dtype=mx.int32), caches, pendings
                    )
        for candidate_index, (source, destination, alpha) in enumerate(candidate_batch):
            results.append(
                {
                    "source_layer": source,
                    "destination_layer": destination,
                    "alpha": alpha,
                    "answer_nll": total_nll[candidate_index] / answer_tokens[candidate_index],
                    "answer_tokens": answer_tokens[candidate_index],
                }
            )
        completed += len(candidate_batch)
        best = min(results, key=lambda item: item["answer_nll"])
        print(f"candidates={completed}/{len(candidates)} best={best}", flush=True)
    report = {
        "settings": {
            "split_role": "tuning",
            "row_start": args.row_start,
            "rows": args.rows,
            "row_stop_exclusive": args.row_start + args.rows,
            "forbidden_ranges": forbidden_ranges,
            "common_prefix_tokens": common_length,
            "alphas": alphas,
            "max_distance": args.max_distance,
            "candidate_batch_size": args.candidate_batch_size,
            "candidate_batches": len(candidate_batches),
        },
        "seconds": time.perf_counter() - started,
        "results": sorted(results, key=lambda item: item["answer_nll"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
