#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Screen Llama recirculation source/destination pairs on GSM8K-Platinum."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from evalution.scorers.gsm8k import numbers_equal
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import (
    RecirculationConfig,
    RecirculationController,
)
from recirculation.screening import paired_selection_entry
from scripts.eval_gsm8k_platinum import (
    _generate,
    _gold_answer,
    _prompt_ids,
    _summary,
    _task_contract,
)


def _parse_candidate(value: str) -> tuple[int, int, float]:
    try:
        source, destination, alpha = value.split(":")
        return int(source), int(destination), float(alpha)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be SOURCE:DESTINATION:ALPHA") from exc


def _paired_candidate_summary(samples, reference_arm: str, candidate_arm: str):
    result = {}
    for answer_key, label in (
        ("numeric_answer", "numeric"),
        ("strict_answer", "strict"),
        ("flexible_answer", "flexible"),
    ):
        changes = wrong_to_correct = correct_to_wrong = 0
        for sample in samples:
            reference = sample[reference_arm][answer_key]
            candidate = sample[candidate_arm][answer_key]
            gold = sample["gold_answer"]
            changes += reference != candidate
            if label == "numeric":
                reference_correct = numbers_equal(reference, gold)
                candidate_correct = numbers_equal(candidate, gold)
            else:
                reference_correct = reference == gold
                candidate_correct = candidate == gold
            wrong_to_correct += not reference_correct and candidate_correct
            correct_to_wrong += reference_correct and not candidate_correct
        result[label] = {
            "answer_changes": changes,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct": wrong_to_correct - correct_to_wrong,
        }
    return result


def _implementation_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--row-start", type=int, default=128)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=None,
        help="Load the top candidates from a robust CUDA screen instead of passing --candidate repeatedly.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--harm-weight",
        type=float,
        default=2.0,
        help="Selection penalty applied to each baseline correct-to-wrong regression.",
    )
    parser.add_argument(
        "--max-correct-to-wrong",
        type=int,
        default=None,
        help="Optional hard E2E validity gate for baseline correct-to-wrong regressions.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        help="Repeat SOURCE:DESTINATION:ALPHA; defaults to four middle-stack pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.screen_results is not None and args.candidate is not None:
        parser.error("use either --screen-results or --candidate, not both")
    if args.top_k < 1 or args.harm_weight < 1.0:
        parser.error("top-k must be positive and harm-weight must be at least 1")
    if args.max_correct_to_wrong is not None and args.max_correct_to_wrong < 0:
        parser.error("max-correct-to-wrong must be non-negative")
    if args.screen_results is not None:
        screen = json.loads(args.screen_results.read_text(encoding="utf-8"))
        candidates = screen.get("results", [])[: args.top_k]
        if not candidates:
            parser.error("screen-results does not contain candidates")
        args.candidate = [
            (int(item["source_layer"]), int(item["destination_layer"]), float(item["alpha"])) for item in candidates
        ]
    elif args.candidate is None:
        args.candidate = [(10, 3, 0.10), (11, 4, 0.10), (12, 5, 0.10), (13, 6, 0.10)]
    if args.row_start < 0 or args.rows < 1:
        raise ValueError("row-start must be non-negative and rows must be positive")

    device = torch.device(args.device)
    fewshots, until = _task_contract(args.task_config)
    dataset = load_dataset(args.dataset, name=args.dataset_config, split="test")
    stop = min(args.row_start + args.rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_start, stop)]
    if len(documents) != args.rows:
        raise ValueError("requested row range exceeds the dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        .eval()
        .to(device)
    )

    samples = []
    for relative_index, document in enumerate(documents):
        prompt_ids = _prompt_ids(tokenizer, str(document["question"]), fewshots)
        samples.append(
            {
                "index": args.row_start + relative_index,
                "question": str(document["question"]),
                "gold_answer": _gold_answer(str(document["answer"])),
                "prompt_ids": prompt_ids,
            }
        )

    started = time.perf_counter()
    if not args.skip_baseline:
        for sample_index, sample in enumerate(samples):
            sample["baseline"] = _generate(model, tokenizer, sample["prompt_ids"], device, args.max_new_tokens, until)
            if (sample_index + 1) % 4 == 0:
                print(f"baseline rows {sample_index + 1}/{len(samples)}", flush=True)
    for candidate_index, (source, destination, alpha) in enumerate(args.candidate):
        config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
        arm = f"source{source}_destination{destination}_alpha{alpha:g}"
        for sample_index, sample in enumerate(samples):
            with RecirculationController(model, config) as controller:
                sample[arm] = _generate(
                    model,
                    tokenizer,
                    sample["prompt_ids"],
                    device,
                    args.max_new_tokens,
                    until,
                    controller=controller,
                )
            if (sample_index + 1) % 4 == 0:
                correct = sum(
                    numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"])
                    for sample in samples[: sample_index + 1]
                )
                print(
                    f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} rows={sample_index + 1} correct={correct}",
                    flush=True,
                )

    summaries = {}
    baseline_summary = None if args.skip_baseline else _summary(samples, "baseline")
    for source, destination, alpha in args.candidate:
        arm = f"source{source}_destination{destination}_alpha{alpha:g}"
        summaries[arm] = _summary(samples, arm)
        if baseline_summary is not None:
            summaries[arm]["delta_vs_baseline"] = (
                summaries[arm]["numeric_correct"] - baseline_summary["numeric_correct"]
            )
            summaries[arm]["paired_vs_baseline"] = _paired_candidate_summary(samples, "baseline", arm)
    ranking = []
    if baseline_summary is not None:
        ranking = [
            paired_selection_entry(
                source,
                destination,
                alpha,
                summaries[f"source{source}_destination{destination}_alpha{alpha:g}"],
                harm_weight=args.harm_weight,
                max_correct_to_wrong=args.max_correct_to_wrong,
            )
            for source, destination, alpha in args.candidate
        ]
        ranking.sort(
            key=lambda item: (
                not item["valid"],
                -item["selection_score"],
                item["correct_to_wrong"],
                -item["numeric_correct"],
            )
        )
    comparison = None
    if len(args.candidate) == 2:
        reference = args.candidate[0]
        candidate = args.candidate[1]
        reference_arm = f"source{reference[0]}_destination{reference[1]}_alpha{reference[2]:g}"
        candidate_arm = f"source{candidate[0]}_destination{candidate[1]}_alpha{candidate[2]:g}"
        comparison = {
            "reference_arm": reference_arm,
            "candidate_arm": candidate_arm,
            "numeric_correct_delta": (
                summaries[candidate_arm]["numeric_correct"] - summaries[reference_arm]["numeric_correct"]
            ),
            "flexible_correct_delta": (
                summaries[candidate_arm]["flexible_correct"] - summaries[reference_arm]["flexible_correct"]
            ),
            "paired": _paired_candidate_summary(samples, reference_arm, candidate_arm),
        }
    report = {
        "implementation_commit": _implementation_commit(),
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "row_start": args.row_start,
            "rows": args.rows,
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
            "fewshot_count": len(fewshots),
            "candidates": [list(candidate) for candidate in args.candidate],
            "baseline_skipped": args.skip_baseline,
            "screen_results": str(args.screen_results) if args.screen_results is not None else None,
            "top_k": args.top_k,
            "harm_weight": args.harm_weight,
            "max_correct_to_wrong": args.max_correct_to_wrong,
        },
        "seconds": time.perf_counter() - started,
        "summaries": summaries,
        "ranking": ranking,
        "best": ranking[0] if ranking else None,
        "comparison": comparison,
        "samples": [{key: value for key, value in sample.items() if key != "prompt_ids"} for sample in samples],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
