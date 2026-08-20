#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Screen Llama recirculation source/destination pairs on GSM8K-Platinum."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import (
    RecirculationConfig,
    RecirculationController,
)
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
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        help="Repeat SOURCE:DESTINATION:ALPHA; defaults to four middle-stack pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate is None:
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
    ).eval().to(device)

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
    for sample_index, sample in enumerate(samples):
        sample["baseline"] = _generate(
            model, tokenizer, sample["prompt_ids"], device, args.max_new_tokens, until
        )
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
                    sample[arm]["flexible_answer"] == sample["gold_answer"]
                    for sample in samples[: sample_index + 1]
                )
                print(f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} rows={sample_index + 1} correct={correct}", flush=True)

    summaries = {}
    for source, destination, alpha in args.candidate:
        arm = f"source{source}_destination{destination}_alpha{alpha:g}"
        summaries[arm] = _summary(samples, arm)
        summaries[arm]["delta_vs_baseline"] = summaries[arm]["flexible_correct"] - _summary(samples, "baseline")["flexible_correct"]
    report = {
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "row_start": args.row_start,
            "rows": args.rows,
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
            "fewshot_count": len(fewshots),
            "candidates": [list(candidate) for candidate in args.candidate],
        },
        "seconds": time.perf_counter() - started,
        "summaries": summaries,
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
