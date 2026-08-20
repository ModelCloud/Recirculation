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
from recirculation.screening import paired_selection_entry, proxy_shortlist
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


def _arm_name(source: int, destination: int, alpha: float) -> str:
    return f"source{source}_destination{destination}_alpha{alpha:g}"


def _status_table(samples, candidates, total_rows: int, phase: str) -> str:
    """Render a compact live table, including paired flips for completed rows."""
    lines = [
        "State      Arm               Config           Rows    Correct    Accuracy    W→C    C→W    Net",
        "────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────",
    ]
    baseline_done = [sample for sample in samples if "baseline" in sample]
    baseline_complete = len(baseline_done) == total_rows
    baseline_correct = sum(numbers_equal(s["baseline"]["numeric_answer"], s["gold_answer"]) for s in baseline_done)
    baseline_state = "Complete" if baseline_complete else ("Active" if phase == "baseline" else "Pending")
    baseline_acc = f"{100 * baseline_correct / len(baseline_done):.2f}%" if baseline_done else "—"
    lines.append(f"{baseline_state:<9} Dense baseline    none             {len(baseline_done):>3}/{total_rows:<4}  {baseline_correct:>8}  {baseline_acc:>9}    —      —      —")
    lines.append("────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────")
    for index, (source, destination, alpha) in enumerate(candidates):
        arm = _arm_name(source, destination, alpha)
        done = [sample for sample in samples if arm in sample]
        complete = len(done) == total_rows
        state = "Complete" if complete else ("Active" if done else ("Active" if phase == f"candidate-{index}" else "Pending"))
        correct = sum(numbers_equal(s[arm]["numeric_answer"], s["gold_answer"]) for s in done)
        accuracy = f"{100 * correct / len(done):.2f}%" if done else "—"
        paired = [s for s in done if "baseline" in s]
        w2c = c2w = 0
        for sample in paired:
            base_ok = numbers_equal(sample["baseline"]["numeric_answer"], sample["gold_answer"])
            cand_ok = numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"])
            w2c += not base_ok and cand_ok
            c2w += base_ok and not cand_ok
        flip = f"{w2c:>5}  {c2w:>5}  {w2c - c2w:>+5}" if paired else "    —      —      —"
        config = f"{source}→{destination}, α={alpha:.2f}"
        lines.append(f"{state:<9} Recirculation     {config:<14} {len(done):>3}/{total_rows:<4}  {correct:>8}  {accuracy:>9}  {flip}")
        if index != len(candidates) - 1:
            lines.append("────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────")
    return "\n".join(lines)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    parser.add_argument("--status-every", type=float, default=60.0, help="Seconds between live status table updates.")
    args = parser.parse_args()
    if args.screen_results is not None and args.candidate is not None:
        parser.error("use either --screen-results or --candidate, not both")
    if args.top_k < 1 or args.harm_weight < 1.0:
        parser.error("top-k must be positive and harm-weight must be at least 1")
    if args.max_correct_to_wrong is not None and args.max_correct_to_wrong < 0:
        parser.error("max-correct-to-wrong must be non-negative")
    if args.status_every <= 0:
        parser.error("status-every must be positive")
    selected_screen_items = {}
    if args.screen_results is not None:
        screen = json.loads(args.screen_results.read_text(encoding="utf-8"))
        screen_results = screen.get("results", [])
        candidates = (
            proxy_shortlist(screen_results, args.top_k)
            if screen_results and "objectives" in screen_results[0]
            else screen_results[: args.top_k]
        )
        if not candidates:
            parser.error("screen-results does not contain candidates")
        selected_screen_items = {
            (int(item["source_layer"]), int(item["destination_layer"]), float(item["alpha"])): item
            for item in candidates
        }
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
    status_path = args.output.expanduser().resolve().with_suffix(".status.json")
    last_status = 0.0

    def maybe_status(phase: str, force: bool = False) -> None:
        nonlocal last_status
        now = time.perf_counter()
        if not force and now - last_status < args.status_every:
            return
        last_status = now
        table = _status_table(samples, args.candidate, len(samples), phase)
        payload = {
            "implementation_commit": _implementation_commit(),
            "phase": phase,
            "elapsed_seconds": now - started,
            "total_rows": len(samples),
            "candidates": [list(candidate) for candidate in args.candidate],
            "table": table,
        }
        _write_status(status_path, payload)
        print(table, flush=True)

    maybe_status("baseline", force=True)
    if not args.skip_baseline:
        for sample_index, sample in enumerate(samples):
            sample["baseline"] = _generate(model, tokenizer, sample["prompt_ids"], device, args.max_new_tokens, until)
            maybe_status("baseline")
            if (sample_index + 1) % 4 == 0:
                print(f"baseline rows {sample_index + 1}/{len(samples)}", flush=True)
        maybe_status("candidates", force=True)
    for candidate_index, (source, destination, alpha) in enumerate(args.candidate):
        config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
        arm = _arm_name(source, destination, alpha)
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
            maybe_status(f"candidate-{candidate_index}")
            if (sample_index + 1) % 4 == 0:
                correct = sum(
                    numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"])
                    for sample in samples[: sample_index + 1]
                )
                print(
                    f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} rows={sample_index + 1} correct={correct}",
                    flush=True,
                )

        maybe_status(f"candidate-{candidate_index + 1}", force=True)

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
        for item in ranking:
            screen_item = selected_screen_items.get((item["source_layer"], item["destination_layer"], item["alpha"]))
            if screen_item is not None and "objectives" in screen_item:
                item["proxy_objectives"] = screen_item["objectives"]
    best_flip_penalized = ranking[0] if ranking else None
    best_accuracy = (
        min(
            ranking,
            key=lambda item: (
                not item["valid"],
                -item["numeric_correct"],
                item["correct_to_wrong"],
                -item["wrong_to_correct"],
            ),
        )
        if ranking
        else None
    )
    candidates_with_proxy = [item for item in ranking if "proxy_objectives" in item]
    best_perplexity = (
        min(
            candidates_with_proxy,
            key=lambda item: item["proxy_objectives"]["final_answer"]["target_perplexity"],
        )
        if candidates_with_proxy
        else None
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
        "best": best_flip_penalized,
        "best_flip_penalized": best_flip_penalized,
        "best_accuracy": best_accuracy,
        "best_perplexity": best_perplexity,
        "comparison": comparison,
        "status_checkpoint": str(status_path),
        "samples": [{key: value for key, value in sample.items() if key != "prompt_ids"} for sample in samples],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    maybe_status("complete", force=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
