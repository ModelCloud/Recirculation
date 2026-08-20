#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Rank CUDA recirculation paths and alphas with shared-prefix answer NLL."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "16")

import torch
from datasets import load_dataset
from logbar import LogBar
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAGraphedConcurrentPrefill, CUDAPrefillRunner
from recirculation.screening import gsm8k_solution_target, screen_result_key, summarize_paired_losses
from scripts.eval_gsm8k_platinum import _gold_answer, _prompt_ids, _task_contract

LOG = LogBar.shared()


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


def _implementation_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_report(path, *, status, implementation_commit, settings, started, results, total):
    ordered_results = sorted(results, key=screen_result_key)
    complete = len(ordered_results)
    report = {
        "status": status,
        "complete": complete,
        "active": 1 if status == "running" and complete < total else 0,
        "pending": max(total - complete - (1 if status == "running" and complete < total else 0), 0),
        "implementation_commit": implementation_commit,
        "settings": settings,
        "seconds": time.perf_counter() - started,
        "best": ordered_results[0] if ordered_results else None,
        "results": ordered_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _score_candidate(
    model,
    prefix,
    contexts,
    candidate,
    scheduler,
    use_python_threads,
    row_batch_size,
    pad_token_id,
    graph_prefix,
):
    candidate_started = time.perf_counter()
    source, destination, alpha = candidate
    config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
    candidate_stream = torch.cuda.Stream(device=prefix.device)
    with torch.inference_mode(), torch.cuda.device(prefix.device), torch.cuda.stream(candidate_stream):
        runner = (
            CUDAConcurrentRunner(model, config, use_python_threads=use_python_threads)
            if scheduler == "concurrent"
            else CUDAPrefillRunner(model, config)
        )
        try:
            prefix_graph = None
            if graph_prefix:
                if not isinstance(runner, CUDAConcurrentRunner):
                    raise ValueError("graph-prefix screening requires the concurrent scheduler")
                runner.prefill(prefix[:, :2])
                prefix_graph = CUDAGraphedConcurrentPrefill(runner, prefix, warmups=0)
                _, cache, pending, _ = prefix_graph.prefill(prefix)
            else:
                _, cache, pending, _ = runner.prefill(prefix)
            snapshot = runner.snapshot(cache, pending)
            if prefix_graph is not None:
                torch.cuda.synchronize(prefix.device)
            if row_batch_size > 1:
                batch_runner = CUDAPrefillRunner(model, config, allow_terminal_padding=True)
                row_nll = []
                row_counts = []
                prefix_length = snapshot.pending.token_position + 1
                for batch_start in range(0, len(contexts), row_batch_size):
                    batch = contexts[batch_start : batch_start + row_batch_size]
                    sequences = [context + answer_ids[:-1] for context, answer_ids in batch]
                    maximum_length = max(map(len, sequences))
                    batch_tokens = torch.full(
                        (len(batch), maximum_length),
                        pad_token_id,
                        dtype=torch.long,
                        device="cuda",
                    )
                    attention_mask = torch.zeros(
                        (len(batch), prefix_length + maximum_length),
                        dtype=torch.long,
                        device="cuda",
                    )
                    attention_mask[:, :prefix_length] = 1
                    targets_by_position = {}
                    for row, ((context, answer_ids), sequence) in enumerate(zip(batch, sequences)):
                        batch_tokens[row, : len(sequence)] = torch.tensor(sequence, device="cuda")
                        attention_mask[row, prefix_length : prefix_length + len(sequence)] = 1
                        for target_index, target in enumerate(answer_ids):
                            position = len(context) - 1 + target_index
                            rows, targets = targets_by_position.setdefault(position, ([], []))
                            rows.append(row)
                            targets.append(int(target))
                    batch_nll, batch_targets = batch_runner.score_from_snapshot(
                        batch_tokens,
                        snapshot,
                        targets_by_position,
                        attention_mask=attention_mask,
                        return_per_row=True,
                    )
                    row_nll.extend(batch_nll)
                    row_counts.extend(batch_targets)
                return {
                    "source_layer": source,
                    "destination_layer": destination,
                    "alpha": alpha,
                    "row_nll_totals": row_nll,
                    "row_target_counts": row_counts,
                    "seconds": time.perf_counter() - candidate_started,
                }
            row_nll = []
            row_counts = []
            for context, answer_ids in contexts:
                sample_nll = 0.0
                context_tensor = torch.tensor([context], dtype=torch.long, device="cuda")
                logits, cache, pending, _ = runner.prefill_from_snapshot(context_tensor, snapshot)
                for token_index, token in enumerate(answer_ids):
                    token_logits = logits[0, -1].float()
                    sample_nll += float(torch.logsumexp(token_logits, dim=-1) - token_logits[int(token)])
                    if token_index + 1 < len(answer_ids):
                        answer_token = torch.tensor([[int(token)]], dtype=torch.long, device="cuda")
                        if scheduler == "concurrent":
                            logits, cache, pending = runner.step(answer_token, cache, pending)
                        else:
                            logits, cache, pending, _ = runner.prefill(
                                answer_token,
                                cache=cache,
                                pending=pending,
                            )
                row_nll.append(sample_nll)
                row_counts.append(len(answer_ids))
            return {
                "source_layer": source,
                "destination_layer": destination,
                "alpha": alpha,
                "row_nll_totals": row_nll,
                "row_target_counts": row_counts,
                "seconds": time.perf_counter() - candidate_started,
            }
        finally:
            if isinstance(runner, CUDAConcurrentRunner):
                runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    parser.add_argument("--row-start", type=int, default=272)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--forbid-range", action="append", type=_parse_range, default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--path", action="append", type=_parse_path, default=None)
    parser.add_argument("--max-distance", type=int, default=12)
    parser.add_argument("--scheduler", choices=("concurrent", "sequential"), default="concurrent")
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=("full_solution", "final_answer"),
        default="full_solution",
        help="Score the full rationale and final answer by default; final_answer preserves the historical proxy.",
    )
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--harm-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--graph-prefix",
        action="store_true",
        help="Graph the prefix only when it is at most 256 tokens; longer captures are rejected as unstable.",
    )
    parser.add_argument(
        "--python-threads",
        action="store_true",
        help="Use two Python submitter threads per candidate instead of single-thread asynchronous stream enqueue.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tuning_range = (args.row_start, args.row_start + args.rows)
    forbidden_ranges = args.forbid_range or []
    overlaps = [interval for interval in forbidden_ranges if _overlaps(tuning_range, interval)]
    if overlaps:
        parser.error(f"tuning range {tuning_range} overlaps forbidden evaluation range(s): {overlaps}")
    if min(args.rows, args.max_distance, args.report_every, args.candidate_workers, args.row_batch_size) < 1:
        parser.error("rows, distances, reporting intervals, and worker/batch sizes must be positive")
    if args.scheduler == "sequential" and args.candidate_workers != 1:
        parser.error("the hook-based sequential scheduler requires --candidate-workers 1")
    if args.scheduler == "sequential" and args.graph_prefix:
        parser.error("the sequential scheduler does not support --graph-prefix")
    if args.candidate_workers != 1 and args.graph_prefix:
        parser.error("CUDA Graph prefix capture requires --candidate-workers 1")
    if not 0.0 <= args.tail_quantile < 1.0:
        parser.error("tail-quantile must be in [0, 1)")
    if args.tail_weight < 0.0 or args.harm_tolerance < 0.0:
        parser.error("tail-weight and harm-tolerance must be non-negative")

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    local_files_only = not args.allow_download
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=local_files_only,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    fewshots, _ = _task_contract(args.task_config)
    dataset = load_dataset(args.dataset, name=args.dataset_config, split="test")
    stop = args.row_start + args.rows
    if stop > len(dataset):
        parser.error("requested row range exceeds the dataset")
    documents = [dataset[index] for index in range(args.row_start, stop)]
    prompts = [_prompt_ids(tokenizer, str(document["question"]), fewshots) for document in documents]
    gold_answers = [_gold_answer(str(document["answer"])) for document in documents]
    common_length = min(_common_prefix_length(prompts), min(len(prompt) - 1 for prompt in prompts))
    prefix = torch.tensor([prompts[0][:common_length]], dtype=torch.long, device="cuda")
    contexts = []
    for prompt, document, gold_answer in zip(prompts, documents, gold_answers):
        if args.target_mode == "full_solution":
            target = gsm8k_solution_target(str(document["answer"]), gold_answer)
            context = prompt[common_length:]
        else:
            target = gold_answer
            context = prompt[common_length:] + tokenizer("The final answer is ", add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids
        contexts.append((context, target_ids))

    layer_count = len(model.get_decoder().layers)
    paths = args.path
    if paths is None:
        paths = [
            (source, destination)
            for destination in range(1, layer_count)
            for source in range(destination + 1, min(layer_count, destination + args.max_distance + 1))
        ]
    alphas = args.alpha or [0.1]
    candidates = [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    results = []
    started = time.perf_counter()
    implementation_commit = _implementation_commit()
    native_candidate = (paths[0][0], paths[0][1], 0.0)
    LOG.info(f"Scoring paired native alpha=0 baseline with path {paths[0][0]}->{paths[0][1]}")
    native = _score_candidate(
        model,
        prefix,
        contexts,
        native_candidate,
        args.scheduler,
        args.python_threads,
        args.row_batch_size,
        int(tokenizer.pad_token_id or tokenizer.eos_token_id),
        args.graph_prefix,
    )
    native_nll = native.pop("row_nll_totals")
    native_counts = native.pop("row_target_counts")
    settings = {
        "split_role": "tuning",
        "model": args.model,
        "dataset": args.dataset,
        "row_start": args.row_start,
        "rows": args.rows,
        "row_stop_exclusive": stop,
        "forbidden_ranges": forbidden_ranges,
        "common_prefix_tokens": common_length,
        "alphas": alphas,
        "max_distance": args.max_distance,
        "scheduler": args.scheduler,
        "candidate_workers": args.candidate_workers,
        "python_threads": args.python_threads,
        "row_batch_size": args.row_batch_size,
        "graph_prefix": args.graph_prefix,
        "target_mode": args.target_mode,
        "tail_quantile": args.tail_quantile,
        "tail_weight": args.tail_weight,
        "harm_tolerance": args.harm_tolerance,
        "ranking": "native_delta_nll + tail_weight * worst-tail positive per-row delta",
        "native_baseline": {
            **native,
            "target_nll": sum(native_nll) / sum(native_counts),
            "target_tokens": sum(native_counts),
        },
        "python": platform.python_version(),
        "python_gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
        "thread_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    _write_report(
        args.output,
        status="running",
        implementation_commit=implementation_commit,
        settings=settings,
        started=started,
        results=results,
        total=len(candidates),
    )
    torch.cuda.synchronize()
    with ThreadPoolExecutor(max_workers=args.candidate_workers, thread_name_prefix="recirculation-screen") as executor:
        futures = {
            executor.submit(
                _score_candidate,
                model,
                prefix,
                contexts,
                candidate,
                args.scheduler,
                args.python_threads,
                args.row_batch_size,
                int(tokenizer.pad_token_id or tokenizer.eos_token_id),
                args.graph_prefix,
            ): candidate
            for candidate in candidates
        }
        for future in as_completed(futures):
            result = future.result()
            candidate_nll = result.pop("row_nll_totals")
            candidate_counts = result.pop("row_target_counts")
            result.update(
                summarize_paired_losses(
                    list(range(args.row_start, stop)),
                    native_nll,
                    native_counts,
                    candidate_nll,
                    candidate_counts,
                    tail_quantile=args.tail_quantile,
                    tail_weight=args.tail_weight,
                    harm_tolerance=args.harm_tolerance,
                )
            )
            results.append(result)
            _write_report(
                args.output,
                status="running",
                implementation_commit=implementation_commit,
                settings=settings,
                started=started,
                results=results,
                total=len(candidates),
            )
            if len(results) % args.report_every == 0 or len(results) == len(candidates):
                best = min(results, key=screen_result_key)
                LOG.info(
                    f"candidates={len(results)}/{len(candidates)} "
                    f"best={best['source_layer']}->{best['destination_layer']} alpha={best['alpha']:g} "
                    f"score={best['screen_score']:.6f} target_nll={best['target_nll']:.6f} "
                    f"improved={best['improved_rows']} regressed={best['regressed_rows']}"
                )

    _write_report(
        args.output,
        status="complete",
        implementation_commit=implementation_commit,
        settings=settings,
        started=started,
        results=results,
        total=len(candidates),
    )
    LOG.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
