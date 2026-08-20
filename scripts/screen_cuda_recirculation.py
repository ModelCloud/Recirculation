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
    candidate_started = time.perf_counter()
    os.environ.setdefault(_thread_variable, "16")

import torch
from datasets import load_dataset
from logbar import LogBar
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAGraphedConcurrentPrefill, CUDAPrefillRunner
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
                total_nll = 0.0
                answer_tokens = 0
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
                    )
                    total_nll += batch_nll
                    answer_tokens += batch_targets
                return {
                    "source_layer": source,
                    "destination_layer": destination,
                    "alpha": alpha,
                    "answer_nll": total_nll / answer_tokens,
                    "answer_tokens": answer_tokens,
                    "seconds": time.perf_counter() - candidate_started,
                }
            total_nll = 0.0
            answer_tokens = 0
            for context, answer_ids in contexts:
                context_tensor = torch.tensor([context], dtype=torch.long, device="cuda")
                logits, cache, pending, _ = runner.prefill_from_snapshot(context_tensor, snapshot)
                for token_index, token in enumerate(answer_ids):
                    token_logits = logits[0, -1].float()
                    total_nll += float(torch.logsumexp(token_logits, dim=-1) - token_logits[int(token)])
                    answer_tokens += 1
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
            return {
                "source_layer": source,
                "destination_layer": destination,
                "alpha": alpha,
                "answer_nll": total_nll / answer_tokens,
                "answer_tokens": answer_tokens,
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
    answers = [_gold_answer(str(document["answer"])) for document in documents]
    common_length = _common_prefix_length(prompts)
    prefix = torch.tensor([prompts[0][:common_length]], dtype=torch.long, device="cuda")
    phrase = tokenizer("The final answer is ", add_special_tokens=False).input_ids
    contexts = []
    for prompt, answer in zip(prompts, answers):
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
        context = prompt[common_length:] + phrase
        contexts.append((context, answer_ids))

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
            results.append(future.result())
            if len(results) % args.report_every == 0 or len(results) == len(candidates):
                best = min(results, key=lambda item: item["answer_nll"])
                LOG.info(f"candidates={len(results)}/{len(candidates)} best={best}")

    report = {
        "implementation_commit": _implementation_commit(),
        "settings": {
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
            "python": platform.python_version(),
            "python_gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
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
        },
        "seconds": time.perf_counter() - started,
        "results": sorted(results, key=lambda item: item["answer_nll"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    LOG.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
