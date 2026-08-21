#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Shortlist CUDA recirculation paths with shared-prefix full-solution NLL.

This proxy never injects an answer cue into the prompt. Final selection belongs
to ``sweep_gsm8k.py``, which lets each model arm generate autoregressively.
"""

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
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAGraphedConcurrentPrefill, CUDAPrefillRunner
from recirculation.screening import (
    DEFAULT_PATH_ALPHA,
    PROXY_RANKINGS,
    gsm8k_solution_target,
    objective_result_key,
    proxy_shortlist,
    render_screen_report_markdown,
    screen_leaders,
    summarize_paired_losses,
)
from scripts.eval_gsm8k_platinum import _gold_answer, _prompt_ids, _task_contract

LOG = LogBar.shared()
MODEL_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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


def _baseline_contract(args, *, common_prefix_tokens: int, objectives) -> dict:
    return {
        "scoring_schema": "dual_objective_v1",
        "model": args.model,
        "model_type": args.model_type,
        "decoder_layers": args.decoder_layers,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "task_config": str(args.task_config.resolve()),
        "row_start": args.row_start,
        "rows": args.rows,
        "target_mode": args.target_mode,
        "common_prefix_tokens": common_prefix_tokens,
        "objectives": sorted(objectives),
    }


def _load_native_baseline(path: Path, expected_contract: dict):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    actual_contract = artifact.get("contract", {})
    mismatches = {
        key: (actual_contract.get(key), value)
        for key, value in expected_contract.items()
        if actual_contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"shared native baseline does not match this screen: {mismatches}")
    objectives = artifact.get("objectives", {})
    if set(objectives) != set(expected_contract["objectives"]):
        raise ValueError("shared native baseline objective set is incomplete")
    native_nll = {objective: list(values["row_nll_totals"]) for objective, values in objectives.items()}
    native_counts = {objective: list(values["row_target_counts"]) for objective, values in objectives.items()}
    if any(len(native_nll[objective]) != expected_contract["rows"] for objective in objectives):
        raise ValueError("shared native baseline row count is incomplete")
    return artifact, native_nll, native_counts


def _write_native_baseline(path: Path, *, contract: dict, implementation_commit: str, native, nll, counts):
    artifact = {
        "implementation_commit": implementation_commit,
        "contract": contract,
        "seconds": native["seconds"],
        "objectives": {
            objective: {
                "row_nll_totals": nll[objective],
                "row_target_counts": counts[objective],
                "target_nll": sum(nll[objective]) / sum(counts[objective]),
                "target_tokens": sum(counts[objective]),
            }
            for objective in nll
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_report(path, *, status, implementation_commit, settings, started, results, total, elapsed_offset=0.0):
    available = list(results[0]["objectives"]) if results else []
    default_objective = (
        "final_answer"
        if "final_answer" in available
        else "full_solution"
        if "full_solution" in available
        else available[0]
        if available
        else "full_solution"
    )
    ordered_results = sorted(
        results,
        key=lambda result: objective_result_key(result, default_objective, robust=True),
    )
    complete = len(ordered_results)
    leaders = screen_leaders(ordered_results)
    default_leader = leaders.get(f"{default_objective}_robust")
    report = {
        "status": status,
        "complete": complete,
        "active": 1 if status == "running" and complete < total else 0,
        "pending": max(total - complete - (1 if status == "running" and complete < total else 0), 0),
        "total": total,
        "implementation_commit": implementation_commit,
        "settings": settings,
        "seconds": elapsed_offset + time.perf_counter() - started,
        "best": default_leader,
        "best_perplexity": leaders.get(f"{default_objective}_perplexity"),
        "best_robust": default_leader,
        "leaders": leaders,
        "shortlist": proxy_shortlist(ordered_results, min(8, len(ordered_results))) if ordered_results else [],
        "results": ordered_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    json_temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json_temporary.replace(path)
    markdown = path.with_suffix(".md")
    markdown_temporary = markdown.with_name(f".{markdown.name}.{os.getpid()}.tmp")
    markdown_temporary.write_text(render_screen_report_markdown(report), encoding="utf-8")
    markdown_temporary.replace(markdown)


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
    device = next(model.parameters()).device
    candidate_stream = torch.cuda.Stream(device=device)
    with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(candidate_stream):
        runner = (
            CUDAConcurrentRunner(model, config, use_python_threads=use_python_threads)
            if scheduler == "concurrent"
            else CUDAPrefillRunner(model, config)
        )
        try:
            prefix_graph = None
            if graph_prefix and prefix is None:
                raise ValueError("graph-prefix screening requires a non-empty shared prefix")
            if graph_prefix:
                if not isinstance(runner, CUDAConcurrentRunner):
                    raise ValueError("graph-prefix screening requires the concurrent scheduler")
                runner.prefill(prefix[:, :2])
                prefix_graph = CUDAGraphedConcurrentPrefill(runner, prefix, warmups=0)
                _, cache, pending, _ = prefix_graph.prefill(prefix)
            elif prefix is not None:
                _, cache, pending, _ = runner.prefill(prefix)
            else:
                cache = pending = snapshot = None
            if prefix is not None:
                snapshot = runner.snapshot(cache, pending)
            if prefix_graph is not None:
                torch.cuda.synchronize(prefix.device)
            if row_batch_size > 1:
                batch_runner = CUDAPrefillRunner(model, config, allow_terminal_padding=True)
                objective_nll = {objective: [] for objective in contexts[0]}
                objective_counts = {objective: [] for objective in contexts[0]}
                prefix_length = 0 if snapshot is None else snapshot.pending.token_position + 1
                for batch_start in range(0, len(contexts), row_batch_size):
                    batch = contexts[batch_start : batch_start + row_batch_size]
                    for objective in contexts[0]:
                        scoring_rows = [row[objective] for row in batch]
                        sequences = [context + answer_ids[:-1] for context, answer_ids in scoring_rows]
                        maximum_length = max(map(len, sequences))
                        batch_tokens = torch.full(
                            (len(scoring_rows), maximum_length),
                            pad_token_id,
                            dtype=torch.long,
                            device="cuda",
                        )
                        attention_mask = torch.zeros(
                            (len(scoring_rows), prefix_length + maximum_length),
                            dtype=torch.long,
                            device="cuda",
                        )
                        attention_mask[:, :prefix_length] = 1
                        targets_by_position = {}
                        for row, ((context, answer_ids), sequence) in enumerate(zip(scoring_rows, sequences)):
                            batch_tokens[row, : len(sequence)] = torch.tensor(sequence, device="cuda")
                            attention_mask[row, prefix_length : prefix_length + len(sequence)] = 1
                            for target_index, target in enumerate(answer_ids):
                                position = len(context) - 1 + target_index
                                rows, targets = targets_by_position.setdefault(position, ([], []))
                                rows.append(row)
                                targets.append(int(target))
                        score = batch_runner.score if snapshot is None else batch_runner.score_from_snapshot
                        score_args = (batch_tokens, targets_by_position)
                        if snapshot is not None:
                            score_args = (batch_tokens, snapshot, targets_by_position)
                        batch_nll, batch_targets = score(
                            *score_args,
                            attention_mask=attention_mask,
                            return_per_row=True,
                        )
                        objective_nll[objective].extend(batch_nll)
                        objective_counts[objective].extend(batch_targets)
                return {
                    "source_layer": source,
                    "destination_layer": destination,
                    "alpha": alpha,
                    "row_nll_totals": objective_nll,
                    "row_target_counts": objective_counts,
                    "seconds": time.perf_counter() - candidate_started,
                }
            objective_nll = {objective: [] for objective in contexts[0]}
            objective_counts = {objective: [] for objective in contexts[0]}
            for row in contexts:
                for objective, (context, answer_ids) in row.items():
                    sample_nll = 0.0
                    context_tensor = torch.tensor([context], dtype=torch.long, device="cuda")
                    if snapshot is None:
                        logits, cache, pending, _ = runner.prefill(context_tensor)
                    else:
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
                    objective_nll[objective].append(sample_nll)
                    objective_counts[objective].append(len(answer_ids))
            return {
                "source_layer": source,
                "destination_layer": destination,
                "alpha": alpha,
                "row_nll_totals": objective_nll,
                "row_target_counts": objective_counts,
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
    parser.add_argument(
        "--dtype",
        choices=tuple(MODEL_DTYPES),
        default="float16",
        help="Model and residual dtype. Baseline artifacts are dtype-specific.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        choices=("c4", "pg19"),
        default=None,
        help="Use repeated language-modeling corpora instead of GSM8K (C4 train and/or PG-19 train).",
    )
    parser.add_argument("--windows-per-corpus", type=int, default=256)
    parser.add_argument("--window-tokens", type=int, default=1024)
    parser.add_argument(
        "--corpus-artifact",
        type=Path,
        default=None,
        help="Load or atomically create the exact tokenized corpus windows shared by every worker.",
    )
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    parser.add_argument("--row-start", type=int, default=272)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--forbid-range", action="append", type=_parse_range, default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--path", action="append", type=_parse_path, default=None)
    parser.add_argument(
        "--max-distance",
        type=int,
        default=None,
        help="Optional maximum source/destination distance; omitted means all source > destination pairs.",
    )
    parser.add_argument("--scheduler", choices=("concurrent", "sequential"), default="concurrent")
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument(
        "--target-mode",
        choices=("final_answer", "full_solution", "dual"),
        default="full_solution",
        help="Teacher-forced objective: numeric final answer, full solution, or both; final selection uses natural generation.",
    )
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--harm-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--native-baseline",
        type=Path,
        default=None,
        help="Load a validated shared baseline artifact, or write it after computing when absent.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Prepare --native-baseline once and exit before scoring candidates.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=4,
        help="Release unused CUDA allocator blocks every N completed candidates; zero disables it.",
    )
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
    if args.corpus and args.target_mode != "full_solution":
        parser.error("--target-mode final_answer/dual applies to GSM8K screening, not corpus language modeling")
    if args.corpus and args.forbid_range:
        parser.error("--forbid-range applies only to indexed GSM8K screening")
    tuning_range = (args.row_start, args.row_start + args.rows)
    forbidden_ranges = args.forbid_range or []
    overlaps = [interval for interval in forbidden_ranges if _overlaps(tuning_range, interval)]
    if overlaps:
        parser.error(f"tuning range {tuning_range} overlaps forbidden evaluation range(s): {overlaps}")
    if min(
        args.rows,
        args.report_every,
        args.candidate_workers,
        args.row_batch_size,
        args.windows_per_corpus,
        args.window_tokens,
    ) < 1:
        parser.error("rows, reporting intervals, and worker/batch sizes must be positive")
    if args.corpus and args.window_tokens < 3:
        parser.error("--window-tokens must be at least 3")
    if args.max_distance is not None and args.max_distance < 1:
        parser.error("max-distance must be positive when supplied")
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
    if args.empty_cache_every < 0:
        parser.error("empty-cache-every must be non-negative")
    if args.baseline_only and args.native_baseline is None:
        parser.error("--baseline-only requires --native-baseline")

    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    local_files_only = not args.allow_download
    tokenizer = Tokenicer.load(args.model, local_files_only=local_files_only)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=local_files_only,
            dtype=MODEL_DTYPES[args.dtype],
            attn_implementation="eager",
        )
        .eval()
        .to("cuda")
    )
    decoder = model.get_decoder()
    args.model_type = str(getattr(model.config, "model_type", type(model).__name__))
    args.decoder_layers = len(decoder.layers)
    LOG.info(
        f"Loaded {args.model_type} causal LM with {args.decoder_layers} decoder layers; "
        f"tokenizer_bos={tokenizer.bos_token_id}"
    )
    corpus_counts = {}
    if args.corpus:
        specs = {
            "c4": ("allenai/c4", "en"),
            "pg19": ("emozilla/pg19", None),
        }
        corpus_artifact = args.corpus_artifact.expanduser().resolve() if args.corpus_artifact else None
        artifact = None
        if corpus_artifact is not None and corpus_artifact.exists():
            artifact = json.loads(corpus_artifact.read_text(encoding="utf-8"))
            expected = {
                "corpora": args.corpus,
                "windows_per_corpus": args.windows_per_corpus,
                "window_tokens": args.window_tokens,
                "tokenizer": args.model,
            }
            mismatches = {key: (artifact.get(key), value) for key, value in expected.items() if artifact.get(key) != value}
            if mismatches:
                parser.error(f"corpus artifact does not match this screen: {mismatches}")
            LOG.info(f"Loaded {len(artifact['windows'])} shared tokenized corpus windows from {corpus_artifact}")
        if artifact is None:
            windows = []
            for corpus in args.corpus:
                dataset_name, dataset_config = specs[corpus]
                stream = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
                accepted = 0
                examined = 0
                for document in stream:
                    examined += 1
                    token_ids = tokenizer(
                        str(document["text"]),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=args.window_tokens,
                    ).input_ids
                    if len(token_ids) < args.window_tokens:
                        continue
                    windows.append({"corpus": corpus, "token_ids": list(map(int, token_ids))})
                    accepted += 1
                    if accepted == args.windows_per_corpus:
                        break
                if accepted != args.windows_per_corpus:
                    parser.error(f"{corpus} yielded only {accepted} qualifying windows after {examined} documents")
                corpus_counts[corpus] = {"windows": accepted, "documents_examined": examined}
            artifact = {
                "corpora": args.corpus,
                "windows_per_corpus": args.windows_per_corpus,
                "window_tokens": args.window_tokens,
                "tokenizer": args.model,
                "corpus_counts": corpus_counts,
                "windows": windows,
            }
            if corpus_artifact is not None:
                corpus_artifact.parent.mkdir(parents=True, exist_ok=True)
                temporary = corpus_artifact.with_name(f".{corpus_artifact.name}.{os.getpid()}.tmp")
                temporary.write_text(json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8")
                temporary.replace(corpus_artifact)
                LOG.info(f"Wrote shared tokenized corpus windows to {corpus_artifact}")
        corpus_counts = artifact["corpus_counts"]
        contexts = [
            {"language_modeling": ([window["token_ids"][0]], window["token_ids"][1:])}
            for window in artifact["windows"]
        ]
        if tokenizer.bos_token_id is None:
            # Qwen3 intentionally defines no BOS. Starting each window at its
            # first text token preserves the checkpoint's tokenizer contract;
            # substituting EOS or PAD would bias every candidate's recurrent state.
            prefix = None
            common_length = 0
            corpus_prefix_policy = "none"
            LOG.info("Tokenizer defines no BOS; scoring corpus windows from an empty cache")
        else:
            prefix = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device="cuda")
            common_length = 1
            corpus_prefix_policy = "bos"
        args.dataset = "+".join(args.corpus)
        args.dataset_config = "train"
        args.row_start = 0
        args.rows = len(contexts)
        stop = len(contexts)
        forbidden_ranges = []
    else:
        corpus_prefix_policy = None
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
        final_answer_prefix = tokenizer("The final answer is ", add_special_tokens=False).input_ids
        for prompt, document, gold_answer in zip(prompts, documents, gold_answers):
            row = {}
            if args.target_mode in ("full_solution", "dual"):
                full_target = gsm8k_solution_target(str(document["answer"]), gold_answer)
                row["full_solution"] = (
                    prompt[common_length:],
                    tokenizer(full_target, add_special_tokens=False).input_ids,
                )
            if args.target_mode in ("final_answer", "dual"):
                row["final_answer"] = (
                    prompt[common_length:] + final_answer_prefix,
                    tokenizer(gold_answer, add_special_tokens=False).input_ids,
                )
            contexts.append(row)
    baseline_contract = _baseline_contract(
        args,
        common_prefix_tokens=common_length,
        objectives=contexts[0],
    )

    layer_count = args.decoder_layers
    paths = args.path
    if paths is None:
        paths = [
            (source, destination)
            for destination in range(layer_count)
            for source in range(
                destination + 1,
                layer_count if args.max_distance is None else min(layer_count, destination + args.max_distance + 1),
            )
        ]
    alphas = args.alpha or [DEFAULT_PATH_ALPHA]
    candidates = [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    candidate_keys = set(candidates)
    results = []
    started = time.perf_counter()
    implementation_commit = _implementation_commit()
    elapsed_offset = 0.0
    resumed_from_commit = None
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_settings = previous.get("settings", {})
        expected_settings = {
            "scoring_schema": "dual_objective_v1",
            "model": args.model,
            "model_type": args.model_type,
            "decoder_layers": args.decoder_layers,
            "dtype": args.dtype,
            "dataset": args.dataset,
            "row_start": args.row_start,
            "rows": args.rows,
            "target_mode": args.target_mode,
            "tail_quantile": args.tail_quantile,
            "tail_weight": args.tail_weight,
            "harm_tolerance": args.harm_tolerance,
            "shared_native_baseline": (
                str(args.native_baseline.resolve()) if args.native_baseline is not None else None
            ),
        }
        mismatches = {
            key: (previous_settings.get(key), value)
            for key, value in expected_settings.items()
            if previous_settings.get(key) != value
        }
        if mismatches:
            parser.error(f"cannot resume output with different settings: {mismatches}")
        results = [
            item
            for item in previous.get("results", [])
            if (item["source_layer"], item["destination_layer"], item["alpha"]) in candidate_keys
        ]
        if results:
            native_nll = {}
            native_counts = {}
            for objective, summary in results[0]["objectives"].items():
                native_rows = summary["row_metrics"]
                counts = [int(row["target_tokens"]) for row in native_rows]
                native_counts[objective] = counts
                native_nll[objective] = [float(row["native_nll"]) * count for row, count in zip(native_rows, counts)]
            native = dict(previous_settings["native_baseline"])
            elapsed_offset = float(previous.get("seconds", 0.0))
            resumed_from_commit = previous.get("implementation_commit")
            LOG.info(
                f"Resuming {len(results)}/{len(candidates)} completed candidates from {args.output} "
                f"at commit {resumed_from_commit}"
            )
    if not results:
        baseline_path = args.native_baseline.expanduser().resolve() if args.native_baseline is not None else None
        if baseline_path is not None and baseline_path.exists():
            artifact, native_nll, native_counts = _load_native_baseline(baseline_path, baseline_contract)
            native = {
                "source_layer": None,
                "destination_layer": None,
                "alpha": 0.0,
                "seconds": float(artifact["seconds"]),
                "shared_artifact": str(baseline_path),
                "artifact_commit": artifact.get("implementation_commit"),
            }
            LOG.info(f"Loaded shared dual-objective native baseline from {baseline_path}")
        else:
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
                int(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                args.graph_prefix,
            )
            native_nll = native.pop("row_nll_totals")
            native_counts = native.pop("row_target_counts")
            if baseline_path is not None:
                _write_native_baseline(
                    baseline_path,
                    contract=baseline_contract,
                    implementation_commit=implementation_commit,
                    native=native,
                    nll=native_nll,
                    counts=native_counts,
                )
                native["shared_artifact"] = str(baseline_path)
                LOG.info(f"Wrote shared dual-objective native baseline to {baseline_path}")
            if args.empty_cache_every:
                torch.cuda.empty_cache()
    if args.baseline_only:
        LOG.info("Shared native baseline is ready; baseline-only run complete")
        return 0
    settings = {
        "scoring_schema": "dual_objective_v1",
        "split_role": "tuning",
        "model": args.model,
        "model_type": args.model_type,
        "decoder_layers": args.decoder_layers,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "corpora": args.corpus,
        "corpus_counts": corpus_counts,
        "window_tokens": args.window_tokens if args.corpus else None,
        "corpus_artifact": str(args.corpus_artifact.resolve()) if args.corpus_artifact else None,
        "row_start": args.row_start,
        "rows": args.rows,
        "row_stop_exclusive": stop,
        "forbidden_ranges": forbidden_ranges,
        "common_prefix_tokens": common_length,
        "corpus_prefix_policy": corpus_prefix_policy,
        "alphas": alphas,
        "max_distance": args.max_distance,
        "scheduler": args.scheduler,
        "candidate_workers": args.candidate_workers,
        "python_threads": args.python_threads,
        "row_batch_size": args.row_batch_size,
        "graph_prefix": args.graph_prefix,
        "target_mode": args.target_mode,
        "objective_batching": "separate shape-stable batches from one candidate prefix snapshot",
        "tail_quantile": args.tail_quantile,
        "tail_weight": args.tail_weight,
        "harm_tolerance": args.harm_tolerance,
        "shared_native_baseline": str(args.native_baseline.resolve()) if args.native_baseline is not None else None,
        "resume": args.resume,
        "resumed_from_commit": resumed_from_commit,
        "empty_cache_every": args.empty_cache_every,
        "ranking": {
            "plain": "minimum target perplexity",
            "robust": "native_delta_nll + tail_weight * worst-tail positive per-row delta",
            "proxy_rankings": [name for name, _, _ in PROXY_RANKINGS],
        },
        "native_baseline": {
            **native,
            "objectives": {
                objective: {
                    "target_nll": sum(native_nll[objective]) / sum(native_counts[objective]),
                    "target_tokens": sum(native_counts[objective]),
                }
                for objective in native_nll
            },
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
        elapsed_offset=elapsed_offset,
    )
    completed_keys = {(item["source_layer"], item["destination_layer"], item["alpha"]) for item in results}
    remaining_candidates = [candidate for candidate in candidates if candidate not in completed_keys]
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
                int(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                args.graph_prefix,
            ): candidate
            for candidate in remaining_candidates
        }
        for future in as_completed(futures):
            result = future.result()
            candidate_nll = result.pop("row_nll_totals")
            candidate_counts = result.pop("row_target_counts")
            result["objectives"] = {
                objective: summarize_paired_losses(
                    list(range(args.row_start, stop)),
                    native_nll[objective],
                    native_counts[objective],
                    candidate_nll[objective],
                    candidate_counts[objective],
                    tail_quantile=args.tail_quantile,
                    tail_weight=args.tail_weight,
                    harm_tolerance=args.harm_tolerance,
                )
                for objective in candidate_nll
            }
            results.append(result)
            _write_report(
                args.output,
                status="running",
                implementation_commit=implementation_commit,
                settings=settings,
                started=started,
                results=results,
                total=len(candidates),
                elapsed_offset=elapsed_offset,
            )
            if args.empty_cache_every and len(results) % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
            if len(results) % args.report_every == 0 or len(results) == len(candidates):
                leaders = {
                    name: min(
                        results,
                        key=lambda item, objective=objective, robust=robust: objective_result_key(
                            item, objective, robust=robust
                        ),
                    )
                    for name, objective, robust in PROXY_RANKINGS
                    if objective in results[0]["objectives"]
                }
                summary = "; ".join(
                    f"{name}={item['source_layer']}->{item['destination_layer']}@{item['alpha']:g}"
                    for name, item in leaders.items()
                )
                LOG.info(f"candidates={len(results)}/{len(candidates)} {summary}")
                for name, objective, robust in PROXY_RANKINGS:
                    if name not in leaders:
                        continue
                    item = leaders[name]
                    metrics = item["objectives"][objective]
                    LOG.info(
                        f"{name}: target_ppl={metrics['target_perplexity']:.6f} "
                        f"score={metrics['screen_score']:.6f} regressed={metrics['regressed_rows']} "
                        f"penalized={robust}"
                    )

    _write_report(
        args.output,
        status="complete",
        implementation_commit=implementation_commit,
        settings=settings,
        started=started,
        results=results,
        total=len(candidates),
        elapsed_offset=elapsed_offset,
    )
    LOG.info(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
