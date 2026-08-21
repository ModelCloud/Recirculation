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
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
from recirculation.cuda_backend import (
    CUDABatchedPathRunner,
    CUDAConcurrentRunner,
    CUDAGraphedConcurrentPrefill,
    CUDAPrefillRunner,
)
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


def _ordered_candidates(paths, alphas, *, scan_order: str, scan_seed: int):
    """Build a reproducible candidate schedule without low-layer-first bias."""

    candidates = [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate paths and alphas must be unique")
    if scan_order == "random":
        random.Random(scan_seed).shuffle(candidates)
    elif scan_order != "sequential":
        raise ValueError(f"unsupported scan order: {scan_order}")
    return candidates


def _candidate_schedule(candidates):
    return [
        {
            "scan_index": scan_index,
            "source_layer": source,
            "destination_layer": destination,
            "alpha": alpha,
        }
        for scan_index, (source, destination, alpha) in enumerate(candidates)
    ]


def _candidate_work(contexts, row_batch_size):
    """Describe host-visible work without touching or synchronizing CUDA."""

    batches = 0
    scoring_rows = 0
    padded_token_positions = 0
    for batch_start in range(0, len(contexts), row_batch_size):
        batch = contexts[batch_start : batch_start + row_batch_size]
        for objective in contexts[0]:
            sequence_lengths = [
                len(context) + max(len(answer_ids) - 1, 0)
                for context, answer_ids in (row[objective] for row in batch)
            ]
            batches += 1
            scoring_rows += len(batch)
            padded_token_positions += len(batch) * max(sequence_lengths)
    return {
        "batches": batches,
        "scoring_rows": scoring_rows,
        "padded_token_positions": padded_token_positions,
    }


def _format_duration(seconds):
    if seconds is None:
        return "—"
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


class _PathTelemetry:
    """Publish coarse CPU-side progress without adding CUDA work or synchronization."""

    def __init__(self, output, schedule, work, completed, interval):
        self.output = output.with_suffix(".status.json")
        self.schedule = schedule
        self.work = work
        self.interval = float(interval)
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = {}
        self._completed = {
            int(item["scan_index"]): {
                "scan_index": int(item["scan_index"]),
                "source_layer": int(item["source_layer"]),
                "destination_layer": int(item["destination_layer"]),
                "alpha": float(item["alpha"]),
                "seconds": float(item.get("seconds", 0.0)),
            }
            for item in completed
        }
        self._thread = None

    def start(self):
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._run, name="screen-telemetry", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=min(self.interval, 2.0))
        self.emit()

    def candidate_started(self, candidate, scan_index):
        if self.interval <= 0:
            return
        source, destination, alpha = candidate
        now = time.perf_counter()
        with self._lock:
            self._active[scan_index] = {
                "scan_index": scan_index,
                "source_layer": source,
                "destination_layer": destination,
                "alpha": alpha,
                "started": now,
                "last_progress": now,
                "batches_completed": 0,
                "scoring_rows_completed": 0,
                "padded_token_positions_completed": 0,
            }

    def batch_completed(self, candidate, scan_index, *, rows, padded_token_positions):
        if self.interval <= 0:
            return
        with self._lock:
            active = self._active.get(scan_index)
            if active is None:
                return
            active["batches_completed"] += 1
            active["scoring_rows_completed"] += rows
            active["padded_token_positions_completed"] += padded_token_positions
            active["last_progress"] = time.perf_counter()

    def candidate_completed(self, result):
        if self.interval <= 0:
            return
        scan_index = int(result["scan_index"])
        with self._lock:
            self._active.pop(scan_index, None)
            self._completed[scan_index] = {
                "scan_index": scan_index,
                "source_layer": int(result["source_layer"]),
                "destination_layer": int(result["destination_layer"]),
                "alpha": float(result["alpha"]),
                "seconds": float(result["seconds"]),
            }

    def snapshot(self):
        now = time.perf_counter()
        with self._lock:
            active_rows = [dict(value) for value in self._active.values()]
            completed_rows = [dict(value) for value in self._completed.values()]
        active = []
        for item in sorted(active_rows, key=lambda value: value["scan_index"]):
            started = item.pop("started")
            last_progress = item.pop("last_progress")
            elapsed = now - started
            done = item["padded_token_positions_completed"]
            total = self.work["padded_token_positions"]
            fraction = min(done / total, 1.0) if total else 0.0
            measured_seconds = last_progress - started
            estimated_total = measured_seconds / fraction if fraction else None
            eta = max(estimated_total - elapsed, 0.0) if estimated_total is not None else None
            item.update(
                {
                    "state": "active",
                    "batches_total": self.work["batches"],
                    "scoring_rows_total": self.work["scoring_rows"],
                    "padded_token_positions_total": total,
                    "progress": fraction,
                    "elapsed_seconds": elapsed,
                    "progress_updated_seconds_ago": now - last_progress,
                    "estimated_total_seconds": estimated_total,
                    "path_eta_seconds": eta,
                }
            )
            active.append(item)
        total_candidates = len(self.schedule)
        complete = len(completed_rows)
        pending = max(total_candidates - complete - len(active), 0)
        durations = [item["seconds"] for item in completed_rows if item["seconds"] > 0]
        estimated_path_seconds = sum(durations) / len(durations) if durations else None
        if estimated_path_seconds is None and active:
            estimates = [
                item["estimated_total_seconds"]
                for item in active
                if item["estimated_total_seconds"] is not None
            ]
            estimated_path_seconds = sum(estimates) / len(estimates) if estimates else None
        sweep_eta = None
        if estimated_path_seconds is not None:
            sweep_eta = pending * estimated_path_seconds + sum(
                item["path_eta_seconds"] or estimated_path_seconds for item in active
            )
        return {
            "status": "complete" if complete == total_candidates else "running",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "telemetry": {
                "interval_seconds": self.interval,
                "method": "host-side counters updated only after existing scoring batches",
                "cuda_synchronizations_added": 0,
                "cuda_queries_added": 0,
            },
            "aggregate": {
                "complete": complete,
                "active": len(active),
                "pending": pending,
                "total": total_candidates,
                "elapsed_seconds": now - self.started,
                "mean_completed_path_seconds": sum(durations) / len(durations) if durations else None,
                "estimated_path_seconds": estimated_path_seconds,
                "sweep_eta_seconds": sweep_eta,
            },
            "active_paths": active,
            "completed_paths": sorted(completed_rows, key=lambda value: value["scan_index"]),
        }

    def emit(self):
        report = self.snapshot()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(f".{self.output.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.output)
        aggregate = report["aggregate"]
        if report["active_paths"]:
            paths = "; ".join(
                f"scan={item['scan_index']} {item['source_layer']}->{item['destination_layer']}@{item['alpha']:g} "
                f"batches={item['batches_completed']}/{item['batches_total']} "
                f"rows={item['scoring_rows_completed']}/{item['scoring_rows_total']} "
                f"progress={item['progress']:.1%} elapsed={_format_duration(item['elapsed_seconds'])} "
                f"path_eta={_format_duration(item['path_eta_seconds'])}"
                for item in report["active_paths"]
            )
        else:
            paths = "no active path"
        LOG.info(
            f"telemetry candidates={aggregate['complete']}/{aggregate['total']} active={aggregate['active']} "
            f"pending={aggregate['pending']} sweep_eta={_format_duration(aggregate['sweep_eta_seconds'])}; {paths}"
        )

    def _run(self):
        self.emit()
        while not self._stop.wait(self.interval):
            self.emit()


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
    projection_chunk_tokens=1,
    mask_free_unpadded=False,
    dual_gemm=False,
    progress_callback=None,
):
    candidate_started = time.perf_counter()
    source, destination, alpha = candidate
    if progress_callback is not None:
        progress_callback("candidate_started")
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
                batch_runner = None
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
                        unpadded = bool(attention_mask.all())
                        use_concurrent_batch = scheduler == "concurrent" and snapshot is None and unpadded
                        if batch_runner is None or use_concurrent_batch != isinstance(batch_runner, CUDAConcurrentRunner):
                            if isinstance(batch_runner, CUDAConcurrentRunner):
                                batch_runner.close()
                            if use_concurrent_batch:
                                batch_runner = CUDAConcurrentRunner(
                                    model,
                                    config,
                                    use_python_threads=use_python_threads,
                                    dual_gemm=dual_gemm,
                                )
                            else:
                                batch_runner = CUDAPrefillRunner(
                                    model,
                                    config,
                                    allow_terminal_padding=True,
                                    projection_chunk_tokens=projection_chunk_tokens,
                                    mask_free_unpadded=mask_free_unpadded,
                                )
                        score = batch_runner.score if snapshot is None else batch_runner.score_from_snapshot
                        score_args = (batch_tokens, targets_by_position)
                        if snapshot is not None:
                            score_args = (batch_tokens, snapshot, targets_by_position)
                        batch_nll, batch_targets = score(
                            *score_args,
                            attention_mask=attention_mask,
                            return_per_row=True,
                            **(
                                {"projection_chunk_tokens": projection_chunk_tokens}
                                if isinstance(batch_runner, CUDAConcurrentRunner)
                                else {}
                            ),
                        )
                        objective_nll[objective].extend(batch_nll)
                        objective_counts[objective].extend(batch_targets)
                        if progress_callback is not None:
                            progress_callback(
                                "batch_completed",
                                rows=len(scoring_rows),
                                padded_token_positions=len(scoring_rows) * maximum_length,
                            )
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
            if "batch_runner" in locals() and isinstance(batch_runner, CUDAConcurrentRunner):
                batch_runner.close()


def _candidate_batches(candidates, width):
    """Group paths by shared replay boundary while retaining randomized group order."""

    pending = list(candidates)
    while pending:
        first = pending.pop(0)
        group = [first]
        matching = []
        for index, candidate in enumerate(pending):
            if candidate[1:] == first[1:]:
                matching.append(index)
                if len(group) + len(matching) == width:
                    break
        for index in reversed(matching):
            group.append(pending.pop(index))
        yield group


def _effective_candidate_batch_size(requested, row_batch_size, sequence_length, token_budget):
    """Bound candidate width by the live candidate-row-token cache footprint."""

    if min(requested, row_batch_size, sequence_length, token_budget) < 1:
        raise ValueError("candidate batch sizing inputs must be positive")
    return max(1, min(requested, token_budget // (row_batch_size * sequence_length)))


def _effective_row_batch_size(requested, candidate_batch_size, sequence_length, token_budget):
    """Reduce scoring rows when one candidate alone would exceed the memory budget."""

    if min(requested, candidate_batch_size, sequence_length, token_budget) < 1:
        raise ValueError("row batch sizing inputs must be positive")
    return max(1, min(requested, token_budget // (candidate_batch_size * sequence_length)))


def _score_candidate_batch(
    model,
    contexts,
    candidates,
    row_batch_size,
    pad_token_id,
    projection_chunk_tokens,
    progress_callbacks=None,
):
    """Score same-destination paths together in the model batch dimension."""

    started = time.perf_counter()
    configs = [
        RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
        for source, destination, alpha in candidates
    ]
    callbacks = progress_callbacks or [None] * len(candidates)
    for callback in callbacks:
        if callback is not None:
            callback("candidate_started")
    runner = CUDABatchedPathRunner(model, configs)
    candidate_nll = [
        {objective: [] for objective in contexts[0]} for _candidate in candidates
    ]
    candidate_counts = [
        {objective: [] for objective in contexts[0]} for _candidate in candidates
    ]
    with torch.inference_mode():
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
                attention_mask = torch.zeros_like(batch_tokens)
                targets_by_position = {}
                for row, ((context, answer_ids), sequence) in enumerate(zip(scoring_rows, sequences)):
                    batch_tokens[row, : len(sequence)] = torch.tensor(sequence, device="cuda")
                    attention_mask[row, : len(sequence)] = 1
                    for target_index, target in enumerate(answer_ids):
                        position = len(context) - 1 + target_index
                        rows, targets = targets_by_position.setdefault(position, ([], []))
                        rows.append(row)
                        targets.append(int(target))
                if not bool(attention_mask.all()):
                    raise ValueError(
                        "candidate batching requires equal-length unpadded rows; use width one for padded data"
                    )
                nll_values, count_values = runner.score(
                    batch_tokens,
                    targets_by_position,
                    attention_mask=attention_mask,
                    projection_chunk_tokens=projection_chunk_tokens,
                )
                for candidate_index in range(len(candidates)):
                    candidate_nll[candidate_index][objective].extend(nll_values[candidate_index])
                    candidate_counts[candidate_index][objective].extend(count_values[candidate_index])
                    callback = callbacks[candidate_index]
                    if callback is not None:
                        callback(
                            "batch_completed",
                            rows=len(scoring_rows),
                            padded_token_positions=len(scoring_rows) * maximum_length,
                        )
    group_seconds = time.perf_counter() - started
    return [
        {
            "source_layer": source,
            "destination_layer": destination,
            "alpha": alpha,
            "row_nll_totals": candidate_nll[index],
            "row_target_counts": candidate_counts[index],
            "seconds": group_seconds / len(candidates),
            "candidate_batch_seconds": group_seconds,
            "candidate_batch_width": len(candidates),
        }
        for index, (source, destination, alpha) in enumerate(candidates)
    ]


def _score_native_dense(model, contexts, row_batch_size, pad_token_id, projection_chunk_tokens):
    """Score alpha=0 with ordinary parallel causal prefill and chunked LM projection."""

    started = time.perf_counter()
    device = next(model.parameters()).device
    decoder = model.get_decoder()
    output_embeddings = model.get_output_embeddings()
    objective_nll = {objective: [] for objective in contexts[0]}
    objective_counts = {objective: [] for objective in contexts[0]}
    with torch.inference_mode():
        for batch_start in range(0, len(contexts), row_batch_size):
            batch = contexts[batch_start : batch_start + row_batch_size]
            for objective in contexts[0]:
                scoring_rows = [row[objective] for row in batch]
                sequences = [context + answer_ids[:-1] for context, answer_ids in scoring_rows]
                maximum_length = max(map(len, sequences))
                tokens = torch.full(
                    (len(batch), maximum_length),
                    pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(tokens)
                targets_by_position = {}
                for row, ((context, answer_ids), sequence) in enumerate(zip(scoring_rows, sequences)):
                    tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                    attention_mask[row, : len(sequence)] = 1
                    for target_index, target in enumerate(answer_ids):
                        position = len(context) - 1 + target_index
                        rows, targets = targets_by_position.setdefault(position, ([], []))
                        rows.append(row)
                        targets.append(int(target))
                hidden = decoder(
                    input_ids=tokens,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
                row_nll = torch.zeros(len(batch), dtype=torch.float32, device=device)
                row_counts = [0] * len(batch)
                positions = sorted(targets_by_position)
                for chunk_start in range(0, len(positions), projection_chunk_tokens):
                    chunk_positions = positions[chunk_start : chunk_start + projection_chunk_tokens]
                    selected_hidden = []
                    selected_rows = []
                    selected_targets = []
                    for position in chunk_positions:
                        row_values, target_values = targets_by_position[position]
                        rows = torch.tensor(row_values, dtype=torch.long, device=device)
                        selected_hidden.append(hidden[rows, position])
                        selected_rows.append(rows)
                        selected_targets.append(torch.tensor(target_values, dtype=torch.long, device=device))
                        for row in row_values:
                            row_counts[row] += 1
                    rows = torch.cat(selected_rows)
                    targets = torch.cat(selected_targets)
                    logits = output_embeddings(torch.cat(selected_hidden)).float()
                    losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]
                    row_nll.index_add_(0, rows, losses)
                objective_nll[objective].extend(row_nll.tolist())
                objective_counts[objective].extend(row_counts)
    return {
        "source_layer": None,
        "destination_layer": None,
        "alpha": 0.0,
        "row_nll_totals": objective_nll,
        "row_target_counts": objective_counts,
        "seconds": time.perf_counter() - started,
        "implementation": "dense_parallel_causal_prefill",
    }


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
        "--scan-order",
        choices=("random", "sequential"),
        default="random",
        help="Candidate execution order. Random is deterministic under --scan-seed and avoids low-layer-first bias.",
    )
    parser.add_argument("--scan-seed", type=int, default=20260821)
    parser.add_argument(
        "--max-distance",
        type=int,
        default=None,
        help="Optional maximum source/destination distance; omitted means all source > destination pairs.",
    )
    parser.add_argument("--scheduler", choices=("concurrent", "sequential"), default="concurrent")
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help="Batch same-destination Qwen3 paths in one process; widths above one require corpus scoring and dual GEMM.",
    )
    parser.add_argument(
        "--candidate-token-budget",
        type=int,
        default=65536,
        help="Cap candidate_width * row_batch * sequence_length to control KV/activation memory.",
    )
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument(
        "--projection-chunk-tokens",
        type=int,
        default=16,
        help="Fuse LM-head projection across scored token steps; set one for legacy dispatch.",
    )
    parser.add_argument(
        "--mask-free-unpadded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip redundant all-ones decoder masks while retaining causal attention.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa"),
        default="eager",
        help="Transformers attention implementation used during screening.",
    )
    parser.add_argument(
        "--dual-gemm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pair Qwen3 replay/current upper-stack projections into shared GEMMs.",
    )
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
    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=60.0,
        help="Emit lightweight per-path progress every N seconds; zero disables telemetry.",
    )
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
        args.candidate_batch_size,
        args.candidate_token_budget,
        args.row_batch_size,
        args.projection_chunk_tokens,
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
    if args.candidate_batch_size > 1:
        if bool(getattr(sys, "_is_gil_enabled", lambda: True)()):
            parser.error("candidate batching requires CPython 3.14t with -X gil=0")
        if not args.corpus:
            parser.error("candidate batching currently requires fixed-window corpus scoring")
        if args.scheduler != "concurrent" or not args.dual_gemm:
            parser.error("candidate batching requires --scheduler concurrent and --dual-gemm")
        if args.candidate_workers != 1 or args.graph_prefix:
            parser.error("candidate batching requires one worker and does not support prefix graphs")
    if not 0.0 <= args.tail_quantile < 1.0:
        parser.error("tail-quantile must be in [0, 1)")
    if args.tail_weight < 0.0 or args.harm_tolerance < 0.0:
        parser.error("tail-weight and harm-tolerance must be non-negative")
    if args.empty_cache_every < 0:
        parser.error("empty-cache-every must be non-negative")
    if args.telemetry_interval < 0:
        parser.error("telemetry-interval must be non-negative")
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
            attn_implementation=args.attention_backend,
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
    requested_candidate_batch_size = args.candidate_batch_size
    requested_row_batch_size = args.row_batch_size
    if args.candidate_batch_size > 1:
        maximum_sequence_length = max(
            len(context) + len(answer_ids) - 1
            for row in contexts
            for context, answer_ids in row.values()
        )
        args.candidate_batch_size = _effective_candidate_batch_size(
            args.candidate_batch_size,
            min(args.row_batch_size, len(contexts)),
            maximum_sequence_length,
            args.candidate_token_budget,
        )
        if args.candidate_batch_size != requested_candidate_batch_size:
            LOG.info(
                f"Reducing candidate batch width {requested_candidate_batch_size}->{args.candidate_batch_size} "
                f"to honor the {args.candidate_token_budget} candidate-row-token memory budget"
            )
        args.row_batch_size = _effective_row_batch_size(
            args.row_batch_size,
            args.candidate_batch_size,
            maximum_sequence_length,
            args.candidate_token_budget,
        )
        if args.row_batch_size != requested_row_batch_size:
            LOG.info(
                f"Reducing row batch size {requested_row_batch_size}->{args.row_batch_size} "
                f"to honor the {args.candidate_token_budget} candidate-row-token memory budget"
            )
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
    candidates = _ordered_candidates(paths, alphas, scan_order=args.scan_order, scan_seed=args.scan_seed)
    candidate_keys = set(candidates)
    candidate_schedule = _candidate_schedule(candidates)
    scan_indices = {candidate: index for index, candidate in enumerate(candidates)}
    results = []
    started = time.perf_counter()
    implementation_commit = _implementation_commit()
    elapsed_offset = 0.0
    resumed_from_commit = None
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_settings = previous.get("settings", {})
        previous_scan_order = previous_settings.get("scan_order", "sequential")
        previous_scan_seed = previous_settings.get("scan_seed", args.scan_seed)
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
        if previous_scan_order != args.scan_order:
            mismatches["scan_order"] = (previous_scan_order, args.scan_order)
        if previous_scan_seed != args.scan_seed:
            mismatches["scan_seed"] = (previous_scan_seed, args.scan_seed)
        if mismatches:
            parser.error(f"cannot resume output with different settings: {mismatches}")
        stored_schedule = previous_settings.get("candidate_schedule")
        if stored_schedule is not None:
            stored_schedule = sorted(stored_schedule, key=lambda item: item["scan_index"])
            stored_candidates = [
                (int(item["source_layer"]), int(item["destination_layer"]), float(item["alpha"]))
                for item in stored_schedule
            ]
            if set(stored_candidates) != candidate_keys or len(stored_candidates) != len(candidates):
                parser.error("cannot resume output whose stored candidate schedule differs from this scan")
            candidates = stored_candidates
            candidate_schedule = _candidate_schedule(candidates)
            scan_indices = {candidate: index for index, candidate in enumerate(candidates)}
        results = [
            item
            for item in previous.get("results", [])
            if (item["source_layer"], item["destination_layer"], item["alpha"]) in candidate_keys
        ]
        for item in results:
            item["scan_index"] = scan_indices[(item["source_layer"], item["destination_layer"], item["alpha"])]
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
            pad_token_id = int(
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            if args.corpus:
                native = _score_native_dense(
                    model,
                    contexts,
                    args.row_batch_size,
                    pad_token_id,
                    args.projection_chunk_tokens,
                )
            else:
                native = _score_candidate(
                    model,
                    prefix,
                    contexts,
                    native_candidate,
                    args.scheduler,
                    args.python_threads,
                    args.row_batch_size,
                    pad_token_id,
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
        "scan_order": args.scan_order,
        "scan_seed": args.scan_seed,
        "candidate_schedule": candidate_schedule,
        "max_distance": args.max_distance,
        "scheduler": args.scheduler,
        "candidate_workers": args.candidate_workers,
        "requested_candidate_batch_size": requested_candidate_batch_size,
        "candidate_batch_size": args.candidate_batch_size,
        "candidate_token_budget": args.candidate_token_budget,
        "python_threads": args.python_threads,
        "requested_row_batch_size": requested_row_batch_size,
        "row_batch_size": args.row_batch_size,
        "projection_chunk_tokens": args.projection_chunk_tokens,
        "mask_free_unpadded": args.mask_free_unpadded,
        "attention_backend": args.attention_backend,
        "dual_gemm": args.dual_gemm,
        "telemetry_interval": args.telemetry_interval,
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
    telemetry = _PathTelemetry(
        args.output,
        candidate_schedule,
        _candidate_work(contexts, args.row_batch_size),
        results,
        args.telemetry_interval,
    )
    telemetry.start()

    def progress_callback(candidate):
        scan_index = scan_indices[candidate]

        def update(event, **values):
            if event == "candidate_started":
                telemetry.candidate_started(candidate, scan_index)
            elif event == "batch_completed":
                telemetry.batch_completed(candidate, scan_index, **values)

        return update

    def accept_result(result, candidate):
        result["scan_index"] = scan_indices[candidate]
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
        telemetry.candidate_completed(result)
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

    torch.cuda.synchronize()
    pad_token_id = int(
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    if args.candidate_batch_size > 1:
        for candidate_batch in _candidate_batches(remaining_candidates, args.candidate_batch_size):
            batch_results = _score_candidate_batch(
                model,
                contexts,
                candidate_batch,
                args.row_batch_size,
                pad_token_id,
                args.projection_chunk_tokens,
                [progress_callback(candidate) for candidate in candidate_batch],
            )
            for result, candidate in zip(batch_results, candidate_batch):
                accept_result(result, candidate)
    else:
        with ThreadPoolExecutor(
            max_workers=args.candidate_workers, thread_name_prefix="recirculation-screen"
        ) as executor:
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
                    pad_token_id,
                    args.graph_prefix,
                    args.projection_chunk_tokens,
                    args.mask_free_unpadded,
                    args.dual_gemm,
                    progress_callback(candidate),
                ): candidate
                for candidate in remaining_candidates
            }
            for future in as_completed(futures):
                candidate = futures[future]
                accept_result(future.result(), candidate)

    telemetry.stop()
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
