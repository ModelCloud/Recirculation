# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for robust, paired recirculation screening."""

from __future__ import annotations

import math
import re

_CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")


def gsm8k_solution_target(answer: str, final_answer: str) -> str:
    """Convert a GSM8K rationale into the response format used by the prompt."""

    if "####" not in answer:
        raise ValueError("GSM8K answer does not contain ####")
    rationale = answer.rsplit("####", 1)[0]
    rationale = _CALCULATOR_ANNOTATION.sub("", rationale)
    rationale = "\n".join(line.strip() for line in rationale.splitlines() if line.strip()).strip()
    if not rationale:
        raise ValueError("GSM8K answer does not contain a rationale")
    return f"{rationale}\nThe final answer is {final_answer}"


def summarize_paired_losses(
    row_indices: list[int],
    native_nll: list[float],
    native_counts: list[int],
    candidate_nll: list[float],
    candidate_counts: list[int],
    *,
    tail_quantile: float = 0.9,
    tail_weight: float = 1.0,
    harm_tolerance: float = 0.0,
) -> dict:
    """Summarize candidate loss relative to native loss without hiding tail harm."""

    lengths = {len(row_indices), len(native_nll), len(native_counts), len(candidate_nll), len(candidate_counts)}
    if lengths != {len(row_indices)} or not row_indices:
        raise ValueError("paired loss inputs must be non-empty and have equal lengths")
    if not 0.0 <= tail_quantile < 1.0:
        raise ValueError("tail_quantile must be in [0, 1)")
    if tail_weight < 0.0 or harm_tolerance < 0.0:
        raise ValueError("tail_weight and harm_tolerance must be non-negative")
    if any(count < 1 for count in native_counts + candidate_counts):
        raise ValueError("every row must contain at least one target token")
    if native_counts != candidate_counts:
        raise ValueError("native and candidate target counts must match per row")

    rows = []
    for index, native_total, candidate_total, count in zip(row_indices, native_nll, candidate_nll, native_counts):
        native_average = native_total / count
        candidate_average = candidate_total / count
        rows.append(
            {
                "index": index,
                "target_tokens": count,
                "native_nll": native_average,
                "candidate_nll": candidate_average,
                "delta_nll": candidate_average - native_average,
            }
        )

    total_tokens = sum(native_counts)
    native_token_nll = sum(native_nll) / total_tokens
    candidate_token_nll = sum(candidate_nll) / total_tokens
    token_delta = candidate_token_nll - native_token_nll
    deltas = [row["delta_nll"] for row in rows]
    harms = sorted((max(delta, 0.0) for delta in deltas), reverse=True)
    tail_count = max(1, math.ceil((1.0 - tail_quantile) * len(rows)))
    tail_harm = sum(harms[:tail_count]) / tail_count
    screen_score = token_delta + tail_weight * tail_harm
    return {
        "screen_score": screen_score,
        "target_nll": candidate_token_nll,
        "target_perplexity": math.exp(candidate_token_nll),
        "native_target_nll": native_token_nll,
        "native_target_perplexity": math.exp(native_token_nll),
        "native_delta_nll": token_delta,
        "mean_row_delta_nll": sum(deltas) / len(deltas),
        "tail_harm_nll": tail_harm,
        "tail_quantile": tail_quantile,
        "tail_rows": tail_count,
        "tail_weight": tail_weight,
        "harm_tolerance": harm_tolerance,
        "improved_rows": sum(delta < -harm_tolerance for delta in deltas),
        "regressed_rows": sum(delta > harm_tolerance for delta in deltas),
        "neutral_rows": sum(abs(delta) <= harm_tolerance for delta in deltas),
        "target_tokens": total_tokens,
        "row_metrics": rows,
    }


def screen_result_key(result: dict) -> tuple[float, float, float]:
    """Sort robust paired results, with compatibility for historical NLL reports."""

    return (
        float(result.get("screen_score", result.get("answer_nll", math.inf))),
        float(result.get("target_nll", result.get("answer_nll", math.inf))),
        float(result.get("alpha", math.inf)),
    )


def paired_selection_entry(
    source: int,
    destination: int,
    alpha: float,
    summary: dict,
    *,
    harm_weight: float,
    max_correct_to_wrong: int | None,
) -> dict:
    """Build an asymmetric E2E selection record from paired outcomes."""

    paired = summary["paired_vs_baseline"]["numeric"]
    correct_to_wrong = paired["correct_to_wrong"]
    valid = max_correct_to_wrong is None or correct_to_wrong <= max_correct_to_wrong
    return {
        "source_layer": source,
        "destination_layer": destination,
        "alpha": alpha,
        "wrong_to_correct": paired["wrong_to_correct"],
        "correct_to_wrong": correct_to_wrong,
        "net_correct": paired["net_correct"],
        "harm_weight": harm_weight,
        "selection_score": paired["wrong_to_correct"] - harm_weight * correct_to_wrong,
        "valid": valid,
        "primary_metric": "numeric",
        "numeric_correct": summary["numeric_correct"],
        "numeric_accuracy": summary["numeric_accuracy"],
    }
