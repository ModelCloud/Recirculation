# SPDX-License-Identifier: Apache-2.0

"""Pure helpers for robust, paired recirculation screening."""

from __future__ import annotations

import math
import re

_CALCULATOR_ANNOTATION = re.compile(r"<<[^<>]*>>")
PROXY_RANKINGS = (
    ("language_modeling_perplexity", "language_modeling", False),
    ("language_modeling_robust", "language_modeling", True),
    ("final_answer_perplexity", "final_answer", False),
    ("final_answer_robust", "final_answer", True),
    ("full_solution_perplexity", "full_solution", False),
    ("full_solution_robust", "full_solution", True),
)


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


def perplexity_result_key(result: dict) -> tuple[float, float, float]:
    """Sort candidates solely by aggregate target perplexity/NLL."""

    return (
        float(result.get("target_perplexity", math.inf)),
        float(result.get("target_nll", result.get("answer_nll", math.inf))),
        float(result.get("alpha", math.inf)),
    )


def objective_result_key(result: dict, objective: str, *, robust: bool) -> tuple[float, float, float]:
    """Sort a dual-objective candidate by plain perplexity or robust paired loss."""

    summary = result["objectives"][objective]
    primary = summary["screen_score"] if robust else summary["target_perplexity"]
    return float(primary), float(summary["target_nll"]), float(result.get("alpha", math.inf))


def screen_leaders(results: list[dict]) -> dict[str, dict | None]:
    """Select every proxy leader from the full completed candidate population."""

    leaders = {}
    available_objectives = set(results[0].get("objectives", {})) if results else set()
    for name, objective, robust in PROXY_RANKINGS:
        if objective not in available_objectives:
            continue
        leaders[name] = min(
            results,
            key=lambda result, selected_objective=objective, selected_robust=robust: objective_result_key(
                result,
                selected_objective,
                robust=selected_robust,
            ),
        )
    return leaders


def proxy_shortlist(results: list[dict], limit: int) -> list[dict]:
    """Round-robin the four proxy rankings into one unique candidate shortlist."""

    if limit < 1:
        raise ValueError("shortlist limit must be positive")
    available_objectives = set(results[0].get("objectives", {})) if results else set()
    rankings = [
        sorted(
            results,
            key=lambda result, objective=objective, robust=robust: objective_result_key(
                result, objective, robust=robust
            ),
        )
        for _, objective, robust in PROXY_RANKINGS
        if objective in available_objectives
    ]
    selected = []
    selected_keys = set()
    depth = 0
    while len(selected) < min(limit, len(results)):
        added = False
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            item = ranking[depth]
            key = (item["source_layer"], item["destination_layer"], item["alpha"])
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
                added = True
                if len(selected) == min(limit, len(results)):
                    break
        depth += 1
        if not added and depth >= max(map(len, rankings), default=0):
            break
    return selected


def render_screen_report_markdown(report: dict) -> str:
    """Render a durable, human-readable ledger of every completed screen result."""

    results = report.get("results", [])
    total = report.get("total")
    if total is None:
        total = int(report.get("complete", 0)) + int(report.get("active", 0)) + int(report.get("pending", 0))
    lines = [
        "# CUDA Recirculation Screening Ledger",
        "",
        f"- Status: {report.get('status', 'unknown')}",
        (
            f"- Complete / active / pending / total: {report.get('complete', 0)} / "
            f"{report.get('active', 0)} / {report.get('pending', 0)} / {total}"
        ),
        f"- Implementation commit: `{report.get('implementation_commit', 'unknown')}`",
        f"- Promotion population: all {len(results)} completed candidates in this ledger",
        "",
        "## Current leaders",
        "",
        "| Ranking | Path | Alpha | Metric |",
        "|---|---:|---:|---:|",
    ]
    for name, objective, robust in PROXY_RANKINGS:
        leader = report.get("leaders", {}).get(name)
        if leader is None:
            lines.append(f"| {name} | — | — | — |")
            continue
        metrics = leader["objectives"][objective]
        metric = metrics["screen_score"] if robust else metrics["target_perplexity"]
        lines.append(
            f"| {name} | {leader['source_layer']}→{leader['destination_layer']} | "
            f"{leader['alpha']:.6g} | {metric:.9g} |"
        )
    language_modeling = bool(results and "language_modeling" in results[0].get("objectives", {}))
    lines.extend(
        [
            "",
            "## All completed candidates",
            "",
            "The promotion shortlist is recalculated from this entire table, never only from the most recent result.",
            "",
        ]
    )
    if language_modeling:
        lines.extend(
            [
                "| Path | Alpha | Seconds | LM PPL | LM ΔNLL | LM robust | LM tail harm | LM I/R/N |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Path | Alpha | Seconds | Final PPL | Final ΔNLL | Final robust | Final tail harm | "
                    "Final I/R/N | Full PPL | Full ΔNLL | Full robust | Full tail harm | Full I/R/N |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for result in results:
        if language_modeling:
            metrics = result["objectives"]["language_modeling"]
            lines.append(
                f"| {result['source_layer']}→{result['destination_layer']} | {result['alpha']:.6g} | "
                f"{result.get('seconds', 0.0):.3f} | {metrics['target_perplexity']:.9g} | "
                f"{metrics['native_delta_nll']:.9g} | {metrics['screen_score']:.9g} | "
                f"{metrics['tail_harm_nll']:.9g} | "
                f"{metrics['improved_rows']}/{metrics['regressed_rows']}/{metrics['neutral_rows']} |"
            )
            continue
        final = result.get("objectives", {}).get("final_answer")
        full = result.get("objectives", {}).get("full_solution")

        def values(summary):
            if summary is None:
                return ("—",) * 5
            return (
                f"{summary['target_perplexity']:.9g}",
                f"{summary['native_delta_nll']:.9g}",
                f"{summary['screen_score']:.9g}",
                f"{summary['tail_harm_nll']:.9g}",
                f"{summary['improved_rows']}/{summary['regressed_rows']}/{summary['neutral_rows']}",
            )

        final_values = values(final)
        full_values = values(full)
        lines.append(
            f"| {result['source_layer']}→{result['destination_layer']} | {result['alpha']:.6g} | "
            f"{result.get('seconds', 0.0):.3f} | " + " | ".join((*final_values, *full_values)) + " |"
        )
    return "\n".join(lines) + "\n"


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
    selection_score = paired["wrong_to_correct"] - harm_weight * correct_to_wrong
    within_harm_cap = max_correct_to_wrong is None or correct_to_wrong <= max_correct_to_wrong
    # A cap alone used to label the least-bad losing candidate as valid.  Proxy
    # likelihood can shortlist an arm, but only a positive naturally generated
    # paired result may be promoted.
    valid = within_harm_cap and paired["net_correct"] > 0 and selection_score > 0
    return {
        "source_layer": source,
        "destination_layer": destination,
        "alpha": alpha,
        "wrong_to_correct": paired["wrong_to_correct"],
        "correct_to_wrong": correct_to_wrong,
        "net_correct": paired["net_correct"],
        "harm_weight": harm_weight,
        "selection_score": selection_score,
        "valid": valid,
        "primary_metric": "numeric",
        "numeric_correct": summary["numeric_correct"],
        "numeric_accuracy": summary["numeric_accuracy"],
    }
