#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Merge CUDA screen shards into one promotion-safe JSON and Markdown ledger."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation.screening import (
    objective_result_key,
    proxy_shortlist,
    render_screen_report_markdown,
    screen_leaders,
)

_SETTING_KEYS = (
    "scoring_schema",
    "model",
    "model_type",
    "decoder_layers",
    "dataset",
    "row_start",
    "rows",
    "row_stop_exclusive",
    "common_prefix_tokens",
    "corpus_prefix_policy",
    "target_mode",
    "tail_quantile",
    "tail_weight",
    "harm_tolerance",
    "shared_native_baseline",
)


def _candidate_key(result: dict) -> tuple[int, int, float]:
    return int(result["source_layer"]), int(result["destination_layer"]), float(result["alpha"])


def _read_report(path: Path, attempts: int = 10) -> dict:
    """Tolerate a legacy shard writer being observed between truncate and replace."""

    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05)
    raise AssertionError("unreachable")


def aggregate_reports(paths: list[Path]) -> dict:
    """Validate and merge disjoint shards, then rank the full completed population."""

    if not paths:
        raise ValueError("at least one shard report is required")
    reports = [_read_report(path) for path in paths]
    commit = reports[0].get("implementation_commit")
    reference_settings = reports[0].get("settings", {})
    results_by_key = {}
    sources = []
    total = 0
    for path, report in zip(paths, reports):
        if report.get("implementation_commit") != commit:
            raise ValueError(f"shard implementation commit differs: {path}")
        settings = report.get("settings", {})
        mismatches = {
            key: (settings.get(key), reference_settings.get(key))
            for key in _SETTING_KEYS
            if settings.get(key) != reference_settings.get(key)
        }
        if mismatches:
            raise ValueError(f"shard settings differ for {path}: {mismatches}")
        declared_total = report.get("total")
        if declared_total is None:
            declared_total = (
                int(report.get("complete", 0))
                + int(report.get("active", 0))
                + int(report.get("pending", 0))
            )
        total += int(declared_total)
        source_results = report.get("results", [])
        if len(source_results) != int(report.get("complete", len(source_results))):
            raise ValueError(f"shard complete count differs from stored results: {path}")
        for result in source_results:
            key = _candidate_key(result)
            if key in results_by_key:
                raise ValueError(f"candidate {key} occurs in more than one shard")
            results_by_key[key] = result
        sources.append(
            {
                "path": str(path.resolve()),
                "status": report.get("status"),
                "complete": len(source_results),
                "active": int(report.get("active", 0)),
                "pending": int(report.get("pending", 0)),
                "total": int(declared_total),
            }
        )

    results = list(results_by_key.values())
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
    results.sort(key=lambda result: objective_result_key(result, default_objective, robust=True))
    leaders = screen_leaders(results)
    complete = len(results)
    active = sum(source["active"] for source in sources)
    pending = max(total - complete - active, 0)
    status = "complete" if all(source["status"] == "complete" for source in sources) else "running"
    default_leader = leaders.get(f"{default_objective}_robust")
    settings = {
        **reference_settings,
        "aggregation": {
            "promotion_scope": "all completed candidates across every validated shard",
            "shards": len(paths),
            "source_reports": [str(path.resolve()) for path in paths],
        },
    }
    return {
        "status": status,
        "complete": complete,
        "active": active,
        "pending": pending,
        "total": total,
        "implementation_commit": commit,
        "settings": settings,
        "source_reports": sources,
        "best": default_leader,
        "best_perplexity": leaders.get(f"{default_objective}_perplexity"),
        "best_robust": default_leader,
        "leaders": leaders,
        "shortlist": proxy_shortlist(results, min(8, len(results))) if results else [],
        "results": results,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_aggregate(paths: list[Path], output: Path, markdown: Path) -> dict:
    report = aggregate_reports(paths)
    _atomic_write(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown, render_screen_report_markdown(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=0.0,
        help="Refresh until every shard completes; zero writes once and exits.",
    )
    args = parser.parse_args()
    if args.watch_seconds < 0:
        parser.error("watch-seconds must be non-negative")
    markdown = args.markdown or args.output.with_suffix(".md")
    while True:
        report = write_aggregate(args.inputs, args.output, markdown)
        print(
            f"aggregate complete={report['complete']} active={report['active']} pending={report['pending']} "
            f"json={args.output} markdown={markdown}",
            flush=True,
        )
        if not args.watch_seconds or report["status"] == "complete":
            return 0
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
