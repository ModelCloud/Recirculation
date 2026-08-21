#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run numeric proxy, full-solution, natural-generation, and holdout screening."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from logbar import LogBar

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation.screening import proxy_shortlist

LOG = LogBar.shared()


def _complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return report.get("status") == "complete" or "summary" in report or "summaries" in report
    except (OSError, json.JSONDecodeError):
        return False


def _run(command: list[str], *, output: Path, resume: bool, dry_run: bool) -> None:
    if resume and _complete(output):
        LOG.info(f"Resume: keeping completed {output}")
        return
    LOG.info("Running: " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _screen_base(args, output: Path, *, target_mode: str) -> list[str]:
    command = [
        sys.executable,
        "scripts/screen_cuda_recirculation.py",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--dtype",
        args.dtype,
        "--row-start",
        str(args.row_start),
        "--rows",
        str(args.rows),
        "--row-batch-size",
        str(args.row_batch_size),
        "--candidate-workers",
        str(args.candidate_workers),
        "--target-mode",
        target_mode,
        "--tail-quantile",
        str(args.tail_quantile),
        "--tail-weight",
        str(args.tail_weight),
        "--report-every",
        "1",
        "--output",
        str(output),
    ]
    for interval in args.forbid_range:
        command.extend(("--forbid-range", interval))
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--row-start", type=int, default=272)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument(
        "--forbid-range",
        action="append",
        default=None,
        help="Evaluation ranges excluded from tuning; explicit values replace the legacy defaults.",
    )
    parser.add_argument(
        "--path-alpha",
        type=float,
        default=None,
        help="Deprecated single-alpha override for stage 1; by default stage 1 evaluates --alpha-grid.",
    )
    parser.add_argument("--top-paths", type=int, default=8)
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[0.05, 0.075, 0.10, 0.125, 0.15, 0.20],
    )
    parser.add_argument("--e2e-top-k", type=int, default=5)
    parser.add_argument("--harm-weight", type=float, default=2.0)
    parser.add_argument("--max-correct-to-wrong", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--holdout-row-start", type=int, default=304)
    parser.add_argument("--holdout-rows", type=int, default=32)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument(
        "--candidate-workers",
        type=int,
        default=1,
        help="Concurrent candidates in one process; keep at 1 because model hooks are process-global.",
    )
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--tail-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/cuda_screening_funnel")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.forbid_range is None:
        args.forbid_range = ["0:272", "304:336"]
    if (
        min(
            args.rows,
            args.top_paths,
            args.e2e_top_k,
            args.holdout_rows,
            args.row_batch_size,
            args.candidate_workers,
        )
        < 1
    ):
        parser.error("row counts, shortlist sizes, and row batch size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF",
        "backend:native,expandable_segments:True,garbage_collection_threshold:0.6,"
        "roundup_power2_divisions:4,graph_capture_record_stream_reuse:True",
    )
    stage1 = args.output_dir / "stage1_paths.json"
    stage1_command = _screen_base(args, stage1, target_mode="final_answer")
    stage1_alphas = [args.path_alpha] if args.path_alpha is not None else args.alpha_grid
    for alpha in stage1_alphas:
        stage1_command.extend(("--alpha", str(alpha)))
    _run(stage1_command, output=stage1, resume=args.resume, dry_run=args.dry_run)
    if args.dry_run:
        return 0

    stage1_report = json.loads(stage1.read_text(encoding="utf-8"))
    top_paths = []
    for result in proxy_shortlist(stage1_report["results"], args.top_paths):
        path = (int(result["source_layer"]), int(result["destination_layer"]))
        if path not in top_paths:
            top_paths.append(path)
        if len(top_paths) == args.top_paths:
            break

    stage2 = args.output_dir / "stage2_alphas.json"
    stage2_command = _screen_base(args, stage2, target_mode="full_solution")
    for source, destination in top_paths:
        stage2_command.extend(("--path", f"{source}:{destination}"))
    for alpha in args.alpha_grid:
        stage2_command.extend(("--alpha", str(alpha)))
    _run(stage2_command, output=stage2, resume=args.resume, dry_run=False)

    e2e = args.output_dir / "stage3_e2e.json"
    e2e_command = [
        sys.executable,
        "scripts/sweep_gsm8k.py",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--device",
        "cuda",
        "--row-start",
        str(args.row_start),
        "--rows",
        str(args.rows),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--screen-results",
        str(stage2),
        "--top-k",
        str(args.e2e_top_k),
        "--harm-weight",
        str(args.harm_weight),
        "--output",
        str(e2e),
    ]
    if args.max_correct_to_wrong is not None:
        e2e_command.extend(("--max-correct-to-wrong", str(args.max_correct_to_wrong)))
    _run(e2e_command, output=e2e, resume=args.resume, dry_run=False)
    e2e_report = json.loads(e2e.read_text(encoding="utf-8"))
    winner = e2e_report.get("best")
    if winner is None or not winner["valid"]:
        raise RuntimeError("no naturally generated E2E candidate produced a positive harm-adjusted gain")

    holdout = args.output_dir / "stage4_holdout.json"
    holdout_command = [
        sys.executable,
        "scripts/eval_gsm8k_platinum.py",
        "--model",
        args.model,
        "--device",
        "cuda",
        "--row-start",
        str(args.holdout_row_start),
        "--rows",
        str(args.holdout_rows),
        "--forbid-range",
        f"{args.row_start}:{args.row_start + args.rows}",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--source-layer",
        str(winner["source_layer"]),
        "--destination-layer",
        str(winner["destination_layer"]),
        "--alpha",
        str(winner["alpha"]),
        "--output",
        str(holdout),
    ]
    _run(holdout_command, output=holdout, resume=args.resume, dry_run=False)
    LOG.info(f"Screening funnel complete: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
