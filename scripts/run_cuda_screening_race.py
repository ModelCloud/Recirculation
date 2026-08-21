#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a conservative multi-fidelity CUDA path race on fixed corpus windows.

Every promoted finalist is rescored on the complete corpus. Earlier stages only
prune paths whose proxy rankings are outside a deliberately wide union of plain
perplexity and harm-penalized rankings.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

from logbar import LogBar

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG = LogBar.shared()


def _model_path_count(model: str) -> int | None:
    """Return the unrestricted source > destination path count for a local model."""
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layers = config.get("num_hidden_layers")
    if not isinstance(layers, int) or layers < 2:
        return None
    return layers * (layers - 1) // 2


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_indices(
    windows: list[dict], corpora: list[str], rows_per_corpus: int | dict[str, int], scan_seed: int = 20260821
) -> list[int]:
    selected = []
    for corpus in corpora:
        matches = [index for index, row in enumerate(windows) if row["corpus"] == corpus]
        requested = rows_per_corpus[corpus] if isinstance(rows_per_corpus, dict) else rows_per_corpus
        if len(matches) < requested:
            raise ValueError(f"{corpus} has only {len(matches)} windows, need {requested}")
        random.Random(f"{scan_seed}:{corpus}").shuffle(matches)
        selected.extend(matches[:requested])
    return selected


def _derive_stage_artifacts(
    corpus_path: Path,
    baseline_path: Path,
    output_dir: Path,
    rows_per_corpus: int | dict[str, int],
    window_tokens: int | None = None,
    scan_seed: int = 20260821,
) -> tuple[Path, Path]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    indices = _stage_indices(corpus["windows"], corpus["corpora"], rows_per_corpus, scan_seed)
    counts = {
        name: rows_per_corpus[name] if isinstance(rows_per_corpus, dict) else rows_per_corpus
        for name in corpus["corpora"]
    }
    window_tokens = corpus["window_tokens"] if window_tokens is None else window_tokens
    if not 3 <= window_tokens <= corpus["window_tokens"]:
        raise ValueError("stage window length must be between 3 and the source window length")
    stage_corpus = {
        **corpus,
        "windows_per_corpus": next(iter(set(counts.values()))) if len(set(counts.values())) == 1 else None,
        "corpus_window_counts": counts,
        "corpus_counts": {
            name: {"windows": counts[name], "derived_from": str(corpus_path)}
            for name in corpus["corpora"]
        },
        "window_tokens": window_tokens,
        "corpus_target_tokens": {
            name: min(corpus.get("corpus_target_tokens", {}).get(name, corpus["window_tokens"] - 1), window_tokens - 1)
            for name in corpus["corpora"]
        },
        "windows": [
            {**corpus["windows"][index], "token_ids": corpus["windows"][index]["token_ids"][:window_tokens]}
            for index in indices
        ],
        "source_indices": indices,
    }
    stage_baseline = {
        **baseline,
        "contract": {**baseline["contract"], "rows": len(indices)},
        "seconds": 0.0,
        "derived_from": str(baseline_path),
        "source_indices": indices,
        "objectives": {},
    }
    for objective, values in baseline["objectives"].items():
        nll = [values["row_nll_totals"][index] for index in indices]
        row_target_counts = [values["row_target_counts"][index] for index in indices]
        stage_baseline["objectives"][objective] = {
            "row_nll_totals": nll,
            "row_target_counts": row_target_counts,
            "target_nll": sum(nll) / sum(row_target_counts),
            "target_tokens": sum(row_target_counts),
        }
    count_suffix = "_".join(f"{name}{counts[name]}" for name in corpus["corpora"])
    suffix = f"{count_suffix}_tok{window_tokens}"
    corpus_output = output_dir / f"windows_{suffix}.json"
    baseline_output = output_dir / f"baseline_{suffix}.json"
    _atomic_json(corpus_output, stage_corpus)
    if window_tokens == corpus["window_tokens"]:
        _atomic_json(baseline_output, stage_baseline)
    return corpus_output, baseline_output


def _promote(results: list[dict], keep: int) -> list[dict]:
    def metrics(item):
        return item["objectives"]["language_modeling"]

    rankings = [
        sorted(
            results,
            key=lambda item: (
                metrics(item).get("paper_mean_percent_perplexity_change", float("inf")),
                item["scan_index"],
            ),
        ),
        sorted(results, key=lambda item: (metrics(item)["target_perplexity"], item["scan_index"])),
        sorted(results, key=lambda item: (metrics(item)["screen_score"], item["scan_index"])),
    ]
    promoted = []
    keys = set()
    depth = 0
    while len(promoted) < min(keep, len(results)):
        for ranking in rankings:
            if depth >= len(ranking):
                continue
            item = ranking[depth]
            key = (item["source_layer"], item["destination_layer"], item["alpha"])
            if key in keys:
                continue
            keys.add(key)
            promoted.append(item)
            if len(promoted) == min(keep, len(results)):
                break
        depth += 1
    return promoted


def _history_entry(stage: int, rows: int, promoted: list[dict]) -> dict:
    return {
        "stage": stage,
        "scored_windows": rows,
        "selected": [
            {
                "source_layer": item["source_layer"],
                "destination_layer": item["destination_layer"],
                "alpha": item["alpha"],
                "scan_index": item["scan_index"],
                "target_perplexity": item["objectives"]["language_modeling"]["target_perplexity"],
                "paper_mean_percent_perplexity_change": item["objectives"]["language_modeling"].get(
                    "paper_mean_percent_perplexity_change"
                ),
                "screen_score": item["objectives"]["language_modeling"]["screen_score"],
                "improved_rows": item["objectives"]["language_modeling"]["improved_rows"],
                "regressed_rows": item["objectives"]["language_modeling"]["regressed_rows"],
            }
            for item in promoted
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Qwen3-8B")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--corpus-artifact", type=Path, required=True)
    parser.add_argument("--native-baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=None)
    parser.add_argument(
        "--exact-final-corpus-counts",
        action="store_true",
        help="Use every artifact window in stage 3, including unequal paper-exact corpus counts.",
    )
    parser.add_argument("--stage-rows-per-corpus", type=int, nargs=3, default=(16, 64, 256))
    parser.add_argument("--stage-window-tokens", type=int, nargs=3, default=(256, 512, 1024))
    parser.add_argument("--stage-keep", type=int, nargs=2, default=(64, 16))
    parser.add_argument("--row-batch-size", type=int, default=128)
    parser.add_argument("--projection-chunk-tokens", type=int, default=16)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--scan-seed", type=int, default=20260821)
    args = parser.parse_args()
    if not (0 < args.stage_rows_per_corpus[0] < args.stage_rows_per_corpus[1] < args.stage_rows_per_corpus[2]):
        parser.error("stage row counts must be strictly increasing")
    if not (3 <= args.stage_window_tokens[0] < args.stage_window_tokens[1] < args.stage_window_tokens[2]):
        parser.error("stage window lengths must be strictly increasing")
    if not (args.stage_keep[0] > args.stage_keep[1] > 0):
        parser.error("stage keep counts must be strictly decreasing and positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus_artifact.read_text(encoding="utf-8"))
    artifact_counts = corpus.get("corpus_window_counts") or {
        name: corpus["windows_per_corpus"] for name in corpus["corpora"]
    }
    if not args.exact_final_corpus_counts and args.stage_rows_per_corpus[2] != corpus["windows_per_corpus"]:
        parser.error("final stage must use every window from each corpus")
    if args.stage_window_tokens[2] != corpus["window_tokens"]:
        parser.error("final stage must use the complete source windows")
    status_path = args.output_dir / "race.status.json"
    history = []
    selected = None
    stage_outputs = []
    unrestricted_path_count = _model_path_count(args.model)
    final_stage_output = None
    final_stage_corpus = None
    final_stage_baseline = None
    for stage_index, (configured_rows, window_tokens) in enumerate(
        zip(args.stage_rows_per_corpus, args.stage_window_tokens), start=1
    ):
        rows_per_corpus = artifact_counts if stage_index == 3 and args.exact_final_corpus_counts else configured_rows
        stage_corpus_counts = (
            rows_per_corpus
            if isinstance(rows_per_corpus, dict)
            else {name: rows_per_corpus for name in corpus["corpora"]}
        )
        stage_rows = sum(stage_corpus_counts.values())
        stage_corpus, stage_baseline = _derive_stage_artifacts(
            args.corpus_artifact.resolve(),
            args.native_baseline.resolve(),
            args.output_dir,
            rows_per_corpus,
            window_tokens,
            args.scan_seed,
        )
        output = args.output_dir / f"stage{stage_index}_paths.json"
        stage_outputs.append(str(output))
        command = [
            sys.executable,
            "-X",
            "gil=0",
            str(REPO_ROOT / "scripts/screen_cuda_recirculation.py"),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--windows-per-corpus",
            str(max(rows_per_corpus.values()) if isinstance(rows_per_corpus, dict) else rows_per_corpus),
            "--window-tokens",
            str(window_tokens),
            "--corpus-artifact",
            str(stage_corpus),
            "--native-baseline",
            str(stage_baseline),
            "--alpha",
            str(args.alpha),
            "--scan-order",
            "random",
            "--scan-seed",
            str(args.scan_seed + stage_index - 1),
            "--row-batch-size",
            str(min(args.row_batch_size, stage_rows)),
            "--candidate-workers",
            "1",
            "--candidate-batch-size",
            str(args.candidate_batch_size),
            "--projection-chunk-tokens",
            str(args.projection_chunk_tokens),
            "--mask-free-unpadded",
            "--attention-backend",
            "eager",
            "--dual-gemm",
            "--report-every",
            "1",
            "--telemetry-interval",
            "60",
            "--output",
            str(output),
        ]
        for corpus_name in corpus["corpora"]:
            command.extend(["--corpus", corpus_name])
            command.extend(["--corpus-window-count", f"{corpus_name}={stage_corpus_counts[corpus_name]}"])
            command.extend(
                [
                    "--corpus-target-tokens",
                    f"{corpus_name}={min(corpus.get('corpus_target_tokens', {}).get(corpus_name, corpus['window_tokens'] - 1), window_tokens - 1)}",
                ]
            )
        if selected is not None:
            for item in selected:
                command.extend(["--path", f"{item['source_layer']}:{item['destination_layer']}"])
        _atomic_json(
            status_path,
            {
                "status": "running",
                "active_stage": stage_index,
                "stage_rows_per_corpus": list(args.stage_rows_per_corpus),
                "stage_window_tokens": list(args.stage_window_tokens),
                "stage_keep": list(args.stage_keep),
                "stage_outputs": stage_outputs,
                "selected_path_history": history,
                "command": command,
            },
        )
        path_count = unrestricted_path_count if selected is None else len(selected)
        LOG.info(
            f"Starting screening race stage {stage_index}/3: {stage_rows} "
            f"windows x {window_tokens} tokens, {path_count if path_count is not None else 'all'} paths"
        )
        if not stage_baseline.exists():
            baseline_command = command + [
                "--baseline-only",
                "--output",
                str(args.output_dir / f"stage{stage_index}_baseline_probe.json"),
            ]
            LOG.info(f"Preparing dense shared baseline for race stage {stage_index}")
            subprocess.run(baseline_command, cwd=REPO_ROOT, check=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise RuntimeError(f"stage {stage_index} did not complete")
        if stage_index <= len(args.stage_keep):
            selected = _promote(report["results"], args.stage_keep[stage_index - 1])
        else:
            selected = _promote(report["results"], len(report["results"]))
        history.append(_history_entry(stage_index, stage_rows, selected))
        if stage_index == 3:
            final_stage_output = output
            final_stage_corpus = stage_corpus
            final_stage_baseline = stage_baseline

    final_report = json.loads(final_stage_output.read_text(encoding="utf-8"))
    winner = min(
        final_report["results"],
        key=lambda item: (
            item["objectives"]["language_modeling"].get(
                "paper_mean_percent_perplexity_change", float("inf")
            ),
            item["scan_index"],
        ),
    )
    alpha_output = None
    if args.alpha_grid:
        alpha_output = args.output_dir / "stage4_alphas.json"
        alpha_command = [
            sys.executable, "-X", "gil=0", str(REPO_ROOT / "scripts/screen_cuda_recirculation.py"),
            "--model", args.model, "--dtype", args.dtype,
            "--windows-per-corpus", str(max(artifact_counts.values())),
            "--window-tokens", str(corpus["window_tokens"]),
            "--corpus-artifact", str(final_stage_corpus),
            "--native-baseline", str(final_stage_baseline),
            "--path", f"{winner['source_layer']}:{winner['destination_layer']}",
            "--scan-order", "sequential", "--row-batch-size", str(args.row_batch_size),
            "--candidate-workers", "1", "--candidate-batch-size", str(args.candidate_batch_size),
            "--projection-chunk-tokens", str(args.projection_chunk_tokens), "--mask-free-unpadded",
            "--attention-backend", "eager", "--dual-gemm", "--report-every", "1",
            "--telemetry-interval", "60", "--output", str(alpha_output),
        ]
        for corpus_name in corpus["corpora"]:
            alpha_command.extend(["--corpus", corpus_name])
            alpha_command.extend(["--corpus-window-count", f"{corpus_name}={artifact_counts[corpus_name]}"])
            alpha_command.extend(
                [
                    "--corpus-target-tokens",
                    f"{corpus_name}={corpus.get('corpus_target_tokens', {}).get(corpus_name, corpus['window_tokens'] - 1)}",
                ]
            )
        for alpha in args.alpha_grid:
            alpha_command.extend(["--alpha", str(alpha)])
        LOG.info(
            f"Stage 3 paper-score winner is {winner['source_layer']}->{winner['destination_layer']}; "
            f"starting {len(args.alpha_grid)}-value alpha sweep"
        )
        subprocess.run(alpha_command, cwd=REPO_ROOT, check=True)

    _atomic_json(
        status_path,
        {
            "status": "complete",
            "active_stage": None,
            "stage_rows_per_corpus": list(args.stage_rows_per_corpus),
            "stage_window_tokens": list(args.stage_window_tokens),
            "stage_keep": list(args.stage_keep),
            "stage_outputs": stage_outputs,
            "selected_path_history": history,
            "finalists": history[-1]["selected"],
            "paper_path_winner": {
                "source_layer": winner["source_layer"],
                "destination_layer": winner["destination_layer"],
                "alpha": winner["alpha"],
                "paper_mean_percent_perplexity_change": winner["objectives"]["language_modeling"].get(
                    "paper_mean_percent_perplexity_change"
                ),
            },
            "alpha_sweep_output": str(alpha_output) if alpha_output is not None else None,
        },
    )
    LOG.info(f"CUDA screening race complete: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
