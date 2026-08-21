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
    windows: list[dict], corpora: list[str], rows_per_corpus: int, scan_seed: int = 20260821
) -> list[int]:
    selected = []
    for corpus in corpora:
        matches = [index for index, row in enumerate(windows) if row["corpus"] == corpus]
        if len(matches) < rows_per_corpus:
            raise ValueError(f"{corpus} has only {len(matches)} windows, need {rows_per_corpus}")
        random.Random(f"{scan_seed}:{corpus}").shuffle(matches)
        selected.extend(matches[:rows_per_corpus])
    return selected


def _derive_stage_artifacts(
    corpus_path: Path,
    baseline_path: Path,
    output_dir: Path,
    rows_per_corpus: int,
    window_tokens: int | None = None,
    scan_seed: int = 20260821,
) -> tuple[Path, Path]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    indices = _stage_indices(corpus["windows"], corpus["corpora"], rows_per_corpus, scan_seed)
    window_tokens = corpus["window_tokens"] if window_tokens is None else window_tokens
    if not 3 <= window_tokens <= corpus["window_tokens"]:
        raise ValueError("stage window length must be between 3 and the source window length")
    stage_corpus = {
        **corpus,
        "windows_per_corpus": rows_per_corpus,
        "corpus_counts": {
            name: {"windows": rows_per_corpus, "derived_from": str(corpus_path)}
            for name in corpus["corpora"]
        },
        "window_tokens": window_tokens,
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
        counts = [values["row_target_counts"][index] for index in indices]
        stage_baseline["objectives"][objective] = {
            "row_nll_totals": nll,
            "row_target_counts": counts,
            "target_nll": sum(nll) / sum(counts),
            "target_tokens": sum(counts),
        }
    suffix = f"{rows_per_corpus}x{len(corpus['corpora'])}_tok{window_tokens}"
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
    if args.stage_rows_per_corpus[2] != corpus["windows_per_corpus"]:
        parser.error("final stage must use every window from each corpus")
    if args.stage_window_tokens[2] != corpus["window_tokens"]:
        parser.error("final stage must use the complete source windows")
    status_path = args.output_dir / "race.status.json"
    history = []
    selected = None
    stage_outputs = []
    unrestricted_path_count = _model_path_count(args.model)
    for stage_index, (rows_per_corpus, window_tokens) in enumerate(
        zip(args.stage_rows_per_corpus, args.stage_window_tokens), start=1
    ):
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
            str(rows_per_corpus),
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
            str(min(args.row_batch_size, rows_per_corpus * len(corpus["corpora"]))),
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
            f"Starting screening race stage {stage_index}/3: {rows_per_corpus * len(corpus['corpora'])} "
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
        history.append(_history_entry(stage_index, rows_per_corpus * len(corpus["corpora"]), selected))

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
        },
    )
    LOG.info(f"CUDA screening race complete: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
