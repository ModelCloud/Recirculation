#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the dense GSM8K-Platinum baseline with Evalution's Transformers engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from evalution import Transformers
from evalution.benchmarks.gsm8k_platinum import GSM8KPlatinum
from evalution.logbar import get_logger

DEFAULT_MODEL = Path("/local-models/Llama-3.2-1B-Instruct")
DEFAULT_DATASET = Path("/local-models/datasets/gsm8k-platinum/test.arrow")
DEFAULT_OUTPUT = Path("results/dense_baselines/llama32_1b_instruct_gsm8k_platinum_evalution_fp16.json")


def _load_local_arrow(path: str, *_args: Any, **_kwargs: Any) -> Dataset:
    """Load the immutable local Arrow artifact without Hub access or a dataset cache."""

    return Dataset.from_file(path)


class LocalGSM8KPlatinum(GSM8KPlatinum):
    """Use Evalution's official suite logic with a directly loaded local dataset."""

    def dataset_loader(self) -> Any:
        return _load_local_arrow


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    return args


def main() -> None:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {model_path / 'config.json'}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"local GSM8K-Platinum Arrow file not found: {dataset_path}")

    logger = get_logger()
    logger.info(
        "dense Evalution Transformers run: dtype=float16 attention=sdpa batch_size=%d model=%s",
        args.batch_size,
        model_path,
    )
    # Paged SDPA has no Transformers decode fast path. Until FlashAttention is available for
    # this Python/CUDA build, fixed FP16 SDPA batches are both faster and substantially leaner.
    engine = Transformers(
        device="cuda",
        dtype="float16",
        batch_size=args.batch_size,
        attn_implementation="sdpa",
        continuous_batching=False,
    )
    suite = LocalGSM8KPlatinum(
        dataset_path=str(dataset_path),
        dataset_name=None,
        variant="cot_llama",
        apply_chat_template=True,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    started = time.perf_counter()
    run = engine.model(path=str(model_path), label="Llama-3.2-1B-Instruct").run(suite)
    result = run.result().to_dict()
    elapsed = time.perf_counter() - started
    samples = result["tests"][0]["samples"]
    result["provenance"] = {
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "rows": len(samples),
        "rows_per_second": len(samples) / elapsed,
        "dtype_policy": "FP16 required for all repository evaluation runs",
        "dataset_local_only": True,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        "dense baseline complete: rows=%d accuracy=%.4f elapsed=%.2fs rows_per_second=%.3f output=%s",
        len(samples),
        result["tests"][0]["metrics"]["acc,num"],
        elapsed,
        len(samples) / elapsed,
        output_path,
    )


if __name__ == "__main__":
    main()
