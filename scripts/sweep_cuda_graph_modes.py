#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Sweep supported CUDA Graph modes and stream priorities after capture."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from recirculation import RecirculationConfig
from recirculation.cuda_backend import (
    CUDAConcurrentRunner,
    CUDAGraphedConcurrentPrefill,
    CUDAPrefillRunner,
    measure_forward_error,
)


@dataclass(frozen=True)
class Variant:
    name: str
    capture_error_mode: str = "global"
    keep_graph: bool = False
    capture_priority: int = 0
    lower_priority: int = 0
    replay_priority: int = 0
    launch_priority: int = 0


VARIANTS = (
    Variant("global_auto_default"),
    Variant("global_manual_default", keep_graph=True),
    Variant("thread_local_auto_default", capture_error_mode="thread_local"),
    Variant("relaxed_auto_default", capture_error_mode="relaxed"),
    Variant("global_auto_high_capture", capture_priority=-3),
    Variant("global_auto_high_branches", lower_priority=-3, replay_priority=-3),
    Variant("global_auto_high_launch", launch_priority=-3),
    Variant(
        "global_manual_high_all",
        keep_graph=True,
        capture_priority=-3,
        lower_priority=-3,
        replay_priority=-3,
        launch_priority=-3,
    ),
)


def implementation_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def time_graph(graphed, tokens, repetitions: int, launch_stream: torch.cuda.Stream) -> list[float]:
    samples = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.cuda.stream(launch_stream):
            graphed.prefill(tokens)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--source-layer", type=int, default=12)
    parser.add_argument("--destination-layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).eval().to("cuda")
    )
    prompt = "Explain why recirculation can improve transformer inference. "
    token_ids = tokenizer(prompt * (args.tokens // 8 + 1), return_tensors="pt").input_ids[:, : args.tokens].cuda()
    changed_tokens = torch.roll(token_ids, shifts=1, dims=1)
    config = RecirculationConfig(
        source_layer=args.source_layer,
        destination_layer=args.destination_layer,
        alpha=args.alpha,
    )
    reference = CUDAPrefillRunner(model, config).prefill(changed_tokens)
    priority_range = torch.cuda.Stream.priority_range()
    variants = []

    for variant in VARIANTS:
        runner = CUDAConcurrentRunner(model, config)
        runner.lower_stream = torch.cuda.Stream(device=runner.device, priority=variant.lower_priority)
        runner.replay_stream = torch.cuda.Stream(device=runner.device, priority=variant.replay_priority)
        launch_stream = torch.cuda.Stream(device=runner.device, priority=variant.launch_priority)
        graphed = None
        try:
            graphed = CUDAGraphedConcurrentPrefill(
                runner,
                token_ids,
                warmups=args.warmups,
                capture_error_mode=variant.capture_error_mode,
                keep_graph=variant.keep_graph,
                capture_stream_priority=variant.capture_priority,
            )
            with torch.cuda.stream(launch_stream):
                candidate = graphed.prefill(changed_tokens)
            torch.cuda.synchronize()
            logits_error = measure_forward_error(reference[0], candidate[0])
            pending_reference = torch.cat((reference[2].destination, reference[2].source), dim=-1)
            pending_candidate = torch.cat((candidate[2].destination, candidate[2].source), dim=-1)
            pending_error = measure_forward_error(pending_reference, pending_candidate)
            logits_error.require()
            pending_error.require()
            with torch.cuda.stream(launch_stream):
                graphed.prefill(token_ids)
            torch.cuda.synchronize()
            samples = time_graph(graphed, token_ids, args.repetitions, launch_stream)
            variants.append(
                {
                    **asdict(variant),
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "mean_ms": statistics.mean(samples),
                    "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                    "logits_error": asdict(logits_error),
                    "logits_error_rate": logits_error.rate,
                    "pending_error": asdict(pending_error),
                    "pending_error_rate": pending_error.rate,
                }
            )
        except Exception as error:  # noqa: BLE001 - one failed experimental variant must not abort the sweep
            variants.append({**asdict(variant), "error": f"{type(error).__name__}: {error}"})
        finally:
            torch.cuda.synchronize()
            if graphed is not None:
                graphed.graph.reset()
            runner.close()
            del graphed, runner, launch_stream
            gc.collect()
            torch.cuda.empty_cache()

    successful = [variant for variant in variants if "median_ms" in variant]
    if not successful:
        raise RuntimeError("every CUDA Graph sweep variant failed")
    best = min(successful, key=lambda variant: variant["median_ms"])
    baseline = next(variant for variant in successful if variant["name"] == "global_auto_default")
    result = {
        "implementation_commit": implementation_commit(),
        "model": str(args.model),
        "tokens": token_ids.shape[1],
        "repetitions": args.repetitions,
        "warmups": args.warmups,
        "source_layer": args.source_layer,
        "destination_layer": args.destination_layer,
        "alpha": args.alpha,
        "python_gil_enabled": bool(getattr(__import__("sys"), "_is_gil_enabled", lambda: True)()),
        "stream_priority_range": priority_range,
        "best_variant": best["name"],
        "best_median_ms": best["median_ms"],
        "baseline_median_ms": baseline["median_ms"],
        "best_speedup_vs_baseline": baseline["median_ms"] / best["median_ms"],
        "variants": variants,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
