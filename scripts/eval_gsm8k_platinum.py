#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Paired GSM8K-Platinum evaluation for dense Llama with and without recirculation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from evalution.scorers.gsm8k import (
    INVALID_ANSWER,
    extract_format_insensitive_numeric_answer,
    gsm8k_platinum_numeric_target,
    numbers_equal,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import (
    RecirculationConfig,
    RecirculationController,
)


def _task_contract(path: Path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    samples = document["fewshot_config"]["samples"]
    if int(document["num_fewshot"]) != len(samples):
        raise ValueError("GSM8K task num_fewshot does not match its fixed samples")
    fewshots = [(str(sample["question"]), str(sample["target"])) for sample in samples]
    until = tuple(str(value) for value in document["generation_kwargs"]["until"])
    return fewshots, until


def _parse_range(value: str) -> tuple[int, int]:
    start, stop = value.split(":")
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise argparse.ArgumentTypeError("ranges must be non-empty START:STOP intervals")
    return start, stop


def _parse_path(value: str) -> tuple[int, int]:
    try:
        source, destination = value.split(":")
        return int(source), int(destination)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("paths must use SOURCE:DESTINATION") from exc


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_accelerator_available() -> bool:
    return torch.cuda.is_available() and _module_available("triton")


def _mlx_accelerator_available() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and _module_available("mlx")
        and _module_available("mlx_lm")
    )


def _automatic_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_backend(requested: str, device: str) -> tuple[str, str]:
    """Choose the fastest installed accelerator while honoring explicit overrides."""

    if requested == "auto" and device != "auto":
        if device.startswith("cuda") and _cuda_accelerator_available():
            return "cuda", device
        return "torch", device
    if requested == "auto":
        if _cuda_accelerator_available():
            return "cuda", "cuda"
        if _mlx_accelerator_available():
            return "mlx", "metal"
        return "torch", _automatic_torch_device()
    if requested == "cuda":
        if not _cuda_accelerator_available():
            raise RuntimeError("the CUDA backend requires an available CUDA device and Triton")
        if device not in ("auto", "cuda") and not device.startswith("cuda:"):
            raise ValueError("the CUDA backend requires --device auto, cuda, or cuda:N")
        return "cuda", "cuda" if device == "auto" else device
    if requested == "mlx":
        if not _mlx_accelerator_available():
            raise RuntimeError("the MLX backend requires Apple Silicon plus the mlx and mlx-lm packages")
        if device not in ("auto", "metal"):
            raise ValueError("the MLX backend requires --device auto or metal")
        return "mlx", "metal"
    if requested == "torch":
        return "torch", _automatic_torch_device() if device == "auto" else device
    raise ValueError(f"unknown backend: {requested}")


def _candidate_arm(source: int, destination: int, alpha: float) -> str:
    return f"source{source}_destination{destination}_alpha{alpha:g}"


def _candidate_specs(paths, alphas, beta, ramp_tokens):
    values = [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    if len(set(values)) != len(values):
        raise ValueError("duplicate recirculation candidates are not allowed")
    single = len(values) == 1
    return [
        (
            "recirculated" if single else _candidate_arm(source, destination, alpha),
            RecirculationConfig(
                source_layer=source,
                destination_layer=destination,
                alpha=alpha,
                beta=beta,
                ramp_tokens=ramp_tokens,
            ),
        )
        for source, destination, alpha in values
    ]


def _candidate_batches(specs, batch_size):
    by_destination = {}
    for spec in specs:
        by_destination.setdefault(spec[1].destination_layer, []).append(spec)
    return [
        group[start : start + batch_size]
        for group in by_destination.values()
        for start in range(0, len(group), batch_size)
    ]


def _instruction(question: str) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {question}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
    )


def _prompt_ids(tokenizer, question: str, fewshots: list[tuple[str, str]]) -> list[int]:
    messages = []
    for fewshot_question, target in fewshots:
        messages.extend(
            [
                {"role": "user", "content": _instruction(fewshot_question)},
                {"role": "assistant", "content": target},
            ]
        )
    messages.append({"role": "user", "content": _instruction(question)})
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return [int(token) for token in tokenizer(rendered, add_special_tokens=False).input_ids]


def _normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace(",", "").replace("$", "").strip().rstrip(".")
    return normalized or None


def _extract_answer(text: str) -> tuple[str | None, str | None]:
    import re

    strict = re.findall(r"The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))", text, re.IGNORECASE)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    strict_value = strict[-1][0] if strict else None
    flexible_value = next((left or right for left, right in reversed(flexible)), None)
    return _normalize_answer(strict_value), _normalize_answer(flexible_value)


def _gold_answer(answer: str) -> str:
    return gsm8k_platinum_numeric_target({"answer": answer})


def _generation_result(tokenizer, continuation: list[int], until):
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    for stop in until:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    strict, flexible = _extract_answer(text)
    return {
        "text": text,
        "numeric_answer": extract_format_insensitive_numeric_answer(text),
        "strict_answer": strict,
        "flexible_answer": flexible,
        "token_count": len(continuation),
    }


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
    until,
    controller=None,
    cache_implementation: str | None = None,
):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if controller is None:
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "use_cache": True,
        }
        if cache_implementation is not None:
            generate_kwargs["cache_implementation"] = cache_implementation
        generated = model.generate(
            **generate_kwargs,
        )
    else:
        generated = controller.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
        )
    continuation = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    return _generation_result(tokenizer, continuation, until)


@torch.inference_mode()
def _generate_cuda(runner, tokenizer, prompt_ids: list[int], device, max_new_tokens: int, until):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = runner.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    continuation = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    return _generation_result(tokenizer, continuation, until)


def _generate_mlx_baseline(model, tokenizer, prompt_ids: list[int], max_new_tokens: int, until):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    continuation = []
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    for token, _ in generate_step(prompt, model, max_tokens=max_new_tokens):
        value = int(token)
        continuation.append(value)
        if value == int(tokenizer.eos_token_id):
            break
    return _generation_result(tokenizer, continuation, until)


def _summary(samples, arm: str):
    rows = len(samples)
    numeric = sum(numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"]) for sample in samples)
    strict = sum(sample[arm]["strict_answer"] == sample["gold_answer"] for sample in samples)
    flexible = sum(sample[arm]["flexible_answer"] == sample["gold_answer"] for sample in samples)
    return {
        "rows": rows,
        "numeric_correct": numeric,
        "numeric_accuracy": numeric / rows,
        "numeric_invalid": sum(sample[arm]["numeric_answer"] == INVALID_ANSWER for sample in samples),
        "strict_correct": strict,
        "strict_accuracy": strict / rows,
        "flexible_correct": flexible,
        "flexible_accuracy": flexible / rows,
        "strict_invalid": sum(sample[arm]["strict_answer"] is None for sample in samples),
        "flexible_invalid": sum(sample[arm]["flexible_answer"] is None for sample in samples),
    }


def _paired(samples, arm: str = "recirculated"):
    result = {}
    for answer_key, label in (
        ("numeric_answer", "numeric"),
        ("strict_answer", "strict"),
        ("flexible_answer", "flexible"),
    ):
        changes = wrong_to_correct = correct_to_wrong = 0
        for sample in samples:
            baseline = sample["baseline"][answer_key]
            recirculated = sample[arm][answer_key]
            gold = sample["gold_answer"]
            changes += baseline != recirculated
            if label == "numeric":
                baseline_correct = numbers_equal(baseline, gold)
                recirculated_correct = numbers_equal(recirculated, gold)
            else:
                baseline_correct = baseline == gold
                recirculated_correct = recirculated == gold
            wrong_to_correct += not baseline_correct and recirculated_correct
            correct_to_wrong += baseline_correct and not recirculated_correct
        result[label] = {
            "answer_changes": changes,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct": wrong_to_correct - correct_to_wrong,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    parser.add_argument("--backend", choices=("auto", "cuda", "mlx", "torch"), default="auto")
    parser.add_argument("--device", default="auto", help="Torch/CUDA device override; auto chooses the fastest backend.")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument(
        "--forbid-range",
        action="append",
        type=_parse_range,
        default=None,
        help="Reject an evaluation range overlapping this START:STOP interval; repeat as needed.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--source-layer", type=int, default=11)
    parser.add_argument("--destination-layer", type=int, default=4)
    parser.add_argument(
        "--path",
        action="append",
        type=_parse_path,
        default=None,
        help="Repeat SOURCE:DESTINATION to evaluate multiple paths in one model-loaded process.",
    )
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--ramp-tokens", type=int, default=0)
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=8,
        help="Maximum same-destination candidates per exact MLX shared-lower group.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-every", type=int, default=8)
    args = parser.parse_args()
    if args.row_start < 0 or args.rows < 1 or args.max_new_tokens < 1 or args.candidate_batch_size < 1:
        raise ValueError("row-start must be non-negative and rows/max-new-tokens must be positive")
    paths = args.path or [(args.source_layer, args.destination_layer)]
    alphas = args.alpha or [0.10]
    try:
        candidate_specs = _candidate_specs(paths, alphas, args.beta, args.ramp_tokens)
        backend, resolved_device = _resolve_backend(args.backend, args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    evaluation_range = (args.row_start, args.row_start + args.rows)
    forbidden_ranges = args.forbid_range or []
    overlaps = [interval for interval in forbidden_ranges if _overlaps(evaluation_range, interval)]
    if overlaps:
        parser.error(f"evaluation range {evaluation_range} overlaps forbidden search range(s): {overlaps}")

    fewshots, until = _task_contract(args.task_config)
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
    stop = min(args.row_start + args.rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_start, stop)]
    if len(documents) != args.rows:
        raise ValueError(f"Requested {args.rows} rows but only {len(documents)} are available in the selected range")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device(resolved_device) if backend != "mlx" else None
    candidate_runners = []
    mlx_batches = []
    if backend == "mlx":
        from mlx_lm import load as load_mlx_model

        from recirculation.mlx_backend import CompiledNormMix, MLXCandidateGroupRecirculator

        model, _ = load_mlx_model(args.model)
        for batch in _candidate_batches(candidate_specs, args.candidate_batch_size):
            configs = [config for _, config in batch]
            runner = MLXCandidateGroupRecirculator(model, configs, [CompiledNormMix(config) for config in configs])
            mlx_batches.append((batch, runner))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            attn_implementation="eager",
            local_files_only=False,
        ).eval().to(device)
        if backend == "cuda":
            from recirculation.cuda_backend import CUDAConcurrentRunner

            candidate_runners = [
                (arm, config, CUDAConcurrentRunner(model, config)) for arm, config in candidate_specs
            ]

    samples = []
    started = time.perf_counter()
    print(
        f"Selected backend={backend} device={resolved_device} candidates={len(candidate_specs)} "
        f"mlx_batches={len(mlx_batches) if backend == 'mlx' else 0}",
        flush=True,
    )
    try:
        for relative_index, document in enumerate(documents):
            prompt_ids = _prompt_ids(tokenizer, str(document["question"]), fewshots)
            sample = {
                "index": args.row_start + relative_index,
                "question": str(document["question"]),
                "gold_answer": _gold_answer(str(document["answer"])),
            }
            if backend == "mlx":
                sample["baseline"] = _generate_mlx_baseline(
                    model, tokenizer, prompt_ids, args.max_new_tokens, until
                )
                for batch, runner in mlx_batches:
                    continuations = runner.generate(
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=int(tokenizer.eos_token_id),
                    )
                    for (arm, _), continuation in zip(batch, continuations):
                        sample[arm] = _generation_result(tokenizer, continuation, until)
            elif backend == "cuda":
                sample["baseline"] = _generate(
                    model, tokenizer, prompt_ids, device, args.max_new_tokens, until
                )
                for arm, _, runner in candidate_runners:
                    sample[arm] = _generate_cuda(
                        runner, tokenizer, prompt_ids, device, args.max_new_tokens, until
                    )
            else:
                sample["baseline"] = _generate(
                    model, tokenizer, prompt_ids, device, args.max_new_tokens, until
                )
                for arm, config in candidate_specs:
                    with RecirculationController(model, config) as controller:
                        sample[arm] = _generate(
                            model,
                            tokenizer,
                            prompt_ids,
                            device,
                            args.max_new_tokens,
                            until,
                            controller=controller,
                        )
            samples.append(sample)
            if (relative_index + 1) % args.report_every == 0 or relative_index + 1 == len(documents):
                counts = [f"baseline={_summary(samples, 'baseline')['numeric_correct']}"]
                paired = {}
                for arm, _ in candidate_specs:
                    counts.append(f"{arm}={_summary(samples, arm)['numeric_correct']}")
                    paired[arm] = _paired(samples, arm)["numeric"]
                print(
                    f"GSM8K rows {relative_index + 1}/{len(documents)} "
                    f"{' '.join(counts)} paired={paired}",
                    flush=True,
                )
    finally:
        for _, _, runner in candidate_runners:
            runner.close()

    candidate_settings = [
        {
            "arm": arm,
            "source_layer": config.source_layer,
            "destination_layer": config.destination_layer,
            "alpha": config.alpha,
            "beta": config.beta,
            "normalize_source": config.normalize_source,
            "ramp_tokens": config.ramp_tokens,
        }
        for arm, config in candidate_specs
    ]
    arms = ["baseline", *(arm for arm, _ in candidate_specs)]
    paired_report = (
        _paired(samples, candidate_specs[0][0])
        if len(candidate_specs) == 1
        else {arm: _paired(samples, arm) for arm, _ in candidate_specs}
    )

    report = {
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "row_start": args.row_start,
            "row_stop_exclusive": args.row_start + len(samples),
            "rows": len(samples),
            "forbidden_ranges": forbidden_ranges,
            "backend_requested": args.backend,
            "backend": backend,
            "device": resolved_device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "fewshot_count": len(fewshots),
            "chat_template": True,
            "max_new_tokens": args.max_new_tokens,
            "candidates": candidate_settings,
            "candidate_batch_size": args.candidate_batch_size,
            "candidate_batches": len(mlx_batches) if backend == "mlx" else len(candidate_specs),
            "until": list(until),
            "evalution": {
                "package_version": package_version("Evalution"),
                "task_variant": "cot_llama",
                "primary_metric": "acc,num",
                "scoring_mode": "numeric_format_insensitive",
                "extractor": "evalution.scorers.gsm8k.extract_format_insensitive_numeric_answer",
                "target": "evalution.scorers.gsm8k.gsm8k_platinum_numeric_target",
                "equality": "evalution.scorers.gsm8k.numbers_equal",
            },
        },
        "evaluation_seconds": time.perf_counter() - started,
        "summary": {arm: _summary(samples, arm) for arm in arms},
        "paired": paired_report,
        "samples": samples,
    }
    if len(candidate_specs) == 1:
        _, config = candidate_specs[0]
        report["settings"].update(
            {
                "source_layer": config.source_layer,
                "destination_layer": config.destination_layer,
                "alpha": config.alpha,
                "beta": config.beta,
                "normalize_source": config.normalize_source,
                "ramp_tokens": config.ramp_tokens,
            }
        )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "paired": report["paired"]}, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
