#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Unified Evalution benchmark runner with optional paired GSM8K recirculation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import torch
import yaml
from datasets import Dataset, load_dataset
from evalution import Transformers
from evalution.engines.base import LoglikelihoodOutput
from evalution.logbar import get_logger
from evalution.scorers.gsm8k import (
    INVALID_ANSWER,
    extract_format_insensitive_numeric_answer,
    gsm8k_platinum_numeric_target,
    numbers_equal,
)
from evalution.yaml import _TEST_FACTORIES as EVALUTION_BENCHMARK_FACTORIES
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import (
    RecirculationConfig,
    RecirculationController,
)
from recirculation.inference_defaults import resolve_dense_cuda_defaults

DEFAULT_MODEL = "/local-models/Llama-3.2-1B-Instruct"
MMLU_GROUP_ALIASES = {
    "mmlu_stem": "stem",
    "mmlu_humanities": "humanities",
}


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


def _parse_candidate(value: str) -> tuple[int, int, float]:
    try:
        source, destination, alpha = value.split(":")
        return int(source), int(destination), float(alpha)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidates must use SOURCE:DESTINATION:ALPHA") from exc


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


def _candidate_specs(paths, alphas, beta, ramp_tokens, *, candidates=None):
    values = (
        list(candidates)
        if candidates is not None
        else [(source, destination, alpha) for source, destination in paths for alpha in alphas]
    )
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


def _common_prefix_length(token_sequences, *, minimum_suffix_tokens: int = 0) -> int:
    """Return the exact token-identical prefix while reserving each required suffix."""

    if minimum_suffix_tokens < 0:
        raise ValueError("minimum_suffix_tokens must be non-negative")
    sequences = [list(sequence) for sequence in token_sequences]
    if not sequences:
        return 0
    limit = min(len(sequence) - minimum_suffix_tokens for sequence in sequences)
    if limit <= 0:
        return 0
    for index in range(limit):
        token = sequences[0][index]
        if any(sequence[index] != token for sequence in sequences[1:]):
            return index
    return limit


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


def _snapshot_mlx_baseline_prefix(model, prefix_ids: list[int]):
    """Materialize a reusable MLX-LM KV snapshot for an exact token prefix."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    if not prefix_ids:
        raise ValueError("baseline prefix snapshot requires at least one token")
    cache = make_prompt_cache(model)
    logits = model(mx.array([prefix_ids], dtype=mx.int32), cache=cache)
    states = tuple(tuple(layer_cache.state) for layer_cache in cache)
    mx.eval(logits, *(value for state in states for value in state))
    return states


def _restore_mlx_baseline_prefix(model, snapshot):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    if len(cache) != len(snapshot):
        raise ValueError("baseline snapshot layer count differs from model cache")
    for layer_cache, state in zip(cache, snapshot):
        layer_cache.state = state
    return cache


def _generate_mlx_baseline(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int,
    until,
    *,
    prefix_snapshot=None,
):
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    if not prompt_ids:
        raise ValueError("MLX baseline generation requires at least one prompt or suffix token")
    continuation = []
    prompt = mx.array(prompt_ids, dtype=mx.int32)
    prompt_cache = None if prefix_snapshot is None else _restore_mlx_baseline_prefix(model, prefix_snapshot)
    for token, _ in generate_step(prompt, model, max_tokens=max_new_tokens, prompt_cache=prompt_cache):
        value = int(token)
        continuation.append(value)
        if value == int(tokenizer.eos_token_id):
            break
    return _generation_result(tokenizer, continuation, until)


def _generate_mlx_candidates(runner, prompt_ids, snapshot, max_new_tokens, eos_token_id):
    kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": eos_token_id}
    if snapshot is None:
        return runner.generate(prompt_ids, **kwargs)
    return runner.generate_from_snapshot(prompt_ids, snapshot, **kwargs)


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


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _load_local_arrow(path: str, *_args: Any, **_kwargs: Any) -> Dataset:
    """Load a local Arrow dataset directly, without Hub lookup or cache copying."""

    return Dataset.from_file(path)


def _decode_cli_value(value: str) -> Any:
    """Decode JSON scalars/containers while keeping unquoted values ergonomic."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_suite_assignment(value: str) -> tuple[str, Any]:
    try:
        key, raw_value = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("suite arguments must use KEY=VALUE") from error
    if not key or not key.isidentifier():
        raise argparse.ArgumentTypeError("suite argument keys must be Python identifiers")
    return key, _decode_cli_value(raw_value)


def _parse_benchmark_assignment(value: str) -> tuple[str, str, Any]:
    try:
        benchmark_and_key, raw_value = value.split("=", 1)
        benchmark, key = benchmark_and_key.rsplit(".", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "benchmark arguments must use BENCHMARK.KEY=VALUE"
        ) from error
    if not benchmark or not key or not key.isidentifier():
        raise argparse.ArgumentTypeError(
            "benchmark arguments must use a benchmark name and Python-identifier key"
        )
    return benchmark, key, _decode_cli_value(raw_value)


def _available_benchmarks() -> list[str]:
    return sorted({*EVALUTION_BENCHMARK_FACTORIES, *MMLU_GROUP_ALIASES})


def _suite_kwargs(args: argparse.Namespace, benchmark: str) -> dict[str, Any]:
    kwargs = dict(args.suite_arg or [])
    for scoped_benchmark, key, value in args.benchmark_arg or []:
        if scoped_benchmark == benchmark:
            kwargs[key] = value
    if args.max_rows is not None:
        kwargs.setdefault("max_rows", args.max_rows)
    kwargs.setdefault("batch_size", args.batch_size)
    return kwargs


def _build_evalution_suite(benchmark: str, kwargs: dict[str, Any]):
    factory_name = "mmlu" if benchmark in MMLU_GROUP_ALIASES else benchmark
    kwargs = dict(kwargs)
    if benchmark in MMLU_GROUP_ALIASES:
        kwargs.setdefault("subsets", MMLU_GROUP_ALIASES[benchmark])
        # A materialized local Dataset lets the progress observer count the
        # selected subjects before MMLU begins its four-choice request stream.
        kwargs.setdefault("stream", False)
    factory = EVALUTION_BENCHMARK_FACTORIES.get(factory_name)
    if factory is None:
        raise ValueError(
            f"unknown Evalution benchmark {benchmark!r}; use --list-benchmarks to inspect targets"
        )
    try:
        suite = factory(**kwargs)
    except TypeError as error:
        raise ValueError(f"invalid arguments for Evalution benchmark {benchmark!r}: {error}") from error

    dataset_path = kwargs.get("dataset_path")
    if dataset_path is None or Path(str(dataset_path)).suffix != ".arrow":
        return suite
    local_path = Path(str(dataset_path)).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"local Arrow dataset not found: {local_path}")
    local_suite_type = type(
        f"Local{type(suite).__name__}",
        (type(suite),),
        {"dataset_loader": lambda self: _load_local_arrow},
    )
    return local_suite_type(**kwargs)


def _score_recirculation_chunks(runner, chunks, *, batch_size: int):
    """Score Evalution teacher-forcing chunks with paper-faithful CUDA replay."""

    if batch_size < 1:
        raise ValueError("recirculation scoring batch size must be positive")
    device = getattr(runner, "device", None)
    if device is None:
        device = next(runner.model.parameters()).device
    outputs = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        if any(chunk.score_start < 1 or chunk.score_count < 1 for chunk in batch):
            raise ValueError("recirculation scoring requires non-empty prefix and target spans")
        input_rows = [list(chunk.input_ids[:-1]) for chunk in batch]
        padded_length = max(map(len, input_rows))
        pad_token_id = getattr(runner.model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(runner.model.config, "eos_token_id", 0)
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = pad_token_id[0]
        tokens = torch.full(
            (len(batch), padded_length),
            int(pad_token_id or 0),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(tokens)
        targets_by_position: dict[int, tuple[list[int], list[int]]] = {}
        for row_index, (chunk, input_ids) in enumerate(zip(batch, input_rows, strict=True)):
            tokens[row_index, : len(input_ids)] = torch.tensor(
                input_ids,
                dtype=torch.long,
                device=device,
            )
            attention_mask[row_index, : len(input_ids)] = 1
            for target_offset in range(chunk.score_count):
                target_index = chunk.score_start + target_offset
                position = target_index - 1
                rows, targets = targets_by_position.setdefault(position, ([], []))
                rows.append(row_index)
                targets.append(int(chunk.input_ids[target_index]))
        row_nll, row_counts = runner.score(
            tokens,
            targets_by_position,
            attention_mask=attention_mask,
            return_per_row=True,
        )
        for chunk, nll, token_count in zip(batch, row_nll, row_counts, strict=True):
            outputs.append(
                LoglikelihoodOutput(
                    logprob=-float(nll),
                    is_greedy=False,
                    token_count=int(token_count),
                    metadata=dict(chunk.metadata),
                )
            )
    return outputs


@contextmanager
def _patch_evalution_recirculation_scoring(session, config: RecirculationConfig):
    """Route Evalution log-likelihood batches through the CUDA replay runner."""

    from recirculation.cuda_backend import CUDAPrefillRunner

    runner = CUDAPrefillRunner(
        session.model,
        config,
        fused=True,
        allow_terminal_padding=True,
        projection_chunk_tokens=16,
    )
    session_type = type(session)
    original_score_chunks = session_type._score_chunks

    def score_chunks(_session, chunks, *, batch_size):
        return _score_recirculation_chunks(runner, chunks, batch_size=batch_size)

    session_type._score_chunks = score_chunks
    try:
        yield
    finally:
        session_type._score_chunks = original_score_chunks


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write status without exposing a partially serialized JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".new")
    pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(path)


@contextmanager
def _observe_evalution_generation_progress(suite, callback):
    """Expose exact generation and multiple-choice row progress from Evalution."""

    import evalution.benchmarks.base as benchmark_base

    suite_type = type(suite)
    has_score_title = hasattr(suite_type, "score_progress_title")
    had_local_title = has_score_title and "score_progress_title" in suite_type.__dict__
    local_title = suite_type.__dict__.get("score_progress_title") if has_score_title else None
    resolved_title = suite_type.score_progress_title if has_score_title else None
    had_local_estimator = "_estimated_total_for_split" in suite_type.__dict__
    local_estimator = suite_type.__dict__.get("_estimated_total_for_split")
    resolved_estimator = getattr(suite_type, "_estimated_total_for_split", None)

    def score_progress_title(self, *, processed, aggregate_scores, invalid_predictions):
        assert resolved_title is not None
        title = resolved_title(
            self,
            processed=processed,
            aggregate_scores=aggregate_scores,
            invalid_predictions=invalid_predictions,
        )
        if self is suite:
            callback(
                processed=int(processed),
                aggregate_scores=dict(aggregate_scores),
                invalid_predictions=int(invalid_predictions),
            )
        return title

    def estimated_total(self, *, loaded_docs, split):
        assert resolved_estimator is not None
        total = resolved_estimator(self, loaded_docs=loaded_docs, split=split)
        if total is None and self is suite and hasattr(loaded_docs, "column_names"):
            selected = self._selected_subjects()
            if selected is not None and "subject" in loaded_docs.column_names:
                total = sum(
                    self._normalize_subset_token(subject) in selected
                    if hasattr(self, "_normalize_subset_token")
                    else str(subject).strip().lower().replace(" ", "_") in selected
                    for subject in loaded_docs["subject"]
                )
                if self.max_rows is not None:
                    total = min(total, self.max_rows)
        if self is suite and total is not None:
            callback(total=int(total))
        return total

    manual_progress_targets = {benchmark_base}
    suite_module = sys.modules.get(suite_type.__module__)
    if suite_module is not None and hasattr(suite_module, "manual_progress"):
        manual_progress_targets.add(suite_module)
    original_manual_progress = {
        module: module.manual_progress for module in manual_progress_targets
    }

    def wrap_manual_progress(original):
        def manual_progress(total, *args, **kwargs):
            title = kwargs.get("title")
            if title is None and args:
                title = args[0]
            progress = original(total, *args, **kwargs)
            if not isinstance(title, str):
                return progress
            if "scoring answer choices" in title:
                callback(total=int(total) // 4)
                completed_requests = 0

                class ChoiceProgressObserver:
                    def next(self, *next_args, **next_kwargs):
                        nonlocal completed_requests
                        result = progress.next(*next_args, **next_kwargs)
                        completed_requests += 1
                        if completed_requests % 4 == 0:
                            callback(
                                processed=completed_requests // 4,
                                aggregate_scores={},
                                invalid_predictions=0,
                            )
                        return result

                    def __getattr__(self, name):
                        return getattr(progress, name)

                return ChoiceProgressObserver()
            if ": scoring" in title:
                callback(total=int(total))
            return progress

        return manual_progress

    if has_score_title:
        suite_type.score_progress_title = score_progress_title
    if resolved_estimator is not None:
        suite_type._estimated_total_for_split = estimated_total
    for module, original in original_manual_progress.items():
        module.manual_progress = wrap_manual_progress(original)
    try:
        yield
    finally:
        for module, original in original_manual_progress.items():
            module.manual_progress = original
        if has_score_title:
            if had_local_title:
                suite_type.score_progress_title = local_title
            else:
                delattr(suite_type, "score_progress_title")
        if resolved_estimator is not None:
            if had_local_estimator:
                suite_type._estimated_total_for_split = local_estimator
            else:
                delattr(suite_type, "_estimated_total_for_split")


def _dense_evalution_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="evaluate.py run",
        description="Run one model against any number of Evalution benchmark suites.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default=None)
    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Evalution benchmark factory name; repeat to reuse one loaded model across suites.",
    )
    parser.add_argument(
        "--suite-arg",
        action="append",
        type=_parse_suite_assignment,
        default=None,
        help="Common Evalution suite constructor argument as KEY=VALUE; VALUE accepts JSON.",
    )
    parser.add_argument(
        "--benchmark-arg",
        action="append",
        type=_parse_benchmark_assignment,
        default=None,
        help="Per-suite override as BENCHMARK.KEY=VALUE; VALUE accepts JSON.",
    )
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep model/tokenizer loading offline by default.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    parser.add_argument(
        "--continuous-batching",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--paged-attention",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Experimentally capture paged varlen/decode CUDA graphs (off by default).",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=None,
        help=(
            "Continuous scheduler token-admission budget. CUDA auto mode derives a "
            "cache-safe value from the paged-cache geometry (65,536 with defaults)."
        ),
    )
    parser.add_argument("--max-blocks-per-request", type=int, default=8)
    parser.add_argument("--paged-num-blocks", type=int, default=256)
    parser.add_argument("--paged-block-size", type=int, default=256)
    parser.add_argument(
        "--path",
        type=_parse_path,
        default=None,
        help="Apply one recirculation path as SOURCE:DESTINATION to every selected suite.",
    )
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--ramp-tokens", type=int, default=0)
    parser.add_argument(
        "--report-every-seconds",
        type=float,
        default=60.0,
        help="Emit and persist partial Evalution metrics at this interval.",
    )
    args = parser.parse_args(argv)

    if args.list_benchmarks:
        print("\n".join(_available_benchmarks()))
        return 0
    if not args.benchmark:
        parser.error("at least one --benchmark is required")
    if args.output is None:
        parser.error("--output is required")
    if len(set(args.benchmark)) != len(args.benchmark):
        parser.error("duplicate --benchmark values are not allowed")
    unknown_scopes = sorted(
        {name for name, _key, _value in args.benchmark_arg or []} - set(args.benchmark)
    )
    if unknown_scopes:
        parser.error(f"--benchmark-arg refers to unselected benchmark(s): {unknown_scopes}")

    cuda_device = str(args.device).split(":", 1)[0] == "cuda"
    try:
        args.cuda_auto_optimized = resolve_dense_cuda_defaults(
            args,
            flash_available=cuda_device and is_flash_attn_2_available(),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")
    if min(
        args.max_batch_tokens,
        args.max_blocks_per_request,
        args.paged_num_blocks,
        args.paged_block_size,
    ) < 1:
        parser.error("paged scheduler sizes must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.report_every_seconds <= 0:
        parser.error("--report-every-seconds must be positive")
    recirculation_config = None
    if args.path is not None:
        try:
            recirculation_config = RecirculationConfig(
                source_layer=args.path[0],
                destination_layer=args.path[1],
                alpha=args.alpha,
                beta=args.beta,
                ramp_tokens=args.ramp_tokens,
            )
        except ValueError as error:
            parser.error(str(error))
        unsupported = sorted(
            benchmark
            for benchmark in args.benchmark
            if benchmark != "gsm8k_platinum"
            and benchmark != "mmlu"
            and benchmark not in MMLU_GROUP_ALIASES
        )
        if unsupported:
            parser.error(
                "generic recirculation evaluation currently supports gsm8k_platinum and MMLU; "
                f"unsupported: {unsupported}"
            )
        if not cuda_device:
            parser.error("generic recirculation evaluation currently requires a CUDA device")
        if not args.paged_attention:
            parser.error("recirculated GSM8K evaluation requires CUDA paged attention")

    try:
        suites = [
            _build_evalution_suite(benchmark, _suite_kwargs(args, benchmark))
            for benchmark in args.benchmark
        ]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    model_path = str(Path(args.model).expanduser().resolve()) if Path(args.model).exists() else args.model
    if Path(model_path).exists() and not (Path(model_path) / "config.json").is_file():
        raise FileNotFoundError(f"model config not found: {Path(model_path) / 'config.json'}")
    output_path = args.output.expanduser().resolve()
    status_path = output_path.with_suffix(".status.json")
    logger = get_logger()
    logger.info(
        "Evalution run: benchmarks=%s dtype=float16 attention=%s continuous=%s "
        "paged=%s cuda_graph=%s batch_size=%d max_batch_tokens=%d model=%s",
        args.benchmark,
        args.attention_backend,
        args.continuous_batching,
        args.paged_attention,
        args.cuda_graph,
        args.batch_size,
        args.max_batch_tokens,
        model_path,
    )
    attention = (
        f"paged|{args.attention_backend}" if args.paged_attention else args.attention_backend
    )
    engine = Transformers(
        device=args.device,
        dtype="float16",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=attention,
        continuous_batching=args.continuous_batching,
        allow_block_sharing=True,
        max_batch_tokens=args.max_batch_tokens,
        max_blocks_per_request=args.max_blocks_per_request,
        use_async_batching=False,
        use_cuda_graph=args.cuda_graph,
    )
    cache_patch = nullcontext()
    if args.paged_attention:
        from recirculation.transformers_paged_patch import patch_transformers_paged_cache_defaults

        cache_patch = patch_transformers_paged_cache_defaults(
            num_blocks=args.paged_num_blocks,
            block_size=args.paged_block_size,
        )

    started = time.perf_counter()
    suite_status = [
        {
            "benchmark": benchmark,
            "name": suite.task_name(),
            "state": "pending",
            "completed_rows": 0,
            "total_rows": args.max_rows,
            "metrics": {},
        }
        for benchmark, suite in zip(args.benchmark, suites, strict=True)
    ]
    live_status = {
        "status": "running",
        "output": str(output_path),
        "elapsed_seconds": 0.0,
        "active_suite": None,
        "suites": suite_status,
    }

    def write_live_status() -> None:
        live_status["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(status_path, live_status)

    write_live_status()

    with cache_patch:
        evaluation = engine.model(
            path=model_path,
            label=args.model_label or Path(model_path).name,
            model_kwargs={"local_files_only": args.local_files_only},
            tokenizer_kwargs={"local_files_only": args.local_files_only},
        )
        def run_suite(index, suite, execution_patch) -> None:
            entry = suite_status[index]
            entry["state"] = "active"
            live_status["active_suite"] = entry["name"]
            last_report = [0.0]
            write_live_status()

            def progress_callback(**updates) -> None:
                total = updates.get("total")
                if total is not None:
                    entry["total_rows"] = int(total)
                processed = updates.get("processed")
                if processed is not None:
                    processed = int(processed)
                    entry["completed_rows"] = processed
                    aggregate_scores = updates.get("aggregate_scores", {})
                    entry["metric_totals"] = {
                        name: float(value) for name, value in aggregate_scores.items()
                    }
                    entry["metrics"] = {
                        name: float(value) / processed
                        for name, value in aggregate_scores.items()
                    } if processed else {}
                    entry["invalid_predictions"] = int(updates.get("invalid_predictions", 0))
                now = time.perf_counter()
                force = bool(
                    processed
                    and entry.get("total_rows")
                    and processed == entry["total_rows"]
                )
                if processed and (force or now - last_report[0] >= args.report_every_seconds):
                    metric_text = " ".join(
                        f"{name}={value:.4f}" for name, value in entry["metrics"].items()
                    )
                    logger.info(
                        "%s partial rows=%d/%s %s invalid=%d",
                        entry["name"],
                        processed,
                        entry.get("total_rows") or "?",
                        metric_text,
                        entry.get("invalid_predictions", 0),
                    )
                    write_live_status()
                    last_report[0] = now

            try:
                with _observe_evalution_generation_progress(
                    suite, progress_callback
                ), execution_patch:
                    evaluation.run(suite)
            except BaseException as error:
                entry["state"] = "failed"
                entry["error"] = f"{type(error).__name__}: {error}"
                live_status["status"] = "failed"
                live_status["active_suite"] = None
                write_live_status()
                raise
            entry["state"] = "complete"
            completed_results = getattr(evaluation, "_test_results", ())
            if completed_results:
                completed_result = completed_results[-1]
                entry["completed_rows"] = len(completed_result.samples)
                entry["total_rows"] = len(completed_result.samples)
                entry["metrics"] = {
                    name: float(value) for name, value in completed_result.metrics.items()
                }
            live_status["active_suite"] = None
            write_live_status()

        try:
            if recirculation_config is None:
                for index, suite in enumerate(suites):
                    run_suite(index, suite, nullcontext())
            else:
                from evalution.runtime import _describe_execution

                from recirculation.cuda_backend import FusedNormMix
                from recirculation.transformers_paged_patch import (
                    patch_evalution_paged_prefix_seeding,
                    patch_model_paged_recirculation,
                )

                evaluation._session = engine.build(evaluation._model_config)
                evaluation._execution = _describe_execution(evaluation._session)
                for index, (benchmark, suite) in enumerate(
                    zip(args.benchmark, suites, strict=True)
                ):
                    if benchmark == "gsm8k_platinum":
                        @contextmanager
                        def paged_generation_patch():
                            with (
                                patch_model_paged_recirculation(
                                    evaluation._session.model,
                                    recirculation_config,
                                    FusedNormMix(),
                                ),
                                patch_evalution_paged_prefix_seeding(
                                    evaluation._session,
                                    recirculation_config,
                                    block_size=args.paged_block_size,
                                    preview_size=args.batch_size,
                                ),
                            ):
                                yield

                        patch = paged_generation_patch()
                    else:
                        patch = _patch_evalution_recirculation_scoring(
                            evaluation._session,
                            recirculation_config,
                        )
                    run_suite(index, suite, patch)
            result = evaluation.result().to_dict()
        except Exception:
            evaluation.close()
            raise
    elapsed = time.perf_counter() - started
    row_count = sum(len(test.get("samples", [])) for test in result["tests"])
    result["provenance"] = {
        "git_commit": _git_commit(),
        "elapsed_seconds": elapsed,
        "rows": row_count,
        "rows_per_second": row_count / elapsed if elapsed else None,
        "benchmarks": args.benchmark,
        "suite_arguments": {
            benchmark: _suite_kwargs(args, benchmark) for benchmark in args.benchmark
        },
        "dtype_policy": "FP16 required for all repository evaluation runs",
        "model_local_files_only": args.local_files_only,
        "attention_backend": args.attention_backend,
        "continuous_batching": args.continuous_batching,
        "paged_attention_requested": args.paged_attention,
        "cuda_graph": args.cuda_graph,
        "recirculation_prefix_seed": getattr(
            getattr(evaluation, "_session", None), "_recirculation_prefix_seed", None
        ),
        "recirculation_admission": getattr(
            getattr(evaluation, "_session", None), "_recirculation_admission", None
        ),
        "forward_error_gate": {
            "metric": "mean_absolute",
            "limit": 4e-3,
            "kernel_classification": "fused_or_batched",
            "oracle": "unfused_unbatched_same_dtype",
        } if recirculation_config is not None else None,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "max_blocks_per_request": args.max_blocks_per_request,
        "paged_num_blocks": args.paged_num_blocks,
        "paged_block_size": args.paged_block_size,
        "cuda_auto_optimized": args.cuda_auto_optimized,
        "allocator_config": os.environ["PYTORCH_ALLOC_CONF"],
        "recirculation": (
            {
                "source_layer": recirculation_config.source_layer,
                "destination_layer": recirculation_config.destination_layer,
                "alpha": recirculation_config.alpha,
                "beta": recirculation_config.beta,
                "normalize_source": recirculation_config.normalize_source,
                "ramp_tokens": recirculation_config.ramp_tokens,
            }
            if recirculation_config is not None
            else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    live_status["status"] = "complete"
    live_status["active_suite"] = None
    write_live_status()
    logger.info(
        "evaluation complete: suites=%d rows=%d elapsed=%.2fs rows_per_second=%.3f output=%s",
        len(suites),
        row_count,
        elapsed,
        row_count / elapsed if elapsed else 0.0,
        output_path,
    )
    return 0


def _paired_gsm8k_main(argv: list[str]) -> int:
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
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        help="Repeat SOURCE:DESTINATION:ALPHA for explicit path/alpha arms without a Cartesian product.",
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
    parser.add_argument(
        "--no-mlx-shared-prefix",
        action="store_true",
        help="Disable exact cross-row MLX prefix snapshots for controlled performance comparisons.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-every", type=int, default=8)
    args = parser.parse_args(argv)
    if args.row_start < 0 or args.rows < 1 or args.max_new_tokens < 1 or args.candidate_batch_size < 1:
        raise ValueError("row-start must be non-negative and rows/max-new-tokens must be positive")
    paths = args.path or [(args.source_layer, args.destination_layer)]
    alphas = args.alpha or [0.10]
    if args.candidate is not None and (args.path is not None or args.alpha is not None):
        parser.error("--candidate cannot be combined with --path or --alpha")
    try:
        candidate_specs = _candidate_specs(
            paths,
            alphas,
            args.beta,
            args.ramp_tokens,
            candidates=args.candidate,
        )
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

    tokenizer = Tokenicer.load(args.model, local_files_only=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids_by_row = [_prompt_ids(tokenizer, str(document["question"]), fewshots) for document in documents]
    common_prefix_tokens = 0
    if backend == "mlx" and not args.no_mlx_shared_prefix:
        # Keep one token in every suffix because MLX-LM generation reads logits
        # from the suffix's final prompt token.
        common_prefix_tokens = _common_prefix_length(prompt_ids_by_row, minimum_suffix_tokens=1)
    shared_prefix = prompt_ids_by_row[0][:common_prefix_tokens] if common_prefix_tokens else []
    device = torch.device(resolved_device) if backend != "mlx" else None
    candidate_runners = []
    mlx_batches = []
    mlx_baseline_snapshot = None
    if backend == "mlx":
        from mlx_lm import load as load_mlx_model

        from recirculation.mlx_backend import CompiledNormMix, MLXCandidateGroupRecirculator

        model, _ = load_mlx_model(args.model)
        for batch in _candidate_batches(candidate_specs, args.candidate_batch_size):
            configs = [config for _, config in batch]
            runner = MLXCandidateGroupRecirculator(model, configs, [CompiledNormMix(config) for config in configs])
            mlx_batches.append((batch, runner, None))
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
    if backend == "mlx" and shared_prefix:
        mlx_baseline_snapshot = _snapshot_mlx_baseline_prefix(model, shared_prefix)
        snapshotted_batches = []
        for batch, runner, _ in mlx_batches:
            _, caches, pendings, _ = runner.prefill(shared_prefix)
            snapshotted_batches.append((batch, runner, runner.snapshot(caches, pendings)))
        mlx_batches = snapshotted_batches
    print(
        f"Selected backend={backend} device={resolved_device} candidates={len(candidate_specs)} "
        f"mlx_batches={len(mlx_batches) if backend == 'mlx' else 0} "
        f"common_prefix_tokens={common_prefix_tokens}",
        flush=True,
    )
    try:
        for relative_index, document in enumerate(documents):
            prompt_ids = prompt_ids_by_row[relative_index]
            sample = {
                "index": args.row_start + relative_index,
                "question": str(document["question"]),
                "gold_answer": _gold_answer(str(document["answer"])),
            }
            if backend == "mlx":
                suffix_ids = prompt_ids[common_prefix_tokens:]
                sample["baseline"] = _generate_mlx_baseline(
                    model,
                    tokenizer,
                    suffix_ids,
                    args.max_new_tokens,
                    until,
                    prefix_snapshot=mlx_baseline_snapshot,
                )
                for batch, runner, snapshot in mlx_batches:
                    continuations = _generate_mlx_candidates(
                        runner,
                        suffix_ids,
                        snapshot,
                        args.max_new_tokens,
                        int(tokenizer.eos_token_id),
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
            "common_prefix_tokens": common_prefix_tokens,
            "shared_prefix_reuse": backend == "mlx" and common_prefix_tokens > 0,
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Use 'run' for any Evalution benchmark, or 'paired-gsm8k' for the "
            "repository's dense-versus-recirculated GSM8K comparison."
        ),
    )
    parser.add_argument("command", choices=("run", "paired-gsm8k"))
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = parser.parse_args(argv[:1]).command
    if command == "run":
        return _dense_evalution_main(argv[1:])
    return _paired_gsm8k_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
