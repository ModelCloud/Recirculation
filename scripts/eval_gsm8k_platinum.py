#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Paired GSM8K-Platinum evaluation for dense Llama with and without recirculation."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
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
    if "####" not in answer:
        raise ValueError("GSM8K answer does not contain ####")
    value = _normalize_answer(answer.rsplit("####", 1)[1])
    if value is None:
        raise ValueError("GSM8K answer is empty")
    return value


@torch.inference_mode()
def _generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int,
    until,
    controller=None,
):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if controller is None:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            use_cache=True,
        )
    else:
        generated = controller.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=int(tokenizer.eos_token_id),
        )
    continuation = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    for stop in until:
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    strict, flexible = _extract_answer(text)
    return {
        "text": text,
        "strict_answer": strict,
        "flexible_answer": flexible,
        "token_count": len(continuation),
    }


def _summary(samples, arm: str):
    rows = len(samples)
    strict = sum(sample[arm]["strict_answer"] == sample["gold_answer"] for sample in samples)
    flexible = sum(sample[arm]["flexible_answer"] == sample["gold_answer"] for sample in samples)
    return {
        "rows": rows,
        "strict_correct": strict,
        "strict_accuracy": strict / rows,
        "flexible_correct": flexible,
        "flexible_accuracy": flexible / rows,
        "strict_invalid": sum(sample[arm]["strict_answer"] is None for sample in samples),
        "flexible_invalid": sum(sample[arm]["flexible_answer"] is None for sample in samples),
    }


def _paired(samples):
    result = {}
    for answer_key, label in (("strict_answer", "strict"), ("flexible_answer", "flexible")):
        changes = wrong_to_correct = correct_to_wrong = 0
        for sample in samples:
            baseline = sample["baseline"][answer_key]
            recirculated = sample["recirculated"][answer_key]
            gold = sample["gold_answer"]
            changes += baseline != recirculated
            wrong_to_correct += baseline != gold and recirculated == gold
            correct_to_wrong += baseline == gold and recirculated != gold
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--source-layer", type=int, default=11)
    parser.add_argument("--destination-layer", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--ramp-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-every", type=int, default=8)
    args = parser.parse_args()
    if args.row_start < 0 or args.rows < 1 or args.max_new_tokens < 1:
        raise ValueError("row-start must be non-negative and rows/max-new-tokens must be positive")

    device = torch.device(args.device)
    fewshots, until = _task_contract(args.task_config)
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
    stop = min(args.row_start + args.rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_start, stop)]
    if len(documents) != args.rows:
        raise ValueError(f"Requested {args.rows} rows but only {len(documents)} are available in the selected range")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=False,
    ).eval().to(device)
    recirculation = RecirculationConfig(
        source_layer=args.source_layer,
        destination_layer=args.destination_layer,
        alpha=args.alpha,
        beta=args.beta,
        ramp_tokens=args.ramp_tokens,
    )

    samples = []
    started = time.perf_counter()
    for relative_index, document in enumerate(documents):
        prompt_ids = _prompt_ids(tokenizer, str(document["question"]), fewshots)
        sample = {
            "index": args.row_start + relative_index,
            "question": str(document["question"]),
            "gold_answer": _gold_answer(str(document["answer"])),
        }
        sample["baseline"] = _generate(model, tokenizer, prompt_ids, device, args.max_new_tokens, until)
        with RecirculationController(model, recirculation) as controller:
            sample["recirculated"] = _generate(
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
            print(
                f"GSM8K rows {relative_index + 1}/{len(documents)} "
                f"baseline={_summary(samples, 'baseline')['flexible_correct']} "
                f"recirculated={_summary(samples, 'recirculated')['flexible_correct']} "
                f"paired={_paired(samples)['flexible']}",
                flush=True,
            )

    report = {
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "row_start": args.row_start,
            "rows": len(samples),
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "fewshot_count": len(fewshots),
            "chat_template": True,
            "max_new_tokens": args.max_new_tokens,
            "source_layer": args.source_layer,
            "destination_layer": args.destination_layer,
            "alpha": recirculation.alpha,
            "beta": recirculation.beta,
            "normalize_source": recirculation.normalize_source,
            "ramp_tokens": recirculation.ramp_tokens,
            "until": list(until),
        },
        "evaluation_seconds": time.perf_counter() - started,
        "summary": {arm: _summary(samples, arm) for arm in ("baseline", "recirculated")},
        "paired": _paired(samples),
        "samples": samples,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "paired": report["paired"]}, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
