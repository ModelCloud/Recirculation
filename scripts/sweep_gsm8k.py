#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Screen decoder-only LM recirculation paths on GSM8K-Platinum."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)

import torch
from datasets import Dataset, load_dataset
from evalution.scorers.gsm8k import numbers_equal
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from recirculation import (
    RecirculationConfig,
    RecirculationController,
)
from recirculation.inference_defaults import resolve_recirculation_cuda_defaults
from recirculation.screening import paired_selection_entry, proxy_shortlist
from scripts.evaluate import (
    _extract_answer,
    _generate,
    _generation_result,
    _gold_answer,
    _prompt_ids,
    _summary,
    _task_contract,
)


def _load_evalution_baseline(path: Path, samples: list[dict]) -> None:
    """Attach a completed Evalution dense run without repeating its inference."""

    report = json.loads(path.read_text(encoding="utf-8"))
    tests = report.get("tests", [])
    if len(tests) != 1:
        raise ValueError("Evalution baseline must contain exactly one test suite")
    baseline_by_index = {int(item["index"]): item for item in tests[0].get("samples", [])}
    for sample in samples:
        index = int(sample["index"])
        if index not in baseline_by_index:
            raise ValueError(f"Evalution baseline is missing dataset row {index}")
        item = baseline_by_index[index]
        if not numbers_equal(str(item["target"]), sample["gold_answer"]):
            raise ValueError(f"Evalution baseline target disagrees at dataset row {index}")
        text = str(item["prediction"])
        strict, flexible = _extract_answer(text)
        sample["baseline"] = {
            "text": text,
            "numeric_answer": str(item["extracted"]["numeric-extract"]),
            "strict_answer": strict,
            "flexible_answer": flexible,
            "token_count": None,
        }


def _parse_candidate(value: str) -> tuple[int, int, float]:
    try:
        source, destination, alpha = value.split(":")
        return int(source), int(destination), float(alpha)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be SOURCE:DESTINATION:ALPHA") from exc


def _paired_candidate_summary(samples, reference_arm: str, candidate_arm: str):
    result = {}
    for answer_key, label in (
        ("numeric_answer", "numeric"),
        ("strict_answer", "strict"),
        ("flexible_answer", "flexible"),
    ):
        changes = wrong_to_correct = correct_to_wrong = 0
        for sample in samples:
            reference = sample[reference_arm][answer_key]
            candidate = sample[candidate_arm][answer_key]
            gold = sample["gold_answer"]
            changes += reference != candidate
            if label == "numeric":
                reference_correct = numbers_equal(reference, gold)
                candidate_correct = numbers_equal(candidate, gold)
            else:
                reference_correct = reference == gold
                candidate_correct = candidate == gold
            wrong_to_correct += not reference_correct and candidate_correct
            correct_to_wrong += reference_correct and not candidate_correct
        result[label] = {
            "answer_changes": changes,
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "net_correct": wrong_to_correct - correct_to_wrong,
        }
    return result


def _implementation_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _arm_name(source: int, destination: int, alpha: float) -> str:
    return f"source{source}_destination{destination}_alpha{alpha:g}"


def _status_table(samples, candidates, total_rows: int, phase: str) -> str:
    """Render a compact live table, including paired flips for completed rows."""
    lines = [
        "State      Arm               Config           Rows    Correct    Accuracy    W→C    C→W    Net",
        "────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────",
    ]
    baseline_done = [sample for sample in samples if "baseline" in sample]
    baseline_complete = len(baseline_done) == total_rows
    baseline_correct = sum(numbers_equal(s["baseline"]["numeric_answer"], s["gold_answer"]) for s in baseline_done)
    baseline_state = "Complete" if baseline_complete else ("Active" if phase == "baseline" else "Pending")
    baseline_acc = f"{100 * baseline_correct / len(baseline_done):.2f}%" if baseline_done else "—"
    lines.append(f"{baseline_state:<9} Dense baseline    none             {len(baseline_done):>3}/{total_rows:<4}  {baseline_correct:>8}  {baseline_acc:>9}    —      —      —")
    lines.append("────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────")
    for index, (source, destination, alpha) in enumerate(candidates):
        arm = _arm_name(source, destination, alpha)
        done = [sample for sample in samples if arm in sample]
        complete = len(done) == total_rows
        state = "Complete" if complete else ("Active" if done else ("Active" if phase == f"candidate-{index}" else "Pending"))
        correct = sum(numbers_equal(s[arm]["numeric_answer"], s["gold_answer"]) for s in done)
        accuracy = f"{100 * correct / len(done):.2f}%" if done else "—"
        paired = [s for s in done if "baseline" in s]
        w2c = c2w = 0
        for sample in paired:
            base_ok = numbers_equal(sample["baseline"]["numeric_answer"], sample["gold_answer"])
            cand_ok = numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"])
            w2c += not base_ok and cand_ok
            c2w += base_ok and not cand_ok
        flip = f"{w2c:>5}  {c2w:>5}  {w2c - c2w:>+5}" if paired else "    —      —      —"
        config = f"{source}→{destination}, α={alpha:.2f}"
        lines.append(f"{state:<9} Recirculation     {config:<14} {len(done):>3}/{total_rows:<4}  {correct:>8}  {accuracy:>9}  {flip}")
        if index != len(candidates) - 1:
            lines.append("────────  ────────────────  ─────────────  ────────  ─────────  ──────────  ─────  ─────  ─────")
    return "\n".join(lines)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def _generate_cuda_batch(runner, tokenizer, prompt_ids_batch, device, max_new_tokens: int, until):
    """Generate a same-length prompt batch with one CUDA runner invocation."""
    input_ids = torch.tensor(prompt_ids_batch, dtype=torch.long, device=device)
    generated = runner.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=int(tokenizer.eos_token_id),
    )
    prefix_length = input_ids.shape[1]
    return [
        _generation_result(tokenizer, row[prefix_length:].detach().cpu().tolist(), until)
        for row in generated
    ]


@torch.inference_mode()
def _generate_cuda_graph_prefill(graph, runner, tokenizer, prompt_ids, device, max_new_tokens, until):
    """Use a fixed-shape graph for prompt prefill, then stream the decode tail."""
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache, pending, _ = graph.prefill(tokens)
    generated = tokens.clone()
    for _ in range(max_new_tokens):
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, token), dim=1)
        if bool((token == int(tokenizer.eos_token_id)).all()):
            break
        logits, cache, pending = runner.step(token, cache, pending)
    continuation = generated[0, tokens.shape[1] :].detach().cpu().tolist()
    return _generation_result(tokenizer, continuation, until)


@torch.inference_mode()
def _generate_cuda_from_snapshot(runner, snapshot, tokenizer, suffix_ids, device, max_new_tokens, until):
    """Restore a shared prefix and generate from the row-specific suffix."""
    suffix = torch.tensor([suffix_ids], dtype=torch.long, device=device)
    logits, cache, pending, _ = runner.prefill_from_snapshot(suffix, snapshot)
    continuation = []
    for _ in range(max_new_tokens):
        token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        value = int(token.item())
        continuation.append(value)
        if value == int(tokenizer.eos_token_id):
            break
        logits, cache, pending = runner.step(token, cache, pending)
    return _generation_result(tokenizer, continuation, until)


def _common_prefix_length(samples) -> int:
    length = min(len(sample["prompt_ids"]) for sample in samples)
    first = samples[0]["prompt_ids"]
    for position in range(length):
        value = first[position]
        if any(sample["prompt_ids"][position] != value for sample in samples[1:]):
            return position
    return length


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", default="madrylab/gsm8k-platinum")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--task-config", type=Path, default=REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--row-start", type=int, default=128)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--attention-backend",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Attention kernel override; CUDA auto-selects FlashAttention 2 for paged evaluation.",
    )
    parser.add_argument(
        "--cuda-paged-continuous",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the recirculation-aware paged scheduler; enabled automatically on CUDA.",
    )
    parser.add_argument("--cuda-paged-num-blocks", type=int, default=256)
    parser.add_argument("--cuda-paged-block-size", type=int, default=256)
    parser.add_argument(
        "--cuda-paged-admission-batch",
        type=int,
        default=None,
        help=(
            "Number of queued requests admitted together. Defaults to --cuda-batch-size; "
            "cohort admission avoids one-at-a-time long-prompt refills."
        ),
    )
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--baseline-results",
        type=Path,
        default=None,
        help="Reuse a completed Evalution dense-baseline JSON for paired flip metrics.",
    )
    parser.add_argument(
        "--screen-results",
        type=Path,
        default=None,
        help="Load the top candidates from a robust CUDA screen instead of passing --candidate repeatedly.",
    )
    parser.add_argument(
        "--cuda-graph-prefill",
        action="store_true",
        help="Capture fixed-shape CUDA graph prefill per candidate/prompt length before decode.",
    )
    parser.add_argument(
        "--cuda-graph-max-tokens",
        type=int,
        default=256,
        help="Maximum prompt length captured by CUDA graphs (raise cautiously for long prefixes).",
    )
    parser.add_argument(
        "--cuda-compile",
        action="store_true",
        help="Compile the CUDA model with max-autotune before evaluation.",
    )
    parser.add_argument(
        "--cuda-python-threads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Python worker threads to enqueue the two CUDA streams per decode step.",
    )
    parser.add_argument(
        "--cuda-static-cache",
        action="store_true",
        help="Request Transformers StaticCache for dense CUDA generation.",
    )
    parser.add_argument(
        "--cuda-compile-runner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile recirculation stacks; defaults to automatic break-even selection.",
    )
    parser.add_argument(
        "--cuda-shared-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefill the common CUDA prompt prefix once per candidate and restore it for every row.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--harm-weight",
        type=float,
        default=2.0,
        help="Selection penalty applied to each baseline correct-to-wrong regression.",
    )
    parser.add_argument(
        "--max-correct-to-wrong",
        type=int,
        default=None,
        help="Optional hard E2E validity gate for baseline correct-to-wrong regressions.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        help="Repeat SOURCE:DESTINATION:ALPHA; defaults to four middle-stack pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-every", type=float, default=60.0, help="Seconds between live status table updates.")
    parser.add_argument(
        "--cuda-batch-size",
        type=int,
        default=None,
        help="CUDA request width; defaults to 32 on CUDA and 4 elsewhere.",
    )
    args = parser.parse_args()
    try:
        cuda_device, cuda_auto_optimized = resolve_recirculation_cuda_defaults(
            args,
            flash_available=is_flash_attn_2_available(),
        )
    except ValueError as error:
        parser.error(str(error))
    if cuda_device:
        print(
            "CUDA inference settings: "
            f"attention={args.attention_backend} paged_continuous={args.cuda_paged_continuous} "
            f"batch={args.cuda_batch_size} admission={args.cuda_paged_admission_batch or args.cuda_batch_size} "
            f"blocks={args.cuda_paged_num_blocks} allocator={os.environ['PYTORCH_ALLOC_CONF']}",
            flush=True,
        )
    if args.screen_results is not None and args.candidate is not None:
        parser.error("use either --screen-results or --candidate, not both")
    if args.top_k < 1 or args.harm_weight < 1.0:
        parser.error("top-k must be positive and harm-weight must be at least 1")
    if args.max_correct_to_wrong is not None and args.max_correct_to_wrong < 0:
        parser.error("max-correct-to-wrong must be non-negative")
    if args.status_every <= 0:
        parser.error("status-every must be positive")
    if args.cuda_batch_size < 1:
        parser.error("cuda-batch-size must be positive")
    if args.cuda_graph_max_tokens < 1:
        parser.error("cuda-graph-max-tokens must be positive")
    if min(args.cuda_paged_num_blocks, args.cuda_paged_block_size) < 1:
        parser.error("paged cache block counts and sizes must be positive")
    if args.cuda_paged_admission_batch is not None and args.cuda_paged_admission_batch < 1:
        parser.error("cuda-paged-admission-batch must be positive")
    cuda_compile_runner = False
    if not args.cuda_paged_continuous:
        cuda_compile_runner = (
            args.cuda_compile_runner
            if args.cuda_compile_runner is not None
            else args.rows * args.max_new_tokens >= 5_000
        )
    selected_screen_items = {}
    if args.screen_results is not None:
        screen = json.loads(args.screen_results.read_text(encoding="utf-8"))
        screen_results = screen.get("results", [])
        candidates = (
            proxy_shortlist(screen_results, args.top_k)
            if screen_results and "objectives" in screen_results[0]
            else screen_results[: args.top_k]
        )
        if not candidates:
            parser.error("screen-results does not contain candidates")
        selected_screen_items = {
            (int(item["source_layer"]), int(item["destination_layer"]), float(item["alpha"])): item
            for item in candidates
        }
        args.candidate = [
            (int(item["source_layer"]), int(item["destination_layer"]), float(item["alpha"])) for item in candidates
        ]
    elif args.candidate is None:
        args.candidate = [(10, 3, 0.10), (11, 4, 0.10), (12, 5, 0.10), (13, 6, 0.10)]
    if args.row_start < 0 or args.rows < 1:
        raise ValueError("row-start must be non-negative and rows must be positive")

    device = torch.device(args.device)
    # CUDA evaluation must use fused SDPA/Flash attention.  Eager attention
    # dispatches every attention operation through Python and was the dominant
    # reason this accuracy sweep was slower than the MLX evaluator.
    attn_implementation = (
        "flash_attention_2"
        if args.cuda_paged_continuous
        else args.attention_backend
        if device.type == "cuda"
        else "eager"
    )
    if device.type == "cuda" and not args.cuda_paged_continuous:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    fewshots, until = _task_contract(args.task_config)
    dataset_path = Path(args.dataset).expanduser()
    dataset = (
        Dataset.from_file(str(dataset_path.resolve()))
        if dataset_path.is_file() and dataset_path.suffix == ".arrow"
        else load_dataset(args.dataset, name=args.dataset_config, split="test")
    )
    stop = min(args.row_start + args.rows, len(dataset))
    documents = [dataset[index] for index in range(args.row_start, stop)]
    if len(documents) != args.rows:
        raise ValueError("requested row range exceeds the dataset")
    tokenizer = Tokenicer.load(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            attn_implementation=attn_implementation,
        )
        .eval()
        .to(device)
    )
    compile_used = False
    if args.cuda_compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
            compile_used = True
            print("CUDA torch.compile enabled (max-autotune, dynamic=True)", flush=True)
        except (RuntimeError, torch.AcceleratorError) as error:
            print(f"CUDA torch.compile unavailable; using eager SDPA: {error}", flush=True)
    # Construct each CUDA runner once.  Creating a controller per row defeats
    # the fused replay and two-stream scheduling used by the benchmark path.
    candidate_runners = {}
    if device.type == "cuda" and not args.cuda_paged_continuous:
        from recirculation.cuda_backend import CUDAConcurrentRunner

        candidate_runners = {
            _arm_name(source, destination, alpha): CUDAConcurrentRunner(
                model,
                RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha),
                use_python_threads=args.cuda_python_threads,
            )
            for source, destination, alpha in args.candidate
        }
        if cuda_compile_runner:
            for runner in candidate_runners.values():
                compile_options = {"triton.cudagraphs": False}
                runner._run_lower = torch.compile(runner._run_lower, dynamic=True, options=compile_options)
                runner._run_upper = torch.compile(runner._run_upper, dynamic=True, options=compile_options)
            print("Compiled recirculation stacks enabled (Inductor CUDA graph trees disabled)", flush=True)
    graph_prefills = {}

    samples = []
    for relative_index, document in enumerate(documents):
        prompt_ids = _prompt_ids(tokenizer, str(document["question"]), fewshots)
        samples.append(
            {
                "index": args.row_start + relative_index,
                "question": str(document["question"]),
                "gold_answer": _gold_answer(str(document["answer"])),
                "prompt_ids": prompt_ids,
            }
        )

    if args.baseline_results is not None:
        _load_evalution_baseline(args.baseline_results.expanduser().resolve(), samples)

    started = time.perf_counter()
    status_path = args.output.expanduser().resolve().with_suffix(".status.json")
    last_status = 0.0

    def maybe_status(phase: str, force: bool = False) -> None:
        nonlocal last_status
        now = time.perf_counter()
        if not force and now - last_status < args.status_every:
            return
        last_status = now
        table = _status_table(samples, args.candidate, len(samples), phase)
        payload = {
            "implementation_commit": _implementation_commit(),
            "phase": phase,
            "elapsed_seconds": now - started,
            "total_rows": len(samples),
            "candidates": [list(candidate) for candidate in args.candidate],
            "table": table,
        }
        _write_status(status_path, payload)
        print(table, flush=True)

    maybe_status("baseline", force=True)
    if not args.skip_baseline and args.baseline_results is None:
        for sample_index, sample in enumerate(samples):
            try:
                sample["baseline"] = _generate(
                    model,
                    tokenizer,
                    sample["prompt_ids"],
                    device,
                    args.max_new_tokens,
                    until,
                    cache_implementation="static" if args.cuda_static_cache and device.type == "cuda" else None,
                )
            except (ValueError, RuntimeError, torch.AcceleratorError) as error:
                if not args.cuda_static_cache:
                    raise
                print(f"StaticCache unavailable; falling back to DynamicCache: {error}", flush=True)
                sample["baseline"] = _generate(
                    model, tokenizer, sample["prompt_ids"], device, args.max_new_tokens, until
                )
            maybe_status("baseline")
            if (sample_index + 1) % 4 == 0:
                print(f"baseline rows {sample_index + 1}/{len(samples)}", flush=True)
        maybe_status("candidates", force=True)
    elif args.baseline_results is not None:
        maybe_status("candidates", force=True)
    candidate_timings = {}
    for candidate_index, (source, destination, alpha) in enumerate(args.candidate):
        config = RecirculationConfig(source_layer=source, destination_layer=destination, alpha=alpha)
        arm = _arm_name(source, destination, alpha)
        if args.cuda_paged_continuous:
            from transformers import ContinuousBatchingConfig
            from transformers.generation.continuous_batching.requests import RequestStatus

            from recirculation.cuda_backend import FusedNormMix
            from recirculation.transformers_paged_patch import (
                patch_model_paged_recirculation,
                seed_paged_cache_from_snapshot,
            )

            model.generation_config.do_sample = False
            paged_config = ContinuousBatchingConfig(
                block_size=args.cuda_paged_block_size,
                num_blocks=args.cuda_paged_num_blocks,
                max_batch_tokens=max(4096, args.cuda_batch_size),
                max_requests_per_batch=args.cuda_batch_size,
                max_blocks_per_request=max(
                    1,
                    (max(map(len, (sample["prompt_ids"] for sample in samples)))
                     + args.max_new_tokens
                     + args.cuda_paged_block_size - 1)
                    // args.cuda_paged_block_size,
                ),
                allow_block_sharing=True,
                use_cuda_graph=False,
                use_async_batching=False,
            )
            print(
                f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} "
                f"paged_continuous batch={args.cuda_batch_size} blocks={args.cuda_paged_num_blocks}",
                flush=True,
            )
            candidate_started = time.perf_counter()
            common_prefix_tokens = _common_prefix_length(samples) if args.cuda_shared_prefix else 0
            shareable_prefix_tokens = (
                common_prefix_tokens // args.cuda_paged_block_size
            ) * args.cuda_paged_block_size
            prefix_snapshot = None
            prefix_started = time.perf_counter()
            if shareable_prefix_tokens:
                from recirculation.cuda_backend import CUDAConcurrentRunner

                prefix_runner = CUDAConcurrentRunner(model, config, use_python_threads=True)
                try:
                    prefix = torch.tensor(
                        [samples[0]["prompt_ids"][:shareable_prefix_tokens]],
                        dtype=torch.long,
                        device=device,
                    )
                    _, prefix_cache, prefix_pending, _ = prefix_runner.prefill(prefix)
                    prefix_snapshot = prefix_runner.snapshot(prefix_cache, prefix_pending)
                finally:
                    prefix_runner.close()
            prefix_seconds = time.perf_counter() - prefix_started
            scheduler_started = time.perf_counter()
            with patch_model_paged_recirculation(model, config, FusedNormMix()):
                manager = model.init_continuous_batching(
                    continuous_batching_config=paged_config,
                )
                manager.start()
                try:
                    while manager.batch_processor is None:
                        if not manager.is_running():
                            raise RuntimeError("paged manager stopped during initialization")
                        time.sleep(0.001)
                    if prefix_snapshot is not None:
                        seeded_blocks = seed_paged_cache_from_snapshot(
                            manager.batch_processor.cache,
                            request_id="recirculation_prefix_seed",
                            prompt_ids=samples[0]["prompt_ids"][:shareable_prefix_tokens],
                            snapshot=prefix_snapshot,
                        )
                        print(
                            f"seeded {seeded_blocks} shared recurrent prefix blocks "
                            f"({shareable_prefix_tokens} tokens)",
                            flush=True,
                        )
                    outputs = {}
                    admission_batch = min(
                        args.cuda_paged_admission_batch or args.cuda_batch_size,
                        args.cuda_batch_size,
                    )
                    next_to_submit = 0
                    outstanding = set()

                    def submit_cohort(
                        manager=manager,
                        admission_batch=admission_batch,
                        outstanding=outstanding,
                    ) -> None:
                        nonlocal next_to_submit
                        stop = min(next_to_submit + admission_batch, len(samples))
                        for index in range(next_to_submit, stop):
                            manager.add_request(
                                samples[index]["prompt_ids"],
                                request_id=f"row_{index}",
                                max_new_tokens=args.max_new_tokens,
                                streaming=False,
                            )
                            outstanding.add(f"row_{index}")
                        if stop > next_to_submit:
                            print(
                                f"paged admission rows {next_to_submit + 1}-{stop}/{len(samples)}",
                                flush=True,
                            )
                        next_to_submit = stop

                    submit_cohort()
                    while len(outputs) < len(samples):
                        output = manager.get_result(timeout=1)
                        if output is not None and output.is_finished():
                            outputs[output.request_id] = output
                            outstanding.discard(output.request_id)
                            if not outstanding and next_to_submit < len(samples):
                                submit_cohort()
                        elif output is not None and output.status == RequestStatus.FAILED:
                            raise RuntimeError(
                                f"paged request {output.request_id} failed: {output.error}"
                            )
                        elif not manager.is_running():
                            raise RuntimeError("paged recirculation manager stopped early")
                finally:
                    manager.stop(block=True)
                    manager.destroy()
            scheduler_seconds = time.perf_counter() - scheduler_started
            if len(outputs) != len(samples):
                raise RuntimeError(
                    f"paged recirculation returned {len(outputs)}/{len(samples)} requests"
                )
            ordered_outputs = [outputs[f"row_{index}"] for index in range(len(samples))]
            generated_tokens = sum(len(output.generated_tokens) for output in ordered_outputs)
            candidate_seconds = time.perf_counter() - candidate_started
            candidate_timings[arm] = {
                "prefix_seconds": prefix_seconds,
                "scheduler_seconds": scheduler_seconds,
                "candidate_seconds": candidate_seconds,
                "generated_tokens": generated_tokens,
                "scheduler_generated_tokens_per_second": generated_tokens / scheduler_seconds,
                "candidate_generated_tokens_per_second": generated_tokens / candidate_seconds,
            }
            for sample, output in zip(samples, ordered_outputs, strict=True):
                sample[arm] = _generation_result(tokenizer, output.generated_tokens, until)
            maybe_status(f"candidate-{candidate_index + 1}", force=True)
            continue
        if device.type == "cuda":
            common_prefix_length = _common_prefix_length(samples) if args.cuda_shared_prefix else 0
            prefix_snapshot = None
            if common_prefix_length:
                prefix = torch.tensor(
                    [samples[0]["prompt_ids"][:common_prefix_length]], dtype=torch.long, device=device
                )
                _, prefix_cache, prefix_pending, _ = candidate_runners[arm].prefill(prefix)
                prefix_snapshot = candidate_runners[arm].snapshot(prefix_cache, prefix_pending)
                print(
                    f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} shared_prefix={common_prefix_length}",
                    flush=True,
                )
            if prefix_snapshot is not None:
                for sample_index, sample in enumerate(samples):
                    suffix = sample["prompt_ids"][common_prefix_length:]
                    if not suffix:
                        raise RuntimeError("shared prefix consumed the entire prompt")
                    sample[arm] = _generate_cuda_from_snapshot(
                        candidate_runners[arm],
                        prefix_snapshot,
                        tokenizer,
                        suffix,
                        device,
                        args.max_new_tokens,
                        until,
                    )
                    maybe_status(f"candidate-{candidate_index}")
                    if (sample_index + 1) % 4 == 0:
                        correct = sum(
                            numbers_equal(s[arm]["numeric_answer"], s["gold_answer"])
                            for s in samples[: sample_index + 1]
                        )
                        print(
                            f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} "
                            f"rows={sample_index + 1} correct={correct}",
                            flush=True,
                        )
                maybe_status(f"candidate-{candidate_index + 1}", force=True)
                continue
            groups = {}
            for sample in samples:
                groups.setdefault(len(sample["prompt_ids"]), []).append(sample)
            completed = 0
            for group in groups.values():
                for start in range(0, len(group), args.cuda_batch_size):
                    batch = group[start : start + args.cuda_batch_size]
                    use_graph = args.cuda_graph_prefill and len(batch) == 1
                    if use_graph:
                        graph_key = (arm, len(batch[0]["prompt_ids"]))
                        if graph_key not in graph_prefills:
                            from recirculation.cuda_backend import CUDAGraphedConcurrentPrefill

                            example = torch.tensor(
                                [batch[0]["prompt_ids"]], dtype=torch.long, device=device
                            )
                            try:
                                graph_prefills[graph_key] = CUDAGraphedConcurrentPrefill(
                                    candidate_runners[arm],
                                    example,
                                    warmups=0,
                                    max_tokens=args.cuda_graph_max_tokens,
                                )
                            except (ValueError, RuntimeError, torch.AcceleratorError) as error:
                                print(f"CUDA graph prefill unavailable for length {example.shape[1]}: {error}", flush=True)
                                graph_prefills[graph_key] = False
                        if graph_prefills[graph_key] is False:
                            use_graph = False
                    if use_graph:
                        result = _generate_cuda_graph_prefill(
                            graph_prefills[graph_key],
                            candidate_runners[arm],
                            tokenizer,
                            batch[0]["prompt_ids"],
                            device,
                            args.max_new_tokens,
                            until,
                        )
                        batch[0][arm] = result
                        completed += 1
                        maybe_status(f"candidate-{candidate_index}")
                        continue
                    results = _generate_cuda_batch(
                        candidate_runners[arm],
                        tokenizer,
                        [sample["prompt_ids"] for sample in batch],
                        device,
                        args.max_new_tokens,
                        until,
                    )
                    for sample, result in zip(batch, results):
                        sample[arm] = result
                        completed += 1
                        maybe_status(f"candidate-{candidate_index}")
                    if completed % 4 == 0 or completed == len(samples):
                        correct = sum(
                            numbers_equal(s[arm]["numeric_answer"], s["gold_answer"])
                            for s in samples
                            if arm in s
                        )
                        print(
                            f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} rows={completed} correct={correct}",
                            flush=True,
                        )
        else:
            for sample_index, sample in enumerate(samples):
                with RecirculationController(model, config) as controller:
                    sample[arm] = _generate(
                        model,
                        tokenizer,
                        sample["prompt_ids"],
                        device,
                        args.max_new_tokens,
                        until,
                        controller=controller,
                    )
                maybe_status(f"candidate-{candidate_index}")
                if (sample_index + 1) % 4 == 0:
                    correct = sum(
                        numbers_equal(sample[arm]["numeric_answer"], sample["gold_answer"])
                        for sample in samples[: sample_index + 1]
                    )
                    print(
                        f"candidate {candidate_index + 1}/{len(args.candidate)} {arm} rows={sample_index + 1} correct={correct}",
                        flush=True,
                    )

        maybe_status(f"candidate-{candidate_index + 1}", force=True)

    summaries = {}
    baseline_summary = _summary(samples, "baseline") if all("baseline" in sample for sample in samples) else None
    for source, destination, alpha in args.candidate:
        arm = f"source{source}_destination{destination}_alpha{alpha:g}"
        summaries[arm] = _summary(samples, arm)
        if baseline_summary is not None:
            summaries[arm]["delta_vs_baseline"] = (
                summaries[arm]["numeric_correct"] - baseline_summary["numeric_correct"]
            )
            summaries[arm]["paired_vs_baseline"] = _paired_candidate_summary(samples, "baseline", arm)
    ranking = []
    if baseline_summary is not None:
        ranking = [
            paired_selection_entry(
                source,
                destination,
                alpha,
                summaries[f"source{source}_destination{destination}_alpha{alpha:g}"],
                harm_weight=args.harm_weight,
                max_correct_to_wrong=args.max_correct_to_wrong,
            )
            for source, destination, alpha in args.candidate
        ]
        ranking.sort(
            key=lambda item: (
                not item["valid"],
                -item["selection_score"],
                item["correct_to_wrong"],
                -item["numeric_correct"],
            )
        )
        for item in ranking:
            screen_item = selected_screen_items.get((item["source_layer"], item["destination_layer"], item["alpha"]))
            if screen_item is not None and "objectives" in screen_item:
                item["proxy_objectives"] = screen_item["objectives"]
    best_flip_penalized = ranking[0] if ranking else None
    best_accuracy = (
        min(
            ranking,
            key=lambda item: (
                not item["valid"],
                -item["numeric_correct"],
                item["correct_to_wrong"],
                -item["wrong_to_correct"],
            ),
        )
        if ranking
        else None
    )
    candidates_with_proxy = [item for item in ranking if "proxy_objectives" in item]
    best_perplexity = (
        min(
            candidates_with_proxy,
            key=lambda item: item["proxy_objectives"].get(
                "language_modeling",
                item["proxy_objectives"].get(
                    "full_solution", item["proxy_objectives"].get("final_answer")
                ),
            )["target_perplexity"],
        )
        if candidates_with_proxy
        else None
    )
    comparison = None
    if len(args.candidate) == 2:
        reference = args.candidate[0]
        candidate = args.candidate[1]
        reference_arm = f"source{reference[0]}_destination{reference[1]}_alpha{reference[2]:g}"
        candidate_arm = f"source{candidate[0]}_destination{candidate[1]}_alpha{candidate[2]:g}"
        comparison = {
            "reference_arm": reference_arm,
            "candidate_arm": candidate_arm,
            "numeric_correct_delta": (
                summaries[candidate_arm]["numeric_correct"] - summaries[reference_arm]["numeric_correct"]
            ),
            "flexible_correct_delta": (
                summaries[candidate_arm]["flexible_correct"] - summaries[reference_arm]["flexible_correct"]
            ),
            "paired": _paired_candidate_summary(samples, reference_arm, candidate_arm),
        }
    report = {
        "implementation_commit": _implementation_commit(),
        "settings": {
            "model": args.model,
            "dataset": args.dataset,
            "row_start": args.row_start,
            "rows": args.rows,
            "device": str(device),
            "attn_implementation": attn_implementation,
            "max_new_tokens": args.max_new_tokens,
            "fewshot_count": len(fewshots),
            "candidates": [list(candidate) for candidate in args.candidate],
            "baseline_skipped": args.skip_baseline,
            "baseline_results": str(args.baseline_results) if args.baseline_results is not None else None,
            "screen_results": str(args.screen_results) if args.screen_results is not None else None,
            "top_k": args.top_k,
            "harm_weight": args.harm_weight,
            "max_correct_to_wrong": args.max_correct_to_wrong,
            "cuda_batch_size": args.cuda_batch_size,
            "cuda_paged_continuous": args.cuda_paged_continuous,
            "cuda_auto_optimized": cuda_auto_optimized,
            "cuda_paged_num_blocks": args.cuda_paged_num_blocks,
            "cuda_paged_block_size": args.cuda_paged_block_size,
            "cuda_paged_admission_batch": (
                args.cuda_paged_admission_batch or args.cuda_batch_size
            ),
            "cuda_graph_prefill": args.cuda_graph_prefill,
            "cuda_graph_max_tokens": args.cuda_graph_max_tokens,
            "cuda_compile_requested": args.cuda_compile,
            "cuda_compile_used": compile_used,
            "cuda_python_threads": args.cuda_python_threads,
            "cuda_static_cache": args.cuda_static_cache,
            "cuda_compile_runner_requested": args.cuda_compile_runner,
            "cuda_compile_runner": cuda_compile_runner,
            "cuda_shared_prefix": args.cuda_shared_prefix,
            "common_prefix_tokens": _common_prefix_length(samples),
        },
        "seconds": time.perf_counter() - started,
        "summaries": summaries,
        "candidate_timings": candidate_timings,
        "ranking": ranking,
        "best": best_flip_penalized,
        "best_flip_penalized": best_flip_penalized,
        "best_accuracy": best_accuracy,
        "best_perplexity": best_perplexity,
        "comparison": comparison,
        "status_checkpoint": str(status_path),
        "samples": [{key: value for key, value in sample.items() if key != "prompt_ids"} for sample in samples],
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    maybe_status("complete", force=True)
    for runner in candidate_runners.values():
        runner.close()
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
