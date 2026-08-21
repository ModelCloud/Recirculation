#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Gate Transformers paged recirculation against the CUDA Torch runner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tokenicer import Tokenicer
from transformers import AutoModelForCausalLM, ContinuousBatchingConfig

from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAConcurrentRunner, FusedNormMix, measure_forward_error
from recirculation.transformers_paged_patch import patch_model_paged_recirculation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/local-models/Llama-3.2-1B-Instruct")
    parser.add_argument("--path", default="10:1")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--error-limit", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, destination = map(int, args.path.split(":"))
    config = RecirculationConfig(
        source_layer=source,
        destination_layer=destination,
        alpha=args.alpha,
    )

    tokenizer = Tokenicer.load(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
    ).eval().to("cuda")
    prompt = tokenizer(
        "Explain why recurrent state can improve transformer inference. " * 32,
        return_tensors="pt",
    ).input_ids[:, : args.prompt_tokens].to("cuda")

    reference_runner = CUDAConcurrentRunner(model, config, use_python_threads=True)
    try:
        reference_logits, _cache, _pending, _collected = reference_runner.prefill(prompt)
        reference_token = int(reference_logits[:, -1].argmax(-1).item())
    finally:
        reference_runner.close()

    batching = ContinuousBatchingConfig(
        block_size=256,
        num_blocks=64,
        max_batch_tokens=4096,
        max_requests_per_batch=1,
        max_blocks_per_request=8,
        allow_block_sharing=True,
        use_cuda_graph=False,
        use_async_batching=False,
    )
    with patch_model_paged_recirculation(model, config, FusedNormMix()) as implementation:
        manager = model.init_continuous_batching(continuous_batching_config=batching)
        manager.start()
        try:
            manager.add_request(
                prompt[0].tolist(),
                request_id="numerical_gate",
                max_new_tokens=1,
                streaming=False,
            )
            output = None
            deadline = time.monotonic() + 120
            while output is None and time.monotonic() < deadline:
                candidate = manager.get_result(timeout=1)
                if candidate is not None and candidate.is_finished():
                    output = candidate
                elif not manager.is_running():
                    raise RuntimeError("paged manager stopped before returning the numerical gate")
            if output is None:
                raise TimeoutError("paged manager did not finish the numerical gate")
            candidate_logits = implementation.last_logits
            if candidate_logits is None:
                raise RuntimeError("paged forward did not retain numerical-gate logits")
            candidate_token = int(output.generated_tokens[0])
        finally:
            manager.stop(block=True)
            manager.destroy()

    error = measure_forward_error(reference_logits[:, -1:], candidate_logits[:, -1:])
    error.require(limit=args.error_limit)
    if candidate_token != reference_token:
        raise RuntimeError(
            f"greedy token mismatch: reference={reference_token}, paged={candidate_token}"
        )
    report = {
        "model": args.model,
        "dtype": "float16",
        "source_layer": source,
        "destination_layer": destination,
        "alpha": args.alpha,
        "prompt_tokens": int(prompt.shape[1]),
        "error_limit": args.error_limit,
        "forward_error": {
            "max_absolute": error.max_absolute,
            "mean_absolute": error.mean_absolute,
            "relative_l2": error.relative_l2,
            "normalized_max": error.normalized_max,
            "rate": error.rate,
        },
        "reference_token": reference_token,
        "paged_token": candidate_token,
        "exact_greedy_token_parity": candidate_token == reference_token,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
