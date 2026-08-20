# SPDX-License-Identifier: Apache-2.0

from evalution.benchmarks import gsm8k_platinum
from evalution.scorers.gsm8k import INVALID_ANSWER

import scripts.eval_gsm8k_platinum as evaluation
from scripts.eval_gsm8k_platinum import (
    REPO_ROOT,
    _candidate_batches,
    _candidate_specs,
    _gold_answer,
    _instruction,
    _paired,
    _summary,
    _task_contract,
)


def _arm(numeric_answer: str):
    return {
        "numeric_answer": numeric_answer,
        "strict_answer": numeric_answer if numeric_answer != INVALID_ANSWER else None,
        "flexible_answer": numeric_answer if numeric_answer != INVALID_ANSWER else None,
    }


def test_evalution_numeric_scoring_is_the_primary_paired_metric():
    samples = [
        {
            "gold_answer": "1200",
            "baseline": _arm("1199"),
            "recirculated": _arm("1200"),
        },
        {
            "gold_answer": "0.5",
            "baseline": _arm("0.50"),
            "recirculated": _arm(INVALID_ANSWER),
        },
    ]

    baseline = _summary(samples, "baseline")
    recirculated = _summary(samples, "recirculated")
    paired = _paired(samples)["numeric"]

    assert baseline["numeric_correct"] == 1
    assert recirculated["numeric_correct"] == 1
    assert recirculated["numeric_invalid"] == 1
    assert paired == {
        "answer_changes": 2,
        "wrong_to_correct": 1,
        "correct_to_wrong": 1,
        "net_correct": 0,
    }


def test_paired_scoring_accepts_named_multi_candidate_arms():
    samples = [
        {
            "gold_answer": "42",
            "baseline": _arm("41"),
            "source8_destination2_alpha0.1": _arm("42"),
        }
    ]

    assert _paired(samples, "source8_destination2_alpha0.1")["numeric"]["net_correct"] == 1


def test_candidate_matrix_batches_same_destination_paths_together():
    specs = _candidate_specs([(8, 2), (4, 2)], [0.1], None, 0)

    assert [arm for arm, _ in specs] == [
        "source8_destination2_alpha0.1",
        "source4_destination2_alpha0.1",
    ]
    batches = _candidate_batches(specs, batch_size=8)
    assert len(batches) == 1
    assert [config.source_layer for _, config in batches[0]] == [8, 4]


def test_backend_auto_prefers_cuda_then_mlx(monkeypatch):
    monkeypatch.setattr(evaluation, "_cuda_accelerator_available", lambda: True)
    monkeypatch.setattr(evaluation, "_mlx_accelerator_available", lambda: True)
    assert evaluation._resolve_backend("auto", "auto") == ("cuda", "cuda")

    monkeypatch.setattr(evaluation, "_cuda_accelerator_available", lambda: False)
    assert evaluation._resolve_backend("auto", "auto") == ("mlx", "metal")
    assert evaluation._resolve_backend("auto", "mps") == ("torch", "mps")


def test_gold_answer_uses_evalution_numeric_canonicalization():
    assert _gold_answer("Reasoning\n#### 1,200.00") == "1200"


def test_local_prompt_contract_matches_evalution_cot_llama():
    suite = gsm8k_platinum(variant="cot_llama", apply_chat_template=True)
    _, spec = suite._resolved_variant()
    fewshots, until = _task_contract(REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")

    assert fewshots == [(sample["question"], sample["target"]) for sample in spec.fewshots]
    assert until == spec.stop_strings
    assert _instruction("What is 2 + 2?") == spec.prompt_builder({"question": "What is 2 + 2?"})
