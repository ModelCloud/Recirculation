# SPDX-License-Identifier: Apache-2.0

from evalution.benchmarks import gsm8k_platinum
from evalution.scorers.gsm8k import INVALID_ANSWER

from scripts.eval_gsm8k_platinum import REPO_ROOT, _gold_answer, _instruction, _paired, _summary, _task_contract


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


def test_gold_answer_uses_evalution_numeric_canonicalization():
    assert _gold_answer("Reasoning\n#### 1,200.00") == "1200"


def test_local_prompt_contract_matches_evalution_cot_llama():
    suite = gsm8k_platinum(variant="cot_llama", apply_chat_template=True)
    _, spec = suite._resolved_variant()
    fewshots, until = _task_contract(REPO_ROOT / "configs/gsm8k-platinum-cot-llama.yaml")

    assert fewshots == [(sample["question"], sample["target"]) for sample in spec.fewshots]
    assert until == spec.stop_strings
    assert _instruction("What is 2 + 2?") == spec.prompt_builder({"question": "What is 2 + 2?"})
