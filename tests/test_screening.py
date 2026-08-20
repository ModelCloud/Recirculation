# SPDX-License-Identifier: Apache-2.0

import pytest

from recirculation.screening import (
    gsm8k_solution_target,
    paired_selection_entry,
    screen_result_key,
    summarize_paired_losses,
)


def test_gsm8k_solution_target_keeps_reasoning_and_removes_calculator_annotations():
    answer = "She has 2+3 = <<2+3=5>>5 books.\nThen stops.\n#### 5"
    assert gsm8k_solution_target(answer, "5") == ("She has 2+3 = 5 books.\nThen stops.\nThe final answer is 5")


def test_paired_loss_summary_penalizes_tail_regressions():
    summary = summarize_paired_losses(
        [10, 11, 12, 13],
        [2.0, 2.0, 2.0, 2.0],
        [2, 2, 2, 2],
        [1.0, 1.0, 1.0, 5.0],
        [2, 2, 2, 2],
        tail_quantile=0.75,
        tail_weight=2.0,
    )
    assert summary["native_delta_nll"] == pytest.approx(0.0)
    assert summary["tail_harm_nll"] == pytest.approx(1.5)
    assert summary["screen_score"] == pytest.approx(3.0)
    assert summary["improved_rows"] == 3
    assert summary["regressed_rows"] == 1
    assert summary["neutral_rows"] == 0
    assert screen_result_key(summary)[0] == pytest.approx(3.0)


def test_paired_loss_summary_rejects_mismatched_target_counts():
    with pytest.raises(ValueError, match="target counts"):
        summarize_paired_losses([1], [1.0], [2], [1.0], [1])


def test_e2e_selection_penalizes_correct_to_wrong_more_than_wrong_to_correct():
    summary = {
        "numeric_correct": 42,
        "numeric_accuracy": 0.5,
        "paired_vs_baseline": {
            "numeric": {
                "wrong_to_correct": 8,
                "correct_to_wrong": 6,
                "net_correct": 2,
            }
        },
    }
    entry = paired_selection_entry(8, 2, 0.2, summary, harm_weight=2.0, max_correct_to_wrong=5)
    assert entry["selection_score"] == -4
    assert not entry["valid"]
