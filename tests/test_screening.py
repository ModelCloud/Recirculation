# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from recirculation.screening import (
    DEFAULT_PATH_ALPHA,
    gsm8k_solution_target,
    objective_result_key,
    paired_selection_entry,
    perplexity_result_key,
    proxy_shortlist,
    render_screen_report_markdown,
    screen_leaders,
    screen_result_key,
    summarize_paired_losses,
)
from scripts.screen_cuda_recirculation import MODEL_DTYPES, _candidate_schedule, _ordered_candidates


def test_path_search_starts_at_conservative_alpha():
    assert DEFAULT_PATH_ALPHA == 0.05


def test_cuda_screening_exposes_fp16_and_bf16_model_dtypes():
    assert MODEL_DTYPES == {"float16": torch.float16, "bfloat16": torch.bfloat16}


def test_cuda_screening_randomizes_and_persists_a_reproducible_candidate_schedule():
    paths = [(source, destination) for destination in range(4) for source in range(destination + 1, 4)]
    sequential = _ordered_candidates(paths, [0.05, 0.1], scan_order="sequential", scan_seed=7)
    randomized = _ordered_candidates(paths, [0.05, 0.1], scan_order="random", scan_seed=7)

    assert randomized != sequential
    assert randomized == _ordered_candidates(paths, [0.05, 0.1], scan_order="random", scan_seed=7)
    assert set(randomized) == set(sequential)
    schedule = _candidate_schedule(randomized)
    assert [item["scan_index"] for item in schedule] == list(range(len(randomized)))
    assert [
        (item["source_layer"], item["destination_layer"], item["alpha"]) for item in schedule
    ] == randomized


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


def test_perplexity_key_ignores_robust_harm_penalty():
    low_ppl_harmed = {"target_perplexity": 2.0, "target_nll": 0.7, "screen_score": 1.0, "alpha": 0.1}
    robust = {"target_perplexity": 2.1, "target_nll": 0.74, "screen_score": -0.1, "alpha": 0.1}
    assert min([low_ppl_harmed, robust], key=perplexity_result_key) is low_ppl_harmed
    assert min([low_ppl_harmed, robust], key=screen_result_key) is robust


def test_dual_objective_shortlist_unions_plain_and_robust_leaders():
    def candidate(source, final_ppl, final_score, full_ppl, full_score):
        return {
            "source_layer": source,
            "destination_layer": 0,
            "alpha": 0.1,
            "objectives": {
                "final_answer": {
                    "target_perplexity": final_ppl,
                    "target_nll": final_ppl,
                    "screen_score": final_score,
                },
                "full_solution": {
                    "target_perplexity": full_ppl,
                    "target_nll": full_ppl,
                    "screen_score": full_score,
                },
            },
        }

    results = [
        candidate(1, 1.0, 9.0, 9.0, 9.0),
        candidate(2, 9.0, 1.0, 9.0, 9.0),
        candidate(3, 9.0, 9.0, 1.0, 9.0),
        candidate(4, 9.0, 9.0, 9.0, 1.0),
    ]
    assert objective_result_key(results[0], "final_answer", robust=False)[0] == 1.0
    assert [item["source_layer"] for item in proxy_shortlist(results, 4)] == [1, 2, 3, 4]


def test_promotion_recalculates_against_full_population_and_keeps_early_winner():
    def candidate(source, final_ppl, final_score, full_ppl, full_score):
        return {
            "source_layer": source,
            "destination_layer": 0,
            "alpha": 0.1,
            "seconds": float(source),
            "objectives": {
                "final_answer": {
                    "target_perplexity": final_ppl,
                    "target_nll": final_ppl,
                    "native_delta_nll": final_ppl,
                    "screen_score": final_score,
                    "tail_harm_nll": 0.0,
                    "improved_rows": 1,
                    "regressed_rows": 0,
                    "neutral_rows": 0,
                },
                "full_solution": {
                    "target_perplexity": full_ppl,
                    "target_nll": full_ppl,
                    "native_delta_nll": full_ppl,
                    "screen_score": full_score,
                    "tail_harm_nll": 0.0,
                    "improved_rows": 1,
                    "regressed_rows": 0,
                    "neutral_rows": 0,
                },
            },
        }

    early_winner = candidate(1, 1.0, -1.0, 1.0, -1.0)
    completed = [early_winner, candidate(2, 2.0, 2.0, 2.0, 2.0), candidate(3, 3.0, 3.0, 3.0, 3.0)]
    leaders = screen_leaders(completed)
    assert all(leader is early_winner for leader in leaders.values())
    assert proxy_shortlist(completed, 1) == [early_winner]

    report = {
        "status": "running",
        "complete": 3,
        "active": 1,
        "pending": 1,
        "implementation_commit": "abc123",
        "leaders": leaders,
        "results": completed,
    }
    markdown = render_screen_report_markdown(report)
    assert "all 3 completed candidates" in markdown
    assert "1→0" in markdown
    assert "2→0" in markdown
    assert "3→0" in markdown


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


def test_e2e_selection_rejects_losing_candidate_even_without_harm_cap():
    summary = {
        "numeric_correct": 16,
        "numeric_accuracy": 0.5,
        "paired_vs_baseline": {
            "numeric": {
                "wrong_to_correct": 1,
                "correct_to_wrong": 2,
                "net_correct": -1,
            }
        },
    }
    entry = paired_selection_entry(4, 2, 0.2, summary, harm_weight=2.0, max_correct_to_wrong=None)
    assert entry["selection_score"] == -3
    assert not entry["valid"]


def test_e2e_selection_accepts_positive_natural_generation_gain():
    summary = {
        "numeric_correct": 20,
        "numeric_accuracy": 0.625,
        "paired_vs_baseline": {
            "numeric": {
                "wrong_to_correct": 5,
                "correct_to_wrong": 1,
                "net_correct": 4,
            }
        },
    }
    entry = paired_selection_entry(8, 2, 0.2, summary, harm_weight=2.0, max_correct_to_wrong=None)
    assert entry["selection_score"] == 3
    assert entry["valid"]
