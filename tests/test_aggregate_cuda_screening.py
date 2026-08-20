# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from scripts.aggregate_cuda_screening import aggregate_reports


def _candidate(source, score):
    summary = {
        "target_perplexity": score + 2.0,
        "target_nll": score + 1.0,
        "native_delta_nll": score,
        "screen_score": score,
        "tail_harm_nll": 0.0,
        "improved_rows": 1,
        "regressed_rows": 0,
        "neutral_rows": 0,
    }
    return {
        "source_layer": source,
        "destination_layer": 0,
        "alpha": 0.1,
        "seconds": 1.0,
        "objectives": {"final_answer": dict(summary), "full_solution": dict(summary)},
    }


def _report(results, *, model="model"):
    complete = len(results)
    return {
        "status": "running",
        "complete": complete,
        "active": 1,
        "pending": 2 - complete,
        "total": 3,
        "implementation_commit": "abc123",
        "settings": {
            "scoring_schema": "dual_objective_v1",
            "model": model,
            "dataset": "dataset",
            "row_start": 0,
            "rows": 1,
            "row_stop_exclusive": 1,
            "common_prefix_tokens": 10,
            "target_mode": "dual",
            "tail_quantile": 0.9,
            "tail_weight": 1.0,
            "harm_tolerance": 0.0,
            "shared_native_baseline": "baseline.json",
        },
        "results": results,
    }


def test_aggregate_promotes_from_all_shards_not_only_latest_or_one_shard(tmp_path):
    early_global_winner = _candidate(1, -1.0)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_report([early_global_winner])), encoding="utf-8")
    second.write_text(json.dumps(_report([_candidate(2, 2.0), _candidate(3, 3.0)])), encoding="utf-8")

    aggregate = aggregate_reports([first, second])

    assert aggregate["complete"] == 3
    assert len(aggregate["results"]) == 3
    assert all(leader["source_layer"] == 1 for leader in aggregate["leaders"].values())
    assert aggregate["shortlist"][0]["source_layer"] == 1
    assert aggregate["settings"]["aggregation"]["promotion_scope"].startswith("all completed")


def test_aggregate_rejects_incompatible_shards(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(_report([_candidate(1, 1.0)])), encoding="utf-8")
    second.write_text(json.dumps(_report([_candidate(2, 2.0)], model="different")), encoding="utf-8")

    with pytest.raises(ValueError, match="settings differ"):
        aggregate_reports([first, second])
