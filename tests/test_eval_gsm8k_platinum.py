# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
from evalution.benchmarks import gsm8k_platinum
from evalution.scorers.gsm8k import INVALID_ANSWER

import scripts.eval_gsm8k_platinum as evaluation
from scripts.eval_gsm8k_platinum import (
    REPO_ROOT,
    _candidate_batches,
    _candidate_specs,
    _common_prefix_length,
    _gold_answer,
    _instruction,
    _paired,
    _parse_candidate,
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


def test_common_prefix_requires_exact_tokens_and_reserves_a_suffix():
    prompts = ([1, 2, 3, 4], [1, 2, 3, 7], [1, 2, 3, 9, 10])

    assert _common_prefix_length(prompts) == 3
    assert _common_prefix_length(prompts, minimum_suffix_tokens=1) == 3
    assert _common_prefix_length(([1, 2, 3],), minimum_suffix_tokens=1) == 2
    assert _common_prefix_length([], minimum_suffix_tokens=1) == 0
    with pytest.raises(ValueError, match="must be non-negative"):
        _common_prefix_length(prompts, minimum_suffix_tokens=-1)


def test_mlx_baseline_prefix_snapshot_matches_full_prompt_generation():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(31)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=2,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    mx.eval(model.parameters())

    class Tokenizer:
        eos_token_id = 31

        @staticmethod
        def decode(tokens, skip_special_tokens=True):
            del skip_special_tokens
            return " ".join(str(token) for token in tokens)

    prompt = [1, 2, 3, 4]
    expected = evaluation._generate_mlx_baseline(model, Tokenizer(), prompt, 4, ())
    snapshot = evaluation._snapshot_mlx_baseline_prefix(model, prompt[:2])
    candidate = evaluation._generate_mlx_baseline(
        model,
        Tokenizer(),
        prompt[2:],
        4,
        (),
        prefix_snapshot=snapshot,
    )

    assert candidate == expected


def test_mlx_candidate_generation_dispatches_with_and_without_snapshot():
    calls = []
    runner = SimpleNamespace(
        generate=lambda prompt, **kwargs: calls.append(("full", prompt, kwargs)) or [[1]],
        generate_from_snapshot=lambda prompt, snapshot, **kwargs: calls.append(
            ("snapshot", prompt, snapshot, kwargs)
        )
        or [[2]],
    )

    assert evaluation._generate_mlx_candidates(runner, [3], None, 4, 5) == [[1]]
    assert evaluation._generate_mlx_candidates(runner, [7], "state", 8, 9) == [[2]]
    assert calls == [
        ("full", [3], {"max_new_tokens": 4, "eos_token_id": 5}),
        ("snapshot", [7], "state", {"max_new_tokens": 8, "eos_token_id": 9}),
    ]


def test_explicit_candidates_preserve_path_specific_alphas():
    candidates = [_parse_candidate("8:2:0.2"), _parse_candidate("4:2:0.2")]
    specs = _candidate_specs([], [], None, 0, candidates=candidates)

    assert [arm for arm, _ in specs] == [
        "source8_destination2_alpha0.2",
        "source4_destination2_alpha0.2",
    ]
    assert [(config.alpha, config.beta) for _, config in specs] == [(0.2, 0.8), (0.2, 0.8)]
    assert len(_candidate_batches(specs, batch_size=8)) == 1


@pytest.mark.parametrize(
    "coefficient_args",
    (("--candidate", "8:2:-0.2"), ("--path", "8:2", "--alpha", "-0.2")),
)
def test_cli_rejects_negative_alpha_before_backend_initialization(monkeypatch, tmp_path, capsys, coefficient_args):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_gsm8k_platinum.py", *coefficient_args, "--output", str(tmp_path / "result.json")],
    )
    monkeypatch.setattr(
        evaluation,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail("backend initialization must not run for an invalid alpha"),
    )

    with pytest.raises(SystemExit) as error:
        evaluation.main()

    assert error.value.code == 2
    assert "Recirculation alpha must be finite and in [0, 1]" in capsys.readouterr().err


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
