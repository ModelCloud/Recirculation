# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest
from evalution.benchmarks import gsm8k_platinum
from evalution.scorers.gsm8k import INVALID_ANSWER

import scripts.evaluate as evaluation
from scripts.evaluate import (
    REPO_ROOT,
    _build_evalution_suite,
    _candidate_batches,
    _candidate_specs,
    _common_prefix_length,
    _decode_cli_value,
    _gold_answer,
    _instruction,
    _observe_evalution_generation_progress,
    _paired,
    _parse_benchmark_assignment,
    _parse_candidate,
    _parse_suite_assignment,
    _score_recirculation_chunks,
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
        [
            "evaluate.py",
            "paired-gsm8k",
            *coefficient_args,
            "--output",
            str(tmp_path / "result.json"),
        ],
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


def test_generic_suite_arguments_decode_json_and_plain_strings():
    assert _decode_cli_value("true") is True
    assert _decode_cli_value('["stem", "humanities"]') == ["stem", "humanities"]
    assert _decode_cli_value("stem") == "stem"
    assert _parse_suite_assignment("max_rows=32") == ("max_rows", 32)
    assert _parse_benchmark_assignment("mmlu.subsets=stem") == (
        "mmlu",
        "subsets",
        "stem",
    )
    with pytest.raises(evaluation.argparse.ArgumentTypeError, match="KEY=VALUE"):
        _parse_suite_assignment("broken")
    with pytest.raises(evaluation.argparse.ArgumentTypeError, match="BENCHMARK.KEY=VALUE"):
        _parse_benchmark_assignment("broken")


def test_generic_suite_loader_supports_evalution_factories_and_local_arrow(tmp_path):
    mmlu_suite = _build_evalution_suite("mmlu", {"subsets": "stem", "max_rows": 2})
    assert mmlu_suite.subsets == "stem"
    assert mmlu_suite.max_rows == 2

    arrow_path = tmp_path / "gsm8k.arrow"
    arrow_path.write_bytes(b"test placeholder")
    local_suite = _build_evalution_suite(
        "gsm8k_platinum",
        {"dataset_path": str(arrow_path), "dataset_name": None, "max_rows": 1},
    )
    assert local_suite.dataset_loader() is evaluation._load_local_arrow
    with pytest.raises(ValueError, match="unknown Evalution benchmark"):
        _build_evalution_suite("not_a_real_benchmark", {})


def test_generic_suite_loader_supports_named_mmlu_groups():
    stem = _build_evalution_suite("mmlu_stem", {})
    humanities = _build_evalution_suite("mmlu_humanities", {})

    assert stem.task_name() == "mmlu_stem"
    assert humanities.task_name() == "mmlu_humanities"


def test_evalution_chunks_are_shifted_into_cuda_recirculation_targets():
    calls = []

    class ModelConfig:
        pad_token_id = 0
        eos_token_id = 2

    class Runner:
        device = "cpu"
        model = SimpleNamespace(config=ModelConfig())

        @staticmethod
        def score(tokens, targets, *, attention_mask, return_per_row):
            calls.append((tokens.tolist(), targets, attention_mask.tolist(), return_per_row))
            return [1.25, 2.5], [2, 1]

    chunks = [
        SimpleNamespace(
            input_ids=[1, 4, 5, 6],
            score_start=2,
            score_count=2,
            metadata={"row": 0},
        ),
        SimpleNamespace(
            input_ids=[1, 7, 8],
            score_start=2,
            score_count=1,
            metadata={"row": 1},
        ),
    ]
    outputs = _score_recirculation_chunks(Runner(), chunks, batch_size=2)

    assert calls == [
        (
            [[1, 4, 5], [1, 7, 0]],
            {1: ([0, 1], [5, 8]), 2: ([0], [6])},
            [[1, 1, 1], [1, 1, 0]],
            True,
        )
    ]
    assert [(output.logprob, output.token_count, output.metadata) for output in outputs] == [
        (-1.25, 2, {"row": 0}),
        (-2.5, 1, {"row": 1}),
    ]


def test_evalution_generation_progress_exposes_exact_partial_scores(monkeypatch):
    import evalution.benchmarks.base as benchmark_base

    events = []

    class Suite:
        def score_progress_title(self, *, processed, aggregate_scores, invalid_predictions):
            del self
            return f"suite: scoring {processed} {aggregate_scores} {invalid_predictions}"

    monkeypatch.setattr(
        benchmark_base,
        "manual_progress",
        lambda total, *args, **kwargs: (total, args, kwargs),
    )
    suite = Suite()
    with _observe_evalution_generation_progress(
        suite,
        lambda **values: events.append(values),
    ):
        benchmark_base.manual_progress(12, title="suite: scoring")
        suite.score_progress_title(
            processed=3,
            aggregate_scores={"acc,num": 2.0},
            invalid_predictions=1,
        )

    assert events == [
        {"total": 12},
        {
            "processed": 3,
            "aggregate_scores": {"acc,num": 2.0},
            "invalid_predictions": 1,
        },
    ]


def test_generic_evalution_runner_reuses_one_model_for_multiple_suites(monkeypatch, tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "result.json"
    events = []

    class FakeResult:
        @staticmethod
        def to_dict():
            return {
                "model": {},
                "engine": {},
                "tests": [
                    {"name": "gsm8k_platinum", "samples": [{"id": 1}]},
                    {"name": "mmlu_stem", "samples": [{"id": 2}, {"id": 3}]},
                ],
            }

    class FakeEvaluation:
        def run(self, suite):
            events.append(("run", suite.task_name()))
            return self

        @staticmethod
        def result():
            return FakeResult()

        def close(self):
            events.append(("close", None))

    class FakeTransformers:
        def __init__(self, **kwargs):
            events.append(("engine", kwargs))

        def model(self, **kwargs):
            events.append(("model", kwargs))
            return FakeEvaluation()

    monkeypatch.setattr(evaluation, "Transformers", FakeTransformers)
    monkeypatch.setattr(evaluation, "_git_commit", lambda: "abc123")

    assert (
        evaluation.main(
            [
                "run",
                "--model",
                str(model_path),
                "--device",
                "cpu",
                "--benchmark",
                "gsm8k_platinum",
                "--benchmark",
                "mmlu",
                "--max-rows",
                "2",
                "--benchmark-arg",
                "gsm8k_platinum.variant=cot_llama",
                "--benchmark-arg",
                "mmlu.subsets=stem",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    report = evaluation.json.loads(output_path.read_text(encoding="utf-8"))
    assert report["provenance"]["benchmarks"] == ["gsm8k_platinum", "mmlu"]
    assert report["provenance"]["rows"] == 3
    assert report["provenance"]["git_commit"] == "abc123"
    assert [event for event in events if event[0] == "run"] == [
        ("run", "gsm8k_platinum_cot_llama"),
        ("run", "mmlu_stem"),
    ]
    assert len([event for event in events if event[0] == "model"]) == 1


def test_generic_runner_lists_installed_benchmarks_without_loading_model(capsys):
    assert evaluation.main(["run", "--list-benchmarks"]) == 0
    listed = capsys.readouterr().out.splitlines()
    assert "gsm8k_platinum" in listed
    assert "mmlu" in listed
