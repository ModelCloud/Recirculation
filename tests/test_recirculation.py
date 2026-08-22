# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from recirculation import RecirculationConfig, RecirculationController, torch_mix_reference


@pytest.mark.parametrize(
    ("source_layer", "destination_layer"),
    ((-1, -2), (2, -1)),
)
def test_recirculation_config_rejects_negative_layer_indices(source_layer, destination_layer):
    with pytest.raises(ValueError, match="non-negative"):
        RecirculationConfig(source_layer=source_layer, destination_layer=destination_layer)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    (
        (-0.2, None),
        (1.01, None),
        (float("nan"), None),
        (float("inf"), None),
        (0.1, -0.01),
        (0.1, 1.01),
        (0.1, float("nan")),
    ),
)
def test_recirculation_config_rejects_out_of_range_mixture_coefficients(alpha, beta):
    with pytest.raises(ValueError, match="must be finite and in"):
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=alpha, beta=beta)


@pytest.mark.parametrize(
    ("beta", "normalize_source", "expected"),
    (
        (0.9, True, [[[2.7, 4.1]]]),
        (1.0, True, [[[3.0, 4.5]]]),
        (0.9, False, [[[2.7, 3.8]]]),
    ),
)
def test_torch_reference_matches_published_mixture_equations(beta, normalize_source, expected):
    destination = torch.tensor([[[3.0, 4.0]]])
    source = torch.tensor([[[0.0, 2.0]]])

    mixed = torch_mix_reference(destination, source, 0.1, beta, normalize_source)

    torch.testing.assert_close(mixed, torch.tensor(expected), rtol=0, atol=3e-7)


def test_torch_reference_zero_norm_policy_preserves_paper_equation_where_defined():
    source = torch.tensor([[[1.0, -1.0]]])
    zero_destination = torch.zeros_like(source)
    destination = torch.tensor([[[3.0, 4.0]]])
    zero_source = torch.zeros_like(destination)

    torch.testing.assert_close(
        torch_mix_reference(zero_destination, source, 0.1, 0.9, True),
        zero_destination,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch_mix_reference(destination, zero_source, 0.1, 0.9, True),
        0.9 * destination,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_mixture_passes_128_step_torch_accumulation_gate():
    from recirculation.cuda_backend import FusedNormMix, measure_forward_error

    torch.manual_seed(7)
    source_reference = torch.randn(4, 1, 2048, device="cuda", dtype=torch.float16)
    source_candidate = source_reference.clone()
    reference = torch.randn_like(source_reference)
    candidate = reference.clone()
    fused = FusedNormMix()
    reference_trace = []
    candidate_trace = []
    for _ in range(128):
        reference = torch_mix_reference(reference, source_reference, 0.1, 0.9, True)
        candidate = fused(candidate, source_candidate, 0.1, 0.9, True)
        reference_trace.append(reference)
        candidate_trace.append(candidate)
        source_reference = 0.997 * source_reference + 0.003 * reference
        source_candidate = 0.997 * source_candidate + 0.003 * candidate
    error = measure_forward_error(torch.cat(reference_trace, dim=1), torch.cat(candidate_trace, dim=1))
    error.require()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("destination_values", "source_values"),
    (
        ((0.0, 0.0), (1.0, -1.0)),
        ((1.0, -1.0), (0.0, 0.0)),
        ((1.0, -1.0), (1e-10, -1e-10)),
    ),
)
def test_cuda_fused_mixture_matches_torch_zero_and_tiny_norm_boundaries(
    destination_values, source_values
):
    from recirculation.cuda_backend import FusedNormMix, measure_forward_error

    destination = torch.tensor([[destination_values]], device="cuda", dtype=torch.float32)
    source = torch.tensor([[source_values]], device="cuda", dtype=torch.float32)
    reference = torch_mix_reference(destination, source, 0.1, 0.9, True)
    candidate = FusedNormMix()(destination, source, 0.1, 0.9, True)
    error = measure_forward_error(reference, candidate)
    error.require(limit=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_ramp_matches_published_zero_based_schedule():
    from recirculation.controller import _mixing_coefficients
    from recirculation.cuda_backend import FusedNormMix

    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        normalize_source=False,
        ramp_tokens=10,
    )
    destination = torch.tensor([[[1.0, 0.0]]], device="cuda")
    source = torch.tensor([[[0.0, 1.0]]], device="cuda")
    fused = FusedNormMix()

    for position, expected_alpha in ((0, 0.0), (1, 0.02), (9, 0.18), (10, 0.2), (11, 0.2)):
        alpha, beta = _mixing_coefficients(config, position)
        candidate = fused(destination, source, alpha, beta, False)
        expected = torch.tensor([[[1.0 - expected_alpha, expected_alpha]]], device="cuda")
        torch.testing.assert_close(candidate, expected, rtol=1e-6, atol=1e-6)


def test_cuda_graph_rejects_ramping_that_exceeds_changed_input_gate():
    pytest.importorskip("triton")
    from recirculation.cuda_backend import CUDAGraphedPrefill

    runner = SimpleNamespace(
        controller=SimpleNamespace(config=RecirculationConfig(source_layer=2, destination_layer=0, ramp_tokens=10))
    )
    with pytest.raises(ValueError, match="changed-input error exceeds the release gate"):
        CUDAGraphedPrefill(runner, torch.ones(1, 1, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_concurrent_stacks_match_sequential_scheduler(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAPrefillRunner, measure_forward_error

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: True)

    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1, ramp_tokens=2)
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    reference_runner = CUDAPrefillRunner(model, config, fused=False)
    reference = reference_runner.prefill(tokens, collect_logits=True)
    concurrent = CUDAConcurrentRunner(model, config)
    try:
        candidate = concurrent.prefill(tokens, collect_logits=True)
    finally:
        concurrent.close()

    measure_forward_error(reference[3], candidate[3]).require()
    measure_forward_error(reference[2].destination, candidate[2].destination).require()
    measure_forward_error(reference[2].source, candidate[2].source).require()
    reference_snapshot = reference_runner.snapshot(reference[1], reference[2])
    candidate_snapshot = concurrent.snapshot(candidate[1], candidate[2])
    for reference_layer, candidate_layer in zip(reference_snapshot.cache_data, candidate_snapshot.cache_data):
        for reference_value, candidate_value in zip(reference_layer, candidate_layer):
            if torch.is_tensor(reference_value):
                measure_forward_error(reference_value, candidate_value).require()
            else:
                assert reference_value == candidate_value
    assert candidate[2].token_position == reference[2].token_position == 3
    assert concurrent.lower_stream.priority == -3
    assert concurrent.replay_stream.priority == -3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen3_torch_and_cuda_recirculation_match(monkeypatch):
    """Qwen3's decoder contract must produce the same serial and two-stream states."""

    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAPrefillRunner, measure_forward_error

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: True)
    torch.manual_seed(20260821)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=64,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05)
    tokens = torch.tensor([[1, 7, 11, 13]], device="cuda")
    expected_tokens = RecirculationController(model, config).generate(tokens, max_new_tokens=2)
    sequential = CUDAPrefillRunner(model, config, fused=False).prefill(tokens, collect_logits=True)
    concurrent = CUDAConcurrentRunner(model, config, use_python_threads=False)
    try:
        candidate = concurrent.prefill(tokens, collect_logits=True)
        candidate_tokens = concurrent.generate(tokens, max_new_tokens=2)
    finally:
        concurrent.close()

    measure_forward_error(sequential[3], candidate[3]).require()
    measure_forward_error(sequential[2].destination, candidate[2].destination).require()
    measure_forward_error(sequential[2].source, candidate[2].source).require()
    torch.testing.assert_close(candidate_tokens, expected_tokens, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen3_no_bos_batched_scoring_matches_scalar_scoring():
    """Corpus scoring must start from text token zero when Qwen3 defines no BOS."""

    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAPrefillRunner

    torch.manual_seed(7)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                bos_token_id=None,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05)
    scalar = CUDAPrefillRunner(model, config)
    row0 = scalar.prefill(torch.tensor([[1, 2]], device="cuda"), collect_logits=True)[3]
    row1 = scalar.prefill(torch.tensor([[4]], device="cuda"), collect_logits=True)[3]
    expected = torch.stack(
        (
            torch.logsumexp(row0[0, 0].float(), dim=-1) - row0[0, 0, 2].float(),
            torch.logsumexp(row0[0, 1].float(), dim=-1) - row0[0, 1, 3].float(),
            torch.logsumexp(row1[0, 0].float(), dim=-1) - row1[0, 0, 5].float(),
        )
    )
    batched = CUDAPrefillRunner(model, config, allow_terminal_padding=True)
    row_nll, row_counts = batched.score(
        torch.tensor([[1, 2], [4, 0]], device="cuda"),
        {0: ([0, 1], [2, 5]), 1: ([0], [3])},
        attention_mask=torch.tensor([[1, 1], [1, 0]], device="cuda"),
        return_per_row=True,
    )

    torch.testing.assert_close(torch.tensor(row_nll), torch.tensor([expected[:2].sum(), expected[2]]), rtol=2e-3, atol=2e-3)
    assert row_counts == [2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen3_concurrent_chunked_score_matches_sequential_gate(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAPrefillRunner, measure_forward_error

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: True)
    torch.manual_seed(11)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                bos_token_id=None,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05)
    tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], device="cuda")
    mask = torch.ones_like(tokens)
    targets = {
        position: ([0, 1], [int(tokens[0, position] + 1), int(tokens[1, position] + 1)])
        for position in range(tokens.shape[1])
    }
    sequential = CUDAPrefillRunner(
        model,
        config,
        allow_terminal_padding=True,
        projection_chunk_tokens=1,
        mask_free_unpadded=True,
    )
    expected, expected_counts = sequential.score(tokens, targets, attention_mask=mask, return_per_row=True)
    concurrent = CUDAConcurrentRunner(model, config, use_python_threads=False)
    try:
        candidate, candidate_counts = concurrent.score(
            tokens,
            targets,
            attention_mask=mask,
            return_per_row=True,
            projection_chunk_tokens=4,
        )
    finally:
        concurrent.close()

    error = measure_forward_error(torch.tensor(expected, device="cuda"), torch.tensor(candidate, device="cuda"))
    error.require()
    assert candidate_counts == expected_counts == [4, 4]

    dual = CUDAConcurrentRunner(model, config, use_python_threads=False, dual_gemm=True)
    try:
        dual_candidate, dual_counts = dual.score(
            tokens,
            targets,
            attention_mask=mask,
            return_per_row=True,
            projection_chunk_tokens=4,
        )
    finally:
        dual.close()
    dual_error = measure_forward_error(
        torch.tensor(expected, device="cuda"), torch.tensor(dual_candidate, device="cuda")
    )
    dual_error.require()
    assert dual_counts == expected_counts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_qwen3_candidate_batch_matches_individual_dual_gemm():
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDABatchedPathRunner, CUDAConcurrentRunner, measure_forward_error

    torch.manual_seed(17)
    model = (
        transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                bos_token_id=None,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    configs = (
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.05),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], device="cuda")
    mask = torch.ones_like(tokens)
    targets = {
        position: ([0, 1], [int(tokens[0, position] + 1), int(tokens[1, position] + 1)])
        for position in range(tokens.shape[1])
    }
    expected = []
    for config in configs:
        runner = CUDAConcurrentRunner(model, config, use_python_threads=False, dual_gemm=True)
        try:
            nll, counts = runner.score(
                tokens,
                targets,
                attention_mask=mask,
                return_per_row=True,
                projection_chunk_tokens=4,
            )
        finally:
            runner.close()
        expected.append(nll)
        assert counts == [4, 4]

    candidate, candidate_counts = CUDABatchedPathRunner(model, configs).score(
        tokens,
        targets,
        attention_mask=mask,
        projection_chunk_tokens=4,
    )
    measure_forward_error(
        torch.tensor(expected, device="cuda"), torch.tensor(candidate, device="cuda")
    ).require()
    assert candidate_counts == [[4, 4], [4, 4], [4, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_llama_candidate_batch_matches_individual_dual_gemm():
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDABatchedPathRunner, CUDAPrefillRunner, measure_forward_error

    torch.manual_seed(29)
    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    configs = (
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.05),
    )
    tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 10]], device="cuda")
    mask = torch.ones_like(tokens)
    targets = {
        position: ([0, 1], [int(tokens[0, position] + 1), int(tokens[1, position] + 1)])
        for position in range(tokens.shape[1])
    }
    expected = []
    for config in configs:
        runner = CUDAPrefillRunner(
            model,
            config,
            allow_terminal_padding=True,
            projection_chunk_tokens=4,
            mask_free_unpadded=True,
        )
        nll, counts = runner.score(tokens, targets, attention_mask=mask, return_per_row=True)
        expected.append(nll)
        assert counts == [4, 4]

    candidate, candidate_counts = CUDABatchedPathRunner(model, configs).score(
        tokens,
        targets,
        attention_mask=mask,
        projection_chunk_tokens=4,
    )
    measure_forward_error(
        torch.tensor(expected, device="cuda"), torch.tensor(candidate, device="cuda")
    ).require()
    assert candidate_counts == [[4, 4], [4, 4], [4, 4]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gemma3_oracle_concurrent_dual_and_candidate_batch_agree():
    """Gemma 3's typed RoPE and four-norm decoder contract stay exact on CUDA."""

    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import (
        CUDABatchedPathRunner,
        CUDAConcurrentRunner,
        CUDAPrefillRunner,
        measure_forward_error,
    )

    torch.manual_seed(31)
    model = (
        transformers.Gemma3ForCausalLM(
            transformers.Gemma3TextConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=64,
                sliding_window=4,
                layer_types=[
                    "sliding_attention",
                    "full_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                final_logit_softcapping=10.0,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    model.set_attn_implementation("eager")
    tokens = torch.tensor([[2, 7, 11, 13, 17], [2, 4, 8, 16, 32]], device="cuda")
    mask = torch.ones_like(tokens)
    targets = {
        position: ([0, 1], [int(tokens[0, position] + 1), int(tokens[1, position] + 1)])
        for position in range(tokens.shape[1])
    }
    configs = (
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.05),
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.05),
    )

    expected = []
    for config in configs:
        oracle = CUDAPrefillRunner(
            model,
            config,
            fused=False,
            allow_terminal_padding=True,
            projection_chunk_tokens=5,
            mask_free_unpadded=True,
        )
        oracle_nll, counts = oracle.score(
            tokens, targets, attention_mask=mask, return_per_row=True
        )
        expected.append(oracle_nll)
        assert counts == [5, 5]

        dual = CUDAConcurrentRunner(
            model, config, use_python_threads=False, dual_gemm=True
        )
        try:
            dual_nll, dual_counts = dual.score(
                tokens,
                targets,
                attention_mask=mask,
                return_per_row=True,
                projection_chunk_tokens=5,
            )
        finally:
            dual.close()
        measure_forward_error(
            torch.tensor(oracle_nll, device="cuda"),
            torch.tensor(dual_nll, device="cuda"),
        ).require()
        assert dual_counts == counts

    candidate, candidate_counts = CUDABatchedPathRunner(model, configs).score(
        tokens,
        targets,
        attention_mask=mask,
        projection_chunk_tokens=5,
    )
    measure_forward_error(
        torch.tensor(expected, device="cuda"), torch.tensor(candidate, device="cuda")
    ).require()
    assert candidate_counts == [[[5, 5][row] for row in range(2)] for _ in configs]


def test_gemma3_final_logit_softcap_matches_model_forward():
    transformers = pytest.importorskip("transformers")
    from recirculation.controller import project_causal_lm_logits

    model = transformers.Gemma3ForCausalLM(
        transformers.Gemma3TextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            final_logit_softcapping=3.0,
        )
    ).eval()
    tokens = torch.tensor([[2, 3, 4]])
    decoder_output = model.get_decoder()(input_ids=tokens, return_dict=True).last_hidden_state
    expected = model(input_ids=tokens, return_dict=True).logits
    torch.testing.assert_close(project_causal_lm_logits(model, decoder_output), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_generation_matches_updated_torch_controller(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: True)
    torch.manual_seed(20260821)
    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.2, ramp_tokens=4)
    tokens = torch.tensor([[1, 7, 11, 13, 17, 19]], device="cuda")
    expected = RecirculationController(model, config).generate(tokens, max_new_tokens=8)
    runner = CUDAConcurrentRunner(model, config)
    try:
        candidate = runner.generate(tokens, max_new_tokens=8)
    finally:
        runner.close()

    torch.testing.assert_close(candidate, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_single_thread_stream_enqueue_matches_threaded(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: True)
    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    threaded = CUDAConcurrentRunner(model, config)
    single_thread = CUDAConcurrentRunner(model, config, use_python_threads=False)
    try:
        expected = threaded.prefill(tokens, collect_logits=True)
        candidate = single_thread.prefill(tokens, collect_logits=True)
    finally:
        threaded.close()
        single_thread.close()
    torch.testing.assert_close(candidate[3], expected[3], rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].destination, expected[2].destination, rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].source, expected[2].source, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_snapshot_continuation_matches_full_prefill():
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAPrefillRunner

    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    runner = CUDAPrefillRunner(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1, ramp_tokens=2),
    )
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    expected = runner.prefill(tokens, collect_logits=True)
    _, cache, pending, _ = runner.prefill(tokens[:, :2])
    snapshot = runner.snapshot(cache, pending)
    candidate = runner.prefill_from_snapshot(tokens[:, 2:], snapshot, collect_logits=True)

    torch.testing.assert_close(candidate[3], expected[3][:, 2:], rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].destination, expected[2].destination, rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].source, expected[2].source, rtol=0, atol=0)
    assert candidate[2].token_position == expected[2].token_position == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_terminal_padding_batch_score_matches_scalar_gate():
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAPrefillRunner, measure_forward_error

    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    scalar = CUDAPrefillRunner(model, config)
    batched = CUDAPrefillRunner(model, config, allow_terminal_padding=True)
    prefix = torch.tensor([[1, 2]], device="cuda")
    _, cache, pending, _ = scalar.prefill(prefix)
    snapshot = scalar.snapshot(cache, pending)
    row0 = scalar.prefill_from_snapshot(torch.tensor([[3, 4]], device="cuda"), snapshot, collect_logits=True)[3]
    row1 = scalar.prefill_from_snapshot(torch.tensor([[7, 8, 9]], device="cuda"), snapshot, collect_logits=True)[3]
    expected = torch.stack(
        (
            torch.logsumexp(row0[0, 0].float(), dim=-1) - row0[0, 0, 5].float(),
            torch.logsumexp(row0[0, 1].float(), dim=-1) - row0[0, 1, 6].float(),
            torch.logsumexp(row1[0, 1].float(), dim=-1) - row1[0, 1, 10].float(),
        )
    ).sum()
    tokens = torch.tensor([[3, 4, 0], [7, 8, 9]], device="cuda")
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], device="cuda")
    candidate, count = batched.score_from_snapshot(
        tokens,
        snapshot,
        {0: ([0], [5]), 1: ([0, 1], [6, 10])},
        attention_mask=mask,
    )
    error = measure_forward_error(expected[None], torch.tensor([candidate], device="cuda"))
    error.require()
    assert count == 3
    row_nll, row_counts = batched.score_from_snapshot(
        tokens,
        snapshot,
        {0: ([0], [5]), 1: ([0, 1], [6, 10])},
        attention_mask=mask,
        return_per_row=True,
    )
    torch.testing.assert_close(torch.tensor(row_nll).sum(), expected.cpu(), rtol=2e-3, atol=2e-3)
    assert row_counts == [2, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_concurrent_graph_matches_eager_for_changed_tokens(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAGraphedConcurrentPrefill, measure_forward_error

    monkeypatch.setattr(__import__("sys"), "_is_gil_enabled", lambda: False)
    model = (
        transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )
        .half()
        .eval()
        .cuda()
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    changed_tokens = torch.tensor([[4, 3, 2, 1]], device="cuda")
    concurrent = CUDAConcurrentRunner(model, config)
    try:
        graphed = CUDAGraphedConcurrentPrefill(concurrent, tokens, warmups=1)
        expected = concurrent.prefill(changed_tokens)
        candidate = graphed.prefill(changed_tokens)
        logits_error = measure_forward_error(expected[0], candidate[0])
        pending_error = measure_forward_error(
            torch.cat((expected[2].destination, expected[2].source), dim=-1),
            torch.cat((candidate[2].destination, candidate[2].source), dim=-1),
        )
        logits_error.require()
        pending_error.require()
    finally:
        concurrent.close()


def test_mlx_forward_error_gate_uses_mean_absolute_error():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import measure_forward_error

    reference = mx.array([1.0, 2.0, 3.0])
    exact = measure_forward_error(reference, reference)
    exact.require()
    excessive = measure_forward_error(reference, reference + 0.01)
    with pytest.raises(RuntimeError, match="exceeds limit"):
        excessive.require()
    isolated_peak = measure_forward_error(
        mx.array([1000.0, 1000.0]),
        mx.array([1000.003, 1000.0]),
    )
    assert isolated_peak.max_absolute > 2e-3
    assert isolated_peak.mean_absolute < 2e-3
    isolated_peak.require()


@pytest.mark.parametrize("dtype_name", ("bfloat16", "float16", "float32"))
@pytest.mark.parametrize(("out_features", "in_features"), ((20, 128), (19, 130)))
def test_mlx_dual_gemv_matches_two_independent_projections(dtype_name, out_features, in_features):
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_kernels import DualGemvMetal

    dtype = getattr(mx, dtype_name)
    mx.random.seed(17)
    weight = mx.random.normal((out_features, in_features)).astype(dtype)
    input0 = mx.random.normal((1, 1, in_features)).astype(dtype)
    input1 = mx.random.normal((1, 1, in_features)).astype(dtype)
    reference0 = input0 @ weight.T
    reference1 = input1 @ weight.T
    candidate0, candidate1 = DualGemvMetal()(weight, input0, input1)
    mx.eval(reference0, reference1, candidate0, candidate1)

    assert mx.array_equal(reference0, candidate0).item()
    assert mx.array_equal(reference1, candidate1).item()


def test_mlx_dual_gemv_rejects_non_gemv_shapes():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_kernels import DualGemvMetal

    kernel = DualGemvMetal()
    with pytest.raises(ValueError, match="BN=1"):
        kernel(mx.zeros((8, 64)), mx.zeros((1, 64)), mx.zeros((1, 64)))


def test_mlx_qwen3_dual_token_layer_matches_two_serial_layer_calls():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3 import ModelArgs, TransformerBlock

    from recirculation.mlx_kernels import Qwen3DualTokenLayer

    mx.random.seed(23)
    layer = TransformerBlock(
        ModelArgs(
            model_type="qwen3",
            hidden_size=128,
            num_hidden_layers=1,
            intermediate_size=256,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=1_000_000.0,
            head_dim=32,
            tie_word_embeddings=False,
        )
    )
    mx.eval(layer.parameters())
    prefix = mx.random.normal((1, 1, 128)).astype(mx.float16)
    replay = mx.random.normal((1, 1, 128)).astype(mx.float16)
    current = mx.random.normal((1, 1, 128)).astype(mx.float16)
    prefix_cache = KVCache()
    mx.eval(layer(prefix, None, prefix_cache))
    cache_values = tuple(value + mx.zeros_like(value) for value in prefix_cache.state)
    serial_cache = KVCache()
    serial_cache.state = cache_values
    paired_cache = KVCache()
    paired_cache.state = tuple(value + mx.zeros_like(value) for value in cache_values)

    expected_replay = layer(replay, None, serial_cache)
    expected_current = layer(current, None, serial_cache)
    candidate_replay, candidate_current = Qwen3DualTokenLayer()(
        layer,
        replay,
        current,
        None,
        paired_cache,
    )
    mx.eval(expected_replay, expected_current, candidate_replay, candidate_current)

    assert mx.array_equal(expected_replay, candidate_replay).item()
    assert mx.array_equal(expected_current, candidate_current).item()
    for expected, candidate in zip(serial_cache.state, paired_cache.state):
        assert mx.array_equal(expected, candidate).item()


def test_mlx_reference_mixture_matches_published_norm_ratio_equation():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import measure_forward_error, mix_reference

    torch_destination = torch.tensor([[[3.0, 4.0]]])
    torch_source = torch.tensor([[[0.0, 2.0]]])
    destination = mx.array(torch_destination.numpy())
    source = mx.array(torch_source.numpy())
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    reference = torch_mix_reference(torch_destination, torch_source, 0.1, 0.9, True)
    candidate = mix_reference(destination, source, config)

    measure_forward_error(mx.array(reference.numpy()), candidate).require(limit=1e-6)


def test_mlx_mixture_rejects_shapes_rejected_by_torch_reference():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import CompiledNormMix, mix_reference

    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    destination = mx.zeros((1, 1, 4))
    source = mx.zeros((1, 1, 3))
    with pytest.raises(ValueError, match="same non-scalar shape"):
        mix_reference(destination, source, config)
    with pytest.raises(ValueError, match="same non-scalar shape"):
        CompiledNormMix(config)(destination, source, config)


def test_mlx_candidate_group_requires_a_shared_destination():
    pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import MLXCandidateGroupRecirculator

    model = SimpleNamespace(model=SimpleNamespace(layers=[object()] * 4))
    configs = (
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1),
        RecirculationConfig(source_layer=3, destination_layer=1, alpha=0.1),
    )
    with pytest.raises(ValueError, match="share one destination"):
        MLXCandidateGroupRecirculator(model, configs)


def test_mlx_candidate_group_is_exact_for_shared_and_divergent_tokens():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs

    from recirculation.mlx_backend import (
        CompiledNormMix,
        MLXCandidateGroupRecirculator,
        MLXRecirculator,
        measure_forward_error,
    )

    mx.random.seed(9)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    mx.eval(model.parameters())
    configs = (
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.1),
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.2),
    )
    scalar_runners = [MLXRecirculator(model, config, CompiledNormMix(config)) for config in configs]
    grouped = MLXCandidateGroupRecirculator(
        model, configs, [CompiledNormMix(config) for config in configs]
    )
    tokens = [1, 2, 3, 4]
    references = [runner.prefill(tokens, collect_logits=True) for runner in scalar_runners]
    candidate = grouped.prefill(tokens, collect_logits=True)

    for row, reference in enumerate(references):
        measure_forward_error(reference[3], candidate[3][row]).require(limit=0)
        measure_forward_error(reference[2].destination, candidate[2][row].destination).require(limit=0)
        measure_forward_error(reference[2].source, candidate[2][row].source).require(limit=0)
        for reference_cache, candidate_cache in zip(reference[1], candidate[1][row]):
            for reference_value, candidate_value in zip(reference_cache.state, candidate_cache.state):
                measure_forward_error(reference_value, candidate_value).require(limit=0)

    grouped._detach_lower_caches(candidate[1])
    for row, runner in enumerate(grouped.runners):
        grouped_logits, grouped_pending = runner.step(
            mx.array([[5 + row]], dtype=mx.int32),
            candidate[1][row],
            candidate[2][row],
        )
        reference_logits, reference_pending = scalar_runners[row].step(
            mx.array([[5 + row]], dtype=mx.int32),
            references[row][1],
            references[row][2],
        )
        measure_forward_error(reference_logits, grouped_logits).require(limit=0)
        measure_forward_error(reference_pending.destination, grouped_pending.destination).require(limit=0)
        measure_forward_error(reference_pending.source, grouped_pending.source).require(limit=0)


def test_mlx_contiguous_batch_matches_independent_recirculation_rows():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs

    from recirculation.mlx_backend import (
        CompiledNormMix,
        MLXBatchedRecirculator,
        MLXRecirculator,
        measure_forward_error,
    )

    mx.random.seed(41)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    mx.eval(model.parameters())
    config = RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.1)
    sequences = ((1, 2, 3, 4), (4, 3, 2, 1))
    references = [
        MLXRecirculator(model, config, CompiledNormMix(config)).prefill(
            sequence, collect_logits=True
        )
        for sequence in sequences
    ]
    batched = MLXBatchedRecirculator(model, config)
    candidate = batched.prefill(mx.array(sequences, dtype=mx.int32), collect_logits=True)

    for row, reference in enumerate(references):
        measure_forward_error(reference[3], candidate[3][row : row + 1]).require()
        measure_forward_error(reference[2].destination, candidate[2].destination[row : row + 1]).require()
        measure_forward_error(reference[2].source, candidate[2].source[row : row + 1]).require()
        for reference_cache, candidate_cache in zip(reference[1], candidate[1]):
            for reference_value, candidate_value in zip(
                reference_cache.state,
                candidate_cache.extract(row).state,
            ):
                measure_forward_error(reference_value, candidate_value).require()

    candidate_logits, candidate_pending = batched.step(
        mx.array([[5], [6]], dtype=mx.int32),
        candidate[1],
        candidate[2],
    )
    for row, reference in enumerate(references):
        reference_logits, reference_pending = MLXRecirculator(
            model, config, CompiledNormMix(config)
        ).step(
            mx.array([[5 + row]], dtype=mx.int32),
            reference[1],
            reference[2],
        )
        measure_forward_error(reference_logits, candidate_logits[row : row + 1]).require()
        measure_forward_error(
            reference_pending.destination,
            candidate_pending.destination[row : row + 1],
        ).require()


def test_mlx_continuous_batch_accepts_and_retires_requests():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs

    from recirculation.mlx_backend import MLXBatchedRecirculator, MLXContinuousBatch

    mx.random.seed(43)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    mx.eval(model.parameters())
    runner = MLXBatchedRecirculator(
        model,
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.1),
    )
    batch = MLXContinuousBatch(runner)
    batch.add("short", [1, 2], max_new_tokens=1)
    batch.add("long", [3, 4, 5], max_new_tokens=3)
    assert batch.request_ids == ("short", "long")
    assert set(batch.step()) == {"short", "long"}

    batch.add("late", [6, 7, 8, 9], max_new_tokens=1)
    assert batch.request_ids == ("long", "late")
    assert set(batch.step()) == {"long", "late"}
    assert batch.request_ids == ("long",)
    assert len(batch.tokens("late")) == 5
    assert set(batch.step()) == {"long"}
    assert batch.request_ids == ()


def test_mlx_candidate_group_snapshot_generation_matches_full_prompt():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.llama import Model, ModelArgs

    from recirculation.mlx_backend import CompiledNormMix, MLXCandidateGroupRecirculator

    mx.random.seed(15)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-5,
            vocab_size=32,
            max_position_embeddings=64,
        )
    )
    mx.eval(model.parameters())
    configs = (
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.1),
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.2),
    )
    grouped = MLXCandidateGroupRecirculator(
        model,
        configs,
        [CompiledNormMix(config) for config in configs],
    )
    prompt = [1, 2, 3, 4]
    expected = grouped.generate(prompt, max_new_tokens=4, eos_token_id=None)
    _, caches, pendings, _ = grouped.prefill(prompt[:2])
    snapshot = grouped.snapshot(caches, pendings)
    candidate = grouped.generate_from_snapshot(
        prompt[2:],
        snapshot,
        max_new_tokens=4,
        eos_token_id=None,
    )

    assert candidate == expected


def test_mlx_qwen3_candidate_group_auto_uses_exact_dual_gemv():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen3 import Model, ModelArgs

    from recirculation.mlx_backend import (
        CompiledNormMix,
        MLXCandidateGroupRecirculator,
        MLXQwen3DualGemvRecirculator,
        MLXRecirculator,
        measure_forward_error,
    )

    mx.random.seed(29)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=1_000_000.0,
            head_dim=32,
            tie_word_embeddings=False,
        )
    )
    mx.eval(model.parameters())
    configs = (
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1),
        RecirculationConfig(source_layer=3, destination_layer=0, alpha=0.2),
    )
    references = [
        MLXRecirculator(model, config, CompiledNormMix(config)).prefill([1, 2, 3], collect_logits=True)
        for config in configs
    ]
    grouped = MLXCandidateGroupRecirculator(
        model,
        configs,
        [CompiledNormMix(config) for config in configs],
    )
    assert all(isinstance(runner, MLXQwen3DualGemvRecirculator) for runner in grouped.runners)
    candidate = grouped.prefill([1, 2, 3], collect_logits=True)

    for row, reference in enumerate(references):
        measure_forward_error(reference[3], candidate[3][row]).require(limit=0)
        measure_forward_error(reference[2].destination, candidate[2][row].destination).require(limit=0)
        measure_forward_error(reference[2].source, candidate[2][row].source).require(limit=0)
        for reference_cache, candidate_cache in zip(reference[1], candidate[1][row]):
            for reference_value, candidate_value in zip(reference_cache.state, candidate_cache.state):
                measure_forward_error(reference_value, candidate_value).require(limit=0)


def test_mlx_qwen3_recirculation_matches_zero_feedback_serial_inference():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.qwen3 import Model, ModelArgs

    from recirculation.mlx_backend import CompiledNormMix, MLXRecirculator, measure_forward_error

    mx.random.seed(11)
    model = Model(
        ModelArgs(
            model_type="qwen3",
            hidden_size=16,
            num_hidden_layers=4,
            intermediate_size=32,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=1_000_000.0,
            head_dim=4,
            tie_word_embeddings=False,
        )
    )
    mx.eval(model.parameters())
    tokens = [1, 2, 3, 4]
    baseline_cache = make_prompt_cache(model)
    baseline = mx.concatenate(
        [model(mx.array([[token]], dtype=mx.int32), cache=baseline_cache) for token in tokens],
        axis=1,
    )
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.0)
    runner = MLXRecirculator(model, config, CompiledNormMix(config))
    candidate = runner.prefill(tokens, collect_logits=True)[3]

    mx.eval(baseline, candidate)
    measure_forward_error(baseline, candidate).require(limit=0)


@pytest.mark.parametrize(
    ("destination_values", "source_values"),
    (
        ((0.0, 0.0), (1.0, -1.0)),
        ((1.0, -1.0), (0.0, 0.0)),
        ((1.0, -1.0), (1e-10, -1e-10)),
    ),
)
def test_mlx_mixture_matches_torch_zero_and_tiny_norm_boundaries(destination_values, source_values):
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import CompiledNormMix, measure_forward_error

    torch_destination = torch.tensor([[destination_values]], dtype=torch.float32)
    torch_source = torch.tensor([[source_values]], dtype=torch.float32)
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    reference = torch_mix_reference(torch_destination, torch_source, 0.1, 0.9, True)
    compiled = CompiledNormMix(config)
    candidate = compiled(
        mx.array(torch_destination.numpy()),
        mx.array(torch_source.numpy()),
        config,
    )

    measure_forward_error(mx.array(reference.numpy()), candidate).require(limit=1e-6)


def test_mlx_ramp_matches_published_zero_based_schedule():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import CompiledNormMix, mix_reference

    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        normalize_source=False,
        ramp_tokens=10,
    )
    destination = mx.array([[[1.0, 0.0]]])
    source = mx.array([[[0.0, 1.0]]])
    compiled = CompiledNormMix(config)

    with pytest.raises(ValueError, match="input_step is required"):
        mix_reference(destination, source, config)
    for position, expected_alpha in ((0, 0.0), (1, 0.02), (9, 0.18), (10, 0.2), (11, 0.2)):
        expected = mx.array([[[1.0 - expected_alpha, expected_alpha]]])
        reference = mix_reference(destination, source, config, position)
        candidate = compiled(destination, source, config, position)
        assert mx.allclose(reference, expected, rtol=1e-6, atol=1e-6).item()
        assert mx.allclose(candidate, expected, rtol=1e-6, atol=1e-6).item()


def test_mlx_compiled_mixture_passes_128_step_accumulation_gate():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import CompiledNormMix, measure_forward_error

    torch.manual_seed(7)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    source_reference = torch.randn(1, 1, 2048, dtype=torch.float16)
    source_candidate = mx.array(source_reference.numpy())
    reference = torch.randn(1, 1, 2048, dtype=torch.float16)
    candidate = mx.array(reference.numpy())
    compiled = CompiledNormMix(config)
    reference_trace = []
    candidate_trace = []
    for _ in range(128):
        reference = torch_mix_reference(reference, source_reference, 0.1, 0.9, True)
        candidate = compiled(candidate, source_candidate, config)
        reference_trace.append(reference)
        candidate_trace.append(candidate)
        source_reference = 0.997 * source_reference + 0.003 * reference
        source_term = (source_candidate.astype(mx.float32) * 0.997).astype(source_candidate.dtype)
        candidate_term = (candidate.astype(mx.float32) * 0.003).astype(candidate.dtype)
        source_candidate = (source_term + candidate_term).astype(source_candidate.dtype)
        mx.eval(candidate, source_candidate)
    reference_array = mx.array(torch.cat(reference_trace, dim=1).float().numpy())
    candidate_array = mx.concatenate(candidate_trace, axis=1)
    error = measure_forward_error(reference_array, candidate_array)
    error.require()


def test_mlx_prefill_snapshot_holds_cache_and_pending_source():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import MLXPrefillSnapshot, PendingRecirculation

    pending = PendingRecirculation(mx.array([3.0]), mx.array([4.0]), 7)
    snapshot = MLXPrefillSnapshot(((mx.array([1.0]), mx.array([2.0])),), pending)
    assert snapshot.cache_states[0][0].item() == 1.0
    assert snapshot.pending.destination.item() == 3.0
    assert snapshot.pending.source.item() == 4.0
    assert snapshot.pending.token_position == 7


class _BiasLayer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return (hidden_states + self.value,)


class _ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_BiasLayer(1.0), _BiasLayer(2.0), _BiasLayer(3.0)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


class _ReplayLayer(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value
        self.last_input = None

    def forward(self, hidden_states, **kwargs):
        del kwargs
        self.last_input = hidden_states.detach().clone()
        return hidden_states + self.value


class _ReplayDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_ReplayLayer(1.0), _ReplayLayer(2.0), _ReplayLayer(3.0)])
        self.rotary_emb = lambda hidden_states, position_ids: (hidden_states, position_ids)


class _ReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ReplayDecoder()


class _LanguageModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = _ReplayDecoder()


class _CacheLayer:
    def __init__(self, length):
        self.length = length

    def crop(self, amount):
        assert amount == -1
        self.length -= 1


class _ReplayCache:
    def __init__(self, layers, length):
        self.layers = [_CacheLayer(length) for _ in layers]

    def get_seq_length(self):
        return self.layers[0].length


def test_controller_captures_first_iteration_state_without_cross_input_injection():
    model = _ToyDecoder()
    input_hidden = torch.zeros(1, 1, 2)
    with RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    ) as controller:
        first = model(input_hidden)
        second = model(input_hidden)
        pending = controller._pending_first_iteration

    assert torch.equal(first, torch.full_like(first, 6.0))
    assert torch.equal(second, torch.full_like(second, 6.0))
    assert pending is not None
    assert pending.input_step == 1
    assert torch.equal(pending.destination_residual, torch.full_like(input_hidden, 1.0))
    assert torch.equal(pending.source_residual, torch.full_like(input_hidden, 6.0))
    assert torch.equal(model(input_hidden), torch.full_like(input_hidden, 6.0))


def test_controller_runs_additional_top_stack_iteration_and_rewinds_only_top_stack_cache():
    model = _ReplayModel()
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(model.model.layers, length=1)

    controller._run_pending_top_stack_iteration(cache, torch.ones(1, 1, dtype=torch.long))

    assert cache.layers[0].length == 1
    assert [layer.length for layer in cache.layers[1:]] == [0, 0]
    assert torch.equal(model.model.layers[1].last_input, torch.full((1, 1, 2), 3.5))
    assert torch.equal(model.model.layers[2].last_input, torch.full((1, 1, 2), 5.5))
    assert controller._pending_first_iteration is None


def test_controller_replay_uses_the_decoder_that_exposed_the_layers():
    model = _LanguageModelWrapper()
    decoder = model.language_model.model
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(decoder.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(decoder.layers, length=1)

    controller._run_pending_top_stack_iteration(cache, torch.ones(1, 1, dtype=torch.long))

    assert torch.equal(decoder.layers[1].last_input, torch.full((1, 1, 2), 3.5))


def test_controller_validates_padding_before_mutating_the_replay_cache():
    model = _ReplayModel()
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(model.model.layers, length=2)

    with pytest.raises(ValueError, match="requires an unpadded attention mask"):
        controller.generate(
            torch.tensor([[1, 0]]),
            attention_mask=torch.tensor([[1, 0]]),
            max_new_tokens=0,
        )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    with pytest.raises(ValueError, match="requires an unpadded batch"):
        controller._run_pending_top_stack_iteration(cache, torch.tensor([[1, 0]]))

    assert [layer.length for layer in cache.layers] == [2, 2, 2]


def test_controller_rejects_non_terminal_padding_before_mutating_the_replay_cache():
    model = _ReplayModel()
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
        allow_terminal_padding=True,
    )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(model.model.layers, length=3)

    with pytest.raises(ValueError, match="real token after padding"):
        controller._run_pending_top_stack_iteration(cache, torch.tensor([[1, 0, 1]]))

    assert [layer.length for layer in cache.layers] == [3, 3, 3]


@pytest.mark.parametrize(
    "mixed",
    (None, torch.zeros(1, 2, 2), torch.zeros(1, 1, 2, dtype=torch.float64)),
)
def test_controller_rejects_invalid_mixer_outputs_before_mutating_the_replay_cache(mixed):
    model = _ReplayModel()
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0),
        mixer=lambda *_args: mixed,
    )
    controller._first_iteration_destination = torch.ones(1, 1, 2)
    controller._capture_first_iteration_source(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(model.model.layers, length=1)

    with pytest.raises((TypeError, ValueError), match="mixer must"):
        controller._run_pending_top_stack_iteration(cache, torch.ones(1, 1, dtype=torch.long))

    assert [layer.length for layer in cache.layers] == [1, 1, 1]


def test_controller_ramp_forwards_zero_based_input_step_to_mixer():
    model = _ReplayModel()
    observed = []

    def record_mix(destination, source, alpha, beta, normalize_source):
        observed.append((alpha, beta, normalize_source))
        return beta * destination + alpha * source

    controller = RecirculationController(
        model,
        RecirculationConfig(
            source_layer=2,
            destination_layer=0,
            alpha=0.2,
            normalize_source=False,
            ramp_tokens=10,
        ),
        mixer=record_mix,
    )
    for position in range(12):
        controller._first_iteration_destination = torch.ones(1, 1, 2)
        controller._capture_first_iteration_source(model.model.layers[2], (), (torch.zeros(1, 1, 2),))
        assert controller._pending_first_iteration is not None
        assert controller._pending_first_iteration.input_step == position
        cache = _ReplayCache(model.model.layers, length=position + 1)
        controller._run_pending_top_stack_iteration(cache, torch.ones(1, position + 1, dtype=torch.long))

    expected_alpha = [0.2 * min(position / 10.0, 1.0) for position in range(12)]
    assert [item[0] for item in observed] == pytest.approx(expected_alpha)
    assert [item[1] for item in observed] == pytest.approx([1.0 - alpha for alpha in expected_alpha])
    assert all(not item[2] for item in observed)


def test_torch_controller_zero_feedback_matches_serial_generation_and_skips_unused_readout():
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(20260821)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    tokens = torch.tensor([[1, 7, 11, 13]])
    expected = model.generate(tokens, max_new_tokens=3, do_sample=False, use_cache=True)
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.0),
    )
    projection_calls = 0
    original_projection = controller._project_first_iteration_logits

    def count_projection(hidden_states):
        nonlocal projection_calls
        projection_calls += 1
        return original_projection(hidden_states)

    controller._project_first_iteration_logits = count_projection
    candidate = controller.generate(tokens, max_new_tokens=3)

    torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
    assert projection_calls == 3


def test_torch_controller_dense_batch_matches_independent_zero_feedback_rows():
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(20260822)
    model = transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    tokens = torch.tensor([[1, 7, 11, 13], [4, 8, 12, 16]])
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.0)
    expected = torch.cat(
        [
            RecirculationController(model, config).generate(
                row[None, :], max_new_tokens=3
            )
            for row in tokens
        ],
        dim=0,
    )
    candidate = RecirculationController(model, config).generate(
        tokens,
        attention_mask=torch.ones_like(tokens),
        max_new_tokens=3,
    )

    torch.testing.assert_close(candidate, expected, rtol=0, atol=0)
