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


def test_mlx_forward_error_gate_accepts_exact_and_rejects_excess_error():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import measure_forward_error

    reference = mx.array([1.0, 2.0, 3.0])
    exact = measure_forward_error(reference, reference)
    exact.require()
    excessive = measure_forward_error(reference, reference + 0.01)
    with pytest.raises(RuntimeError, match="exceeds limit"):
        excessive.require()
    absolute_only = measure_forward_error(mx.array([1000.0]), mx.array([1000.003]))
    assert absolute_only.relative_l2 < 2e-3
    assert absolute_only.normalized_max < 2e-3
    assert absolute_only.max_absolute > 2e-3
    with pytest.raises(RuntimeError, match="max_absolute"):
        absolute_only.require()


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

    with pytest.raises(ValueError, match="token_position is required"):
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


def test_controller_captures_same_token_path_without_cross_token_injection():
    model = _ToyDecoder()
    input_hidden = torch.zeros(1, 1, 2)
    with RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    ) as controller:
        first = model(input_hidden)
        second = model(input_hidden)
        pending = controller._pending

    assert torch.equal(first, torch.full_like(first, 6.0))
    assert torch.equal(second, torch.full_like(second, 6.0))
    assert pending is not None
    assert pending.token_position == 1
    assert torch.equal(pending.destination, torch.full_like(input_hidden, 1.0))
    assert torch.equal(pending.source, torch.full_like(input_hidden, 6.0))
    assert torch.equal(model(input_hidden), torch.full_like(input_hidden, 6.0))


def test_controller_replays_same_token_from_destination_and_rewinds_only_upper_cache():
    model = _ReplayModel()
    controller = RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    )
    controller._current_destination = torch.ones(1, 1, 2)
    controller._source_hook(model.model.layers[2], (), (torch.full((1, 1, 2), 6.0),))
    cache = _ReplayCache(model.model.layers, length=1)

    controller._recirculate_pending(cache, torch.ones(1, 1, dtype=torch.long))

    assert cache.layers[0].length == 1
    assert [layer.length for layer in cache.layers[1:]] == [0, 0]
    assert torch.equal(model.model.layers[1].last_input, torch.full((1, 1, 2), 3.5))
    assert torch.equal(model.model.layers[2].last_input, torch.full((1, 1, 2), 5.5))
    assert controller._pending is None


def test_controller_ramp_forwards_zero_based_pending_position_to_mixer():
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
        controller._current_destination = torch.ones(1, 1, 2)
        controller._source_hook(model.model.layers[2], (), (torch.zeros(1, 1, 2),))
        assert controller._pending is not None
        assert controller._pending.token_position == position
        cache = _ReplayCache(model.model.layers, length=position + 1)
        controller._recirculate_pending(cache, torch.ones(1, position + 1, dtype=torch.long))

    expected_alpha = [0.2 * min(position / 10.0, 1.0) for position in range(12)]
    assert [item[0] for item in observed] == pytest.approx(expected_alpha)
    assert [item[1] for item in observed] == pytest.approx([1.0 - alpha for alpha in expected_alpha])
    assert all(not item[2] for item in observed)
