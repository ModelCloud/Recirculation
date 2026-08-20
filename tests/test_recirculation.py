# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from recirculation import RecirculationConfig, RecirculationController


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_mixture_passes_forward_error_gate():
    from recirculation.cuda_backend import FusedNormMix, measure_forward_error, mix_reference

    torch.manual_seed(7)
    destination = torch.randn(4, 1, 2048, device="cuda", dtype=torch.float16)
    source = torch.randn_like(destination)
    reference = mix_reference(destination, source, 0.1, 0.9, True)
    candidate = FusedNormMix()(destination, source, 0.1, 0.9, True)
    error = measure_forward_error(reference, candidate)
    error.require()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fused_mixture_matches_reference_epsilon_boundaries():
    from recirculation.cuda_backend import FusedNormMix, measure_forward_error, mix_reference

    destination = torch.tensor([[[1.0, -1.0]]], device="cuda", dtype=torch.float32)
    source = torch.tensor([[[1e-10, -1e-10]]], device="cuda", dtype=torch.float32)
    reference = mix_reference(destination, source, 0.1, 0.9, True)
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
    from recirculation.cuda_backend import CUDAGraphedPrefill

    runner = SimpleNamespace(
        controller=SimpleNamespace(config=RecirculationConfig(source_layer=2, destination_layer=0, ramp_tokens=10))
    )
    with pytest.raises(ValueError, match="changed-input error exceeds the release gate"):
        CUDAGraphedPrefill(runner, torch.ones(1, 1, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_concurrent_stacks_match_sequential_scheduler():
    transformers = pytest.importorskip("transformers")
    from recirculation.cuda_backend import CUDAConcurrentRunner, CUDAPrefillRunner

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
    reference = CUDAPrefillRunner(model, config).prefill(tokens, collect_logits=True)
    concurrent = CUDAConcurrentRunner(model, config)
    try:
        candidate = concurrent.prefill(tokens, collect_logits=True)
    finally:
        concurrent.close()

    torch.testing.assert_close(candidate[3], reference[3], rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].destination, reference[2].destination, rtol=0, atol=0)
    torch.testing.assert_close(candidate[2].source, reference[2].source, rtol=0, atol=0)
    assert candidate[2].token_position == reference[2].token_position == 3


def test_mlx_forward_error_gate_accepts_exact_and_rejects_excess_error():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import measure_forward_error

    reference = mx.array([1.0, 2.0, 3.0])
    exact = measure_forward_error(reference, reference)
    exact.require()
    excessive = measure_forward_error(reference, reference + 0.01)
    with __import__("pytest").raises(RuntimeError, match="exceeds limit"):
        excessive.require()


def test_mlx_reference_mixture_matches_published_norm_ratio_equation():
    mx = pytest.importorskip("mlx.core")
    from recirculation.mlx_backend import mix_reference

    destination = mx.array([[[3.0, 4.0]]])
    source = mx.array([[[0.0, 2.0]]])
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    mixed = mix_reference(destination, source, config)
    assert mx.allclose(mixed, mx.array([[[2.7, 4.1]]])).item()


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
    from recirculation.mlx_backend import CompiledNormMix, measure_forward_error, mix_reference

    mx.random.seed(7)
    config = RecirculationConfig(source_layer=12, destination_layer=5, alpha=0.1)
    source = mx.random.normal((1, 1, 2048)).astype(mx.float16)
    reference = mx.random.normal((1, 1, 2048)).astype(mx.float16)
    candidate = reference
    compiled = CompiledNormMix(config)
    reference_trace = []
    candidate_trace = []
    for _ in range(128):
        reference = mix_reference(reference, source, config)
        candidate = compiled(candidate, source, config)
        reference_trace.append(reference)
        candidate_trace.append(candidate)
        source = 0.997 * source + 0.003 * reference
    error = measure_forward_error(mx.concatenate(reference_trace), mx.concatenate(candidate_trace))
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
