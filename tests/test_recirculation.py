# SPDX-License-Identifier: Apache-2.0

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
    from recirculation.mlx_backend import MLXPrefillSnapshot

    snapshot = MLXPrefillSnapshot(((mx.array([1.0]), mx.array([2.0])),), mx.array([3.0]))
    assert snapshot.cache_states[0][0].item() == 1.0
    assert snapshot.pending_source.item() == 3.0


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


def test_recirculation_uses_previous_token_source_and_detaches_cleanly():
    model = _ToyDecoder()
    input_hidden = torch.zeros(1, 1, 2)
    with RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    ):
        first = model(input_hidden)
        second = model(input_hidden)

    assert torch.equal(first, torch.full_like(first, 6.0))
    assert torch.equal(second, torch.full_like(second, 8.5))
    assert torch.equal(model(input_hidden), torch.full_like(input_hidden, 6.0))


def test_recirculation_keeps_final_prompt_token_for_first_decode_step():
    model = _ToyDecoder()
    prompt = torch.zeros(1, 3, 1)
    with RecirculationController(
        model,
        RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.5, beta=0.5, normalize_source=False),
    ):
        prompt_output = model(prompt)
        decode_output = model(torch.zeros(1, 1, 1))

    assert prompt_output.shape == (1, 3, 1)
    assert torch.equal(decode_output, torch.full_like(decode_output, 8.5))
