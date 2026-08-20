# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from recirculation import RecirculationConfig, RecirculationController


def test_mlx_forward_error_gate_accepts_exact_and_rejects_excess_error():
    mx = __import__("mlx.core", fromlist=["core"])
    from recirculation.mlx_backend import measure_forward_error

    reference = mx.array([1.0, 2.0, 3.0])
    exact = measure_forward_error(reference, reference)
    exact.require()
    excessive = measure_forward_error(reference, reference + 0.01)
    with __import__("pytest").raises(RuntimeError, match="exceeds limit"):
        excessive.require()


def test_mlx_reference_mixture_matches_published_norm_ratio_equation():
    mx = __import__("mlx.core", fromlist=["core"])
    from recirculation.mlx_backend import mix_reference

    destination = mx.array([[[3.0, 4.0]]])
    source = mx.array([[[0.0, 2.0]]])
    config = RecirculationConfig(source_layer=2, destination_layer=0, alpha=0.1)
    mixed = mix_reference(destination, source, config)
    assert mx.allclose(mixed, mx.array([[[2.7, 4.1]]])).item()


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
