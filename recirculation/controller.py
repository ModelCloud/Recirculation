# SPDX-License-Identifier: Apache-2.0

"""Inference-only deep-to-shallow residual feedback.

This is the one-path, one-iteration variant from "Recirculation". The
controller operates through decoder-layer hooks and does not modify model
weights or checkpoint tensors. The controller's ``generate`` method performs serial prefill and cached
 decoding, so every token receives the source activation from the preceding
 token, including prompt tokens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RecirculationConfig:
    """Controls for the opt-in recirculation experiment."""

    source_layer: int
    destination_layer: int
    alpha: float = 0.10
    beta: float | None = None
    normalize_source: bool = True
    ramp_tokens: int = 0
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.source_layer <= self.destination_layer:
            raise ValueError("Recirculation source_layer must be deeper than destination_layer.")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Recirculation alpha must be in [0, 1].")
        beta = 1.0 - self.alpha if self.beta is None else self.beta
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Recirculation beta must be in [0, 1].")
        if self.ramp_tokens < 0:
            raise ValueError("Recirculation ramp_tokens must be non-negative.")
        if self.iterations != 1:
            raise ValueError("This implementation supports exactly one recirculation iteration.")
        object.__setattr__(self, "beta", float(beta))


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the standard Hugging Face decoder layer list without assuming a wrapper depth."""

    for path in ("layers", "model.layers", "model.model.layers", "language_model.model.layers"):
        current: Any = model
        try:
            for component in path.split("."):
                current = getattr(current, component)
        except AttributeError:
            continue
        if isinstance(current, nn.ModuleList):
            return current
    raise ValueError("Recirculation requires a model exposing a decoder ModuleList at model.layers.")


def _hidden_states_from_output(output: Any) -> torch.Tensor:
    """Extract the residual stream from a decoder-layer output."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "hidden_states") and torch.is_tensor(output.hidden_states):
        return output.hidden_states
    raise TypeError("Recirculation source layer returned no tensor hidden states.")


def _replace_hidden_states_output(output: Any, hidden_states: torch.Tensor) -> Any:
    """Replace the residual-stream field while preserving decoder auxiliary outputs."""

    if torch.is_tensor(output):
        return hidden_states
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    raise TypeError("Recirculation destination layer returned an unsupported output container.")


class RecirculationController:
    """Attach and detach one delayed deep-to-shallow residual feedback path.

    This is an inference-time intervention and does not alter model weights or
    serialized metadata. Use :meth:`generate` for paper-faithful serial prefill.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RecirculationConfig,
        *,
        mixer: Callable[[torch.Tensor, torch.Tensor, float, float, bool], torch.Tensor] | None = None,
    ):
        self.model = model
        self.config = config
        layers = _find_decoder_layers(model)
        if not 0 <= config.destination_layer < len(layers):
            raise ValueError(f"destination_layer is outside the {len(layers)}-layer decoder.")
        if not 0 <= config.source_layer < len(layers):
            raise ValueError(f"source_layer is outside the {len(layers)}-layer decoder.")
        self._destination = layers[config.destination_layer]
        self._source = layers[config.source_layer]
        self._handles: list[Any] = []
        self._pending_source: torch.Tensor | None = None
        self._position = 0
        self._active = False
        self._mixer = mixer

    def reset(self) -> None:
        """Discard delayed state before starting another prompt."""

        self._pending_source = None
        self._position = 0

    def _destination_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any):
        if self._pending_source is None:
            return output
        hidden_states = _hidden_states_from_output(output)
        source = self._pending_source.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if source.shape[:2] != hidden_states.shape[:2]:
            raise RuntimeError(
                "Recirculation source/destination shapes differ; use standard generation with a stable batch size."
            )
        strength = self.config.alpha
        if self.config.ramp_tokens:
            strength *= min(1.0, self._position / float(self.config.ramp_tokens))
        # With the default convex mixture, ramping alpha also ramps beta so
        # the residual magnitude does not collapse on the first positions.
        beta = self.config.beta
        if self.config.beta == 1.0 - self.config.alpha:
            beta = 1.0 - strength
        if self._mixer is not None:
            mixed = self._mixer(hidden_states, source, strength, beta, self.config.normalize_source)
            self._pending_source = None
            return _replace_hidden_states_output(output, mixed)
        if self.config.normalize_source:
            destination_norm = (
                hidden_states.float().norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            )
            source_norm = source.float().norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
            source = source * (destination_norm.to(source.dtype) / source_norm.to(source.dtype))
        mixed = beta * hidden_states + strength * source
        self._pending_source = None
        return _replace_hidden_states_output(output, mixed)

    def _source_hook(self, _module: nn.Module, _args: tuple[Any, ...], output: Any):
        hidden_states = _hidden_states_from_output(output)
        # The source from token t is consumed by token t+1. During prefill we
        # retain only the final prompt token, which is the first useful state
        # for autoregressive generation.
        self._pending_source = hidden_states[:, -1:, :].detach()
        self._position += int(hidden_states.shape[1])
        return output

    def attach(self) -> RecirculationController:
        if self._active:
            return self
        self.reset()
        self._handles = [
            self._destination.register_forward_hook(self._destination_hook),
            self._source.register_forward_hook(self._source_hook),
        ]
        self._active = True
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_source = None
        self._active = False

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate with paper-faithful serial prefill and cached decode.

        The paper's recurrence is across input steps, so a normal parallel
        ``model.generate`` prefill is not equivalent. This method runs every
        prompt token and generated token as one cached forward step.
        """

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Recirculation generation currently requires input_ids with shape [1, sequence].")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        was_active = self._active
        if not was_active:
            self.attach()
        try:
            prompt_length = input_ids.shape[1]
            generated = input_ids.clone()
            past_key_values = None
            logits = None
            for position in range(prompt_length):
                token = input_ids[:, position : position + 1]
                prefix_mask = attention_mask[:, : position + 1]
                outputs = self.model(
                    input_ids=token,
                    attention_mask=prefix_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
            for _ in range(max_new_tokens):
                if logits is None:
                    break
                token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, token), dim=1)
                if eos_token_id is not None and bool((token == eos_token_id).all()):
                    break
                full_mask = torch.ones((1, generated.shape[1]), dtype=attention_mask.dtype, device=generated.device)
                outputs = self.model(
                    input_ids=token,
                    attention_mask=full_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
            return generated
        finally:
            if not was_active:
                self.detach()

    def __enter__(self) -> RecirculationController:  # noqa: PYI034
        return self.attach()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.detach()


__all__ = ["RecirculationConfig", "RecirculationController"]
