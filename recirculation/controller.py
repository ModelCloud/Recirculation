# SPDX-License-Identifier: Apache-2.0

"""Inference-only same-token deep-to-shallow residual feedback.

This is the one-path, one-additional-iteration variant from "Recirculation":
https://arxiv.org/html/2608.17981v1

The controller operates through decoder-layer hooks and does not modify model
weights or checkpoint tensors. Its ``generate`` method is the normative Torch
implementation of the serial equivalent of Figure 3(c): each input step's first
iteration supplies the source and destination residuals for its later additional
top-stack iteration, while readout occurs only after a first iteration.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class RecirculationConfig:
    """Controls for the paper's fixed-coefficient recirculation experiment.

    ``iterations`` counts additional iterations beyond the ordinary first
    iteration. ``normalize_source`` applies the source renormalization from
    Equation (2), matching its L2 norm to the destination residual.
    """

    source_layer: int
    destination_layer: int
    alpha: float = 0.10
    beta: float | None = None
    normalize_source: bool = True
    ramp_tokens: int = 0
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.source_layer < 0 or self.destination_layer < 0:
            raise ValueError("Recirculation layer indices must be non-negative.")
        if self.source_layer <= self.destination_layer:
            raise ValueError("Recirculation source_layer must be deeper than destination_layer.")
        if not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Recirculation alpha must be finite and in [0, 1].")
        beta = 1.0 - self.alpha if self.beta is None else self.beta
        if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
            raise ValueError("Recirculation beta must be finite and in [0, 1].")
        if self.ramp_tokens < 0:
            raise ValueError("Recirculation ramp_tokens must be non-negative.")
        if self.iterations != 1:
            raise ValueError("This implementation supports exactly one recirculation iteration.")
        object.__setattr__(self, "beta", float(beta))


@dataclass(frozen=True)
class _PendingFirstIterationState:
    """First-iteration residuals awaiting their additional top-stack iteration."""

    destination_residual: torch.Tensor
    source_residual: torch.Tensor
    input_step: int


def _mixing_coefficients(config: RecirculationConfig, input_step: int | None = None) -> tuple[float, float]:
    """Return the paper's alpha_t and its configured beta coefficient."""

    if input_step is not None and input_step < 0:
        raise ValueError("Recirculation input_step must be non-negative.")
    alpha_t = config.alpha
    if config.ramp_tokens:
        if input_step is None:
            raise ValueError("Recirculation input_step is required when ramp_tokens is enabled.")
        alpha_t *= min(input_step / float(config.ramp_tokens), 1.0)
    # The paper's convex mixture uses beta_t = 1 - alpha_t. An explicitly
    # supplied non-default beta remains fixed while alpha ramps.
    beta_t = config.beta
    if config.beta == 1.0 - config.alpha:
        beta_t = 1.0 - alpha_t
    return float(alpha_t), float(beta_t)


def torch_mix_reference(
    destination: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    beta: float,
    normalize_source: bool,
) -> torch.Tensor:
    """Apply the paper's Equations (1)-(2) using ordinary Torch operations.

    This function is the numerical oracle for accelerated backends. For the
    paper's undefined zero-source-norm edge case, the normalized source is
    defined as zero; every backend must mirror that policy.
    """

    if destination.ndim < 1 or destination.shape != source.shape:
        raise ValueError("source and destination must have the same non-scalar shape")
    if not destination.is_floating_point() or not source.is_floating_point():
        raise TypeError("source and destination must be floating-point tensors")
    source = source.to(device=destination.device, dtype=destination.dtype)
    if normalize_source:
        destination_norm = destination.float().norm(dim=-1, keepdim=True).to(destination.dtype)
        source_norm = source.float().norm(dim=-1, keepdim=True).to(destination.dtype)
        scale = torch.where(
            source_norm > 0,
            destination_norm / source_norm,
            torch.zeros_like(source_norm),
        )
        source = source * scale
    return beta * destination + alpha * source


def _find_decoder(model: nn.Module) -> nn.Module:
    """Find a Hugging Face decoder without assuming a specific causal-LM family.

    Llama and Qwen3 both expose ``get_decoder()``, while lightweight test
    doubles and some wrapped models expose the decoder only as an attribute.
    Keeping this lookup model-family neutral is important because the replay
    path also needs the decoder's rotary embedding and final norm.
    """

    get_decoder = getattr(model, "get_decoder", None)
    if callable(get_decoder):
        decoder = get_decoder()
        if isinstance(getattr(decoder, "layers", None), nn.ModuleList):
            return decoder
    if isinstance(getattr(model, "layers", None), nn.ModuleList):
        return model

    for path in ("model", "model.model", "language_model.model"):
        current: Any = model
        try:
            for component in path.split("."):
                current = getattr(current, component)
        except AttributeError:
            continue
        if isinstance(getattr(current, "layers", None), nn.ModuleList):
            return current
    raise ValueError("Recirculation requires a model exposing a decoder ModuleList at decoder.layers.")


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the standard Hugging Face decoder layer list."""

    return _find_decoder(model).layers


def _hidden_states_from_output(output: Any) -> torch.Tensor:
    """Extract the residual stream from a decoder-layer output."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if hasattr(output, "hidden_states") and torch.is_tensor(output.hidden_states):
        return output.hidden_states
    raise TypeError("Recirculation source layer returned no tensor hidden states.")


class RecirculationController:
    """Attach and detach one same-token, one-additional-iteration path.

    This is an inference-time intervention and does not alter model weights or
    serialized metadata. Use :meth:`generate` for paper-faithful serial prefill.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RecirculationConfig,
        *,
        mixer: Callable[[torch.Tensor, torch.Tensor, float, float, bool], torch.Tensor] | None = None,
        allow_terminal_padding: bool = False,
    ):
        self.model = model
        self.config = config
        self.decoder = _find_decoder(model)
        layers = self.decoder.layers
        if not 0 <= config.destination_layer < len(layers):
            raise ValueError(f"destination_layer is outside the {len(layers)}-layer decoder.")
        if not 0 <= config.source_layer < len(layers):
            raise ValueError(f"source_layer is outside the {len(layers)}-layer decoder.")
        self._destination = layers[config.destination_layer]
        self._source = layers[config.source_layer]
        self._handles: list[Any] = []
        self._pending_first_iteration: _PendingFirstIterationState | None = None
        self._first_iteration_destination: torch.Tensor | None = None
        self._in_additional_iteration = False
        self._next_input_step = 0
        self._active = False
        self._mixer = mixer
        self._allow_terminal_padding = allow_terminal_padding

    def reset(self) -> None:
        """Discard pending first-iteration state before starting another prompt."""

        self._pending_first_iteration = None
        self._first_iteration_destination = None
        self._next_input_step = 0

    def _capture_first_iteration_destination(self, _module: nn.Module, _args: tuple[Any, ...], output: Any):
        """Capture z_{t,t,d} without changing the first-iteration residual stream."""

        hidden_states = _hidden_states_from_output(output)
        if not self._in_additional_iteration:
            self._first_iteration_destination = hidden_states[:, -1:, :].detach()
        return output

    def _compute_recirculated_destination(
        self,
        destination_residual: torch.Tensor,
        source_residual: torch.Tensor,
        input_step: int,
    ) -> torch.Tensor:
        """Compute Equation (1)'s destination state for the additional iteration."""

        alpha_t, beta_t = _mixing_coefficients(self.config, input_step)
        if self._mixer is not None:
            return self._mixer(
                destination_residual,
                source_residual,
                alpha_t,
                beta_t,
                self.config.normalize_source,
            )
        return torch_mix_reference(
            destination_residual,
            source_residual,
            alpha_t,
            beta_t,
            self.config.normalize_source,
        )

    def _capture_first_iteration_source(self, _module: nn.Module, _args: tuple[Any, ...], output: Any):
        """Complete the first-iteration state captured for the current input step."""

        hidden_states = _hidden_states_from_output(output)
        if self._in_additional_iteration:
            return output
        if self._first_iteration_destination is None:
            raise RuntimeError("destination activation was not captured before source activation")
        input_step_count = int(hidden_states.shape[1])
        if input_step_count < 1:
            raise RuntimeError("source residual contains no input steps")
        self._pending_first_iteration = _PendingFirstIterationState(
            destination_residual=self._first_iteration_destination,
            source_residual=hidden_states[:, -1:, :].detach(),
            input_step=self._next_input_step + input_step_count - 1,
        )
        self._next_input_step += input_step_count
        return output

    def _run_pending_top_stack_iteration(self, past_key_values, attention_mask: torch.Tensor) -> None:
        """Run the preceding input step's additional top-stack iteration.

        This is the serial form of Figure 3(c)'s top-stack branch. It begins
        with Equation (1)'s mixed destination and replaces only that input
        step's top-stack KV entries.
        """

        if self._pending_first_iteration is None:
            return
        first_iteration_state = self._pending_first_iteration
        decoder = self.decoder
        sequence_length = past_key_values.get_seq_length()
        if sequence_length < 1:
            raise RuntimeError("cannot recirculate an empty KV cache")
        top_stack_start = self.config.destination_layer + 1
        for layer in past_key_values.layers[top_stack_start:]:
            layer.crop(-1)
        hidden_states = self._compute_recirculated_destination(
            first_iteration_state.destination_residual,
            first_iteration_state.source_residual,
            first_iteration_state.input_step,
        )
        # Construct on-device so the additional iteration remains CUDA Graph capture-safe.
        position_ids = torch.full((1, 1), sequence_length - 1, dtype=torch.long, device=hidden_states.device)
        capturing = hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        if (
            not capturing
            and not self._allow_terminal_padding
            and not bool(attention_mask[:, :sequence_length].all())
        ):
            raise ValueError("paper-faithful additional iteration currently requires an unpadded batch")
        position_embeddings = decoder.rotary_emb(hidden_states, position_ids=position_ids)
        self._in_additional_iteration = True
        try:
            for layer in decoder.layers[top_stack_start:]:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
        finally:
            self._in_additional_iteration = False
        # Figure 3: the additional iteration updates state but has no readout.
        self._pending_first_iteration = None

    def attach(self) -> RecirculationController:
        if self._active:
            return self
        self.reset()
        self._handles = [
            self._destination.register_forward_hook(self._capture_first_iteration_destination),
            self._source.register_forward_hook(self._capture_first_iteration_source),
        ]
        self._active = True
        return self

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending_first_iteration = None
        self._first_iteration_destination = None
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
        """Generate with the paper's first/additional-iteration schedule.

        Before input step ``t`` begins its first iteration, this serial reference
        completes input step ``t-1``'s additional top-stack iteration. Readout
        follows the first iteration only, as specified by Figure 3.
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
        else:
            self.reset()
        try:
            prompt_length = input_ids.shape[1]
            generated = input_ids.clone()
            past_key_values = None
            first_iteration_logits = None
            for input_step in range(prompt_length):
                current_input = input_ids[:, input_step : input_step + 1]
                prefix_mask = attention_mask[:, : input_step + 1]
                if past_key_values is not None:
                    # Serial equivalent of Figure 3(c)'s preceding top-stack branch.
                    self._run_pending_top_stack_iteration(past_key_values, prefix_mask)
                first_iteration_outputs = self.model(
                    input_ids=current_input,
                    attention_mask=prefix_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = first_iteration_outputs.past_key_values
                # Figure 3: readout follows the current input's first iteration.
                first_iteration_logits = first_iteration_outputs.logits[:, -1, :]
            for _ in range(max_new_tokens):
                if first_iteration_logits is None:
                    break
                next_input = first_iteration_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat((generated, next_input), dim=1)
                if eos_token_id is not None and bool((next_input == eos_token_id).all()):
                    break
                full_mask = torch.ones(
                    (1, generated.shape[1]), dtype=attention_mask.dtype, device=generated.device
                )
                self._run_pending_top_stack_iteration(past_key_values, full_mask)
                first_iteration_outputs = self.model(
                    input_ids=next_input,
                    attention_mask=full_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = first_iteration_outputs.past_key_values
                first_iteration_logits = first_iteration_outputs.logits[:, -1, :]
            return generated
        finally:
            if not was_active:
                self.detach()

    def __enter__(self) -> RecirculationController:  # noqa: PYI034
        return self.attach()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.detach()


__all__ = ["RecirculationConfig", "RecirculationController", "torch_mix_reference"]
