# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful MLX same-token recirculation with upper-KV replacement."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from .controller import RecirculationConfig, _mixing_coefficients

MAX_FORWARD_ERROR = 2e-3


@dataclass(frozen=True)
class ForwardError:
    max_absolute: float
    relative_l2: float
    normalized_max: float

    @property
    def rate(self) -> float:
        return max(self.relative_l2, self.normalized_max)

    def require(self, limit: float = MAX_FORWARD_ERROR) -> None:
        if self.rate > limit:
            raise RuntimeError(f"forward accumulation error {self.rate:.6g} exceeds limit {limit:.6g}")


@dataclass(frozen=True)
class PendingRecirculation:
    destination: mx.array
    source: mx.array
    token_position: int


@dataclass(frozen=True)
class MLXPrefillSnapshot:
    cache_states: tuple
    pending: PendingRecirculation


def measure_forward_error(reference: mx.array, candidate: mx.array) -> ForwardError:
    """Measure an optimized accumulated forward against the unfused MLX oracle."""

    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    difference = mx.abs(reference - candidate)
    max_absolute = float(mx.max(difference).item())
    reference_l2 = float(mx.linalg.norm(reference).item())
    relative_l2 = float(mx.linalg.norm(difference).item()) / max(reference_l2, 1e-12)
    reference_max = float(mx.max(mx.abs(reference)).item())
    normalized_max = max_absolute / max(reference_max, 1e-12)
    return ForwardError(max_absolute, relative_l2, normalized_max)


def _mix_expression(
    destination: mx.array,
    source: mx.array,
    alpha: float,
    beta: float,
    normalize_source: bool,
) -> mx.array:
    source = source.astype(destination.dtype)
    if normalize_source:
        destination_norm = mx.linalg.norm(destination.astype(mx.float32), axis=-1, keepdims=True)
        source_norm = mx.linalg.norm(source.astype(mx.float32), axis=-1, keepdims=True)
        source = source * (destination_norm / mx.maximum(source_norm, mx.array(1e-12))).astype(source.dtype)
    return beta * destination + alpha * source


def mix_reference(
    destination: mx.array,
    source: mx.array,
    config: RecirculationConfig,
    token_position: int | None = None,
) -> mx.array:
    """Published convex/non-convex norm-ratio mixture using ordinary MLX ops."""

    alpha, beta = _mixing_coefficients(config, token_position)
    return _mix_expression(destination, source, alpha, beta, config.normalize_source)


class CompiledNormMix:
    """MLX-compiled form of the exact norm-ratio reference expression."""

    def __init__(self, config: RecirculationConfig):
        self.config = config

        @mx.compile
        def compiled(destination, source, alpha, beta):
            return _mix_expression(destination, source, alpha, beta, config.normalize_source)

        self._compiled = compiled

    def __call__(
        self,
        destination: mx.array,
        source: mx.array,
        config: RecirculationConfig,
        token_position: int | None = None,
    ) -> mx.array:
        if config != self.config:
            raise ValueError("compiled mixer configuration cannot change after compilation")
        alpha, beta = _mixing_coefficients(config, token_position)
        return self._compiled(destination, source, alpha, beta)


class MLXRecirculator:
    """Run a loaded MLX-LM Llama model with delayed deep-to-shallow feedback."""

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        mixer: Callable[[mx.array, mx.array, RecirculationConfig, int], mx.array] = mix_reference,
    ):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("MLX recirculation requires an MLX-LM decoder model")
        if config.source_layer >= len(model.model.layers):
            raise ValueError("source layer is outside the decoder")
        self.model = model
        self.config = config
        self.mixer = mixer

    def make_cache(self):
        return self.model.make_cache()

    def snapshot(self, cache, pending: PendingRecirculation) -> MLXPrefillSnapshot:
        """Capture KV state and the final token's pending same-token second pass."""

        states = tuple(layer_cache.state for layer_cache in cache)
        mx.eval(*(array for state in states for array in state), pending.destination, pending.source)
        return MLXPrefillSnapshot(states, pending)

    def restore(self, snapshot: MLXPrefillSnapshot):
        """Restore a prefix into fresh cache objects without recomputing it."""

        cache = self.make_cache()
        if len(cache) != len(snapshot.cache_states):
            raise ValueError("snapshot layer count differs from model cache")
        for layer_cache, state in zip(cache, snapshot.cache_states):
            layer_cache.state = state
        return cache, snapshot.pending

    def _recirculate_pending(self, cache, pending: PendingRecirculation | None) -> None:
        """Replace the preceding token's upper-stack KV states with its second pass."""

        if pending is None:
            return
        decoder = self.model.model
        first_upper_layer = self.config.destination_layer + 1
        for layer_cache in cache[first_upper_layer:]:
            if layer_cache.trim(1) != 1:
                raise RuntimeError("cannot rewind the preceding token for recirculation")
        hidden = self.mixer(
            pending.destination,
            pending.source,
            self.config,
            pending.token_position,
        )
        full_mask = create_attention_mask(hidden, cache[first_upper_layer])
        sliding_mask = None
        if decoder.swa_idx is not None:
            sliding_mask = create_attention_mask(
                hidden, cache[first_upper_layer], window_size=decoder.sliding_window
            )
        for layer, layer_cache in zip(
            decoder.layers[first_upper_layer:], cache[first_upper_layer:]
        ):
            mask = sliding_mask if layer.use_sliding else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)
        mx.eval(hidden)

    def step(
        self,
        token: mx.array,
        cache,
        pending: PendingRecirculation | None = None,
        *,
        project_logits: bool = True,
    ):
        """Second-pass the preceding token, then first-pass one new token."""

        if token.ndim == 1:
            token = token[None, :]
        if token.shape != (1, 1):
            raise ValueError("step requires one token with shape [1, 1]")
        decoder = self.model.model
        token_position = int(cache[decoder.fa_idx].size())
        self._recirculate_pending(cache, pending)
        hidden = decoder.embed_tokens(token)
        full_mask = create_attention_mask(hidden, cache[decoder.fa_idx])
        sliding_mask = None
        if decoder.swa_idx is not None:
            sliding_mask = create_attention_mask(
                hidden, cache[decoder.swa_idx], window_size=decoder.sliding_window
            )
        destination = source = None
        for index, (layer, layer_cache) in enumerate(zip(decoder.layers, cache)):
            mask = sliding_mask if layer.use_sliding else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)
            if index == self.config.destination_layer:
                destination = hidden
            if index == self.config.source_layer:
                source = hidden
        if destination is None or source is None:
            raise RuntimeError("recirculation path activations were not captured")
        next_pending = PendingRecirculation(destination, source, token_position)
        if not project_logits:
            return None, next_pending
        hidden = decoder.norm(hidden)
        if self.model.args.tie_word_embeddings:
            logits = decoder.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits, next_pending

    def prefill(
        self,
        tokens: Sequence[int] | mx.array,
        *,
        cache=None,
        pending=None,
        collect_logits: bool = False,
    ):
        """Serially prefill tokens, preserving the paper's recurrence across prompt steps."""

        if isinstance(tokens, mx.array):
            tokens = tokens.reshape(-1).tolist()
        cache = self.make_cache() if cache is None else cache
        logits = None
        collected = []
        tokens = list(tokens)
        for index, token in enumerate(tokens):
            project_logits = collect_logits or index == len(tokens) - 1
            logits, pending = self.step(
                mx.array([[int(token)]], dtype=mx.int32),
                cache,
                pending,
                project_logits=project_logits,
            )
            if logits is not None:
                collected.append(logits)
        if logits is None:
            raise ValueError("prefill requires at least one token")
        mx.eval(logits, pending.destination, pending.source)
        return logits, cache, pending, mx.concatenate(collected, axis=1)

    def prefill_from_snapshot(
        self, tokens: Sequence[int] | mx.array, snapshot: MLXPrefillSnapshot, *, collect_logits: bool = False
    ):
        cache, pending = self.restore(snapshot)
        return self.prefill(
            tokens, cache=cache, pending=pending, collect_logits=collect_logits
        )


__all__ = [
    "MAX_FORWARD_ERROR",
    "CompiledNormMix",
    "ForwardError",
    "MLXPrefillSnapshot",
    "MLXRecirculator",
    "PendingRecirculation",
    "measure_forward_error",
    "mix_reference",
]
