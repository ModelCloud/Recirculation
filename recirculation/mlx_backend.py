# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful MLX recirculation with serial prefill and KV caching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask

from .controller import RecirculationConfig

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


def mix_reference(destination: mx.array, source: mx.array, config: RecirculationConfig) -> mx.array:
    """Published convex/non-convex norm-ratio mixture using ordinary MLX ops."""

    source = source.astype(destination.dtype)
    if config.normalize_source:
        destination_norm = mx.linalg.norm(destination.astype(mx.float32), axis=-1, keepdims=True)
        source_norm = mx.linalg.norm(source.astype(mx.float32), axis=-1, keepdims=True)
        source = source * (destination_norm / mx.maximum(source_norm, mx.array(1e-12))).astype(source.dtype)
    return config.beta * destination + config.alpha * source


class MLXRecirculator:
    """Run a loaded MLX-LM Llama model with delayed deep-to-shallow feedback."""

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        mixer: Callable[[mx.array, mx.array, RecirculationConfig], mx.array] = mix_reference,
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

    def step(self, token: mx.array, cache, pending_source: mx.array | None = None):
        """Process exactly one token and return logits plus the source for the next token."""

        if token.ndim == 1:
            token = token[None, :]
        if token.shape != (1, 1):
            raise ValueError("step requires one token with shape [1, 1]")
        decoder = self.model.model
        hidden = decoder.embed_tokens(token)
        full_mask = create_attention_mask(hidden, cache[decoder.fa_idx])
        sliding_mask = None
        if decoder.swa_idx is not None:
            sliding_mask = create_attention_mask(
                hidden, cache[decoder.swa_idx], window_size=decoder.sliding_window
            )
        next_source = None
        for index, (layer, layer_cache) in enumerate(zip(decoder.layers, cache)):
            mask = sliding_mask if layer.use_sliding else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)
            if index == self.config.destination_layer and pending_source is not None:
                hidden = self.mixer(hidden, pending_source, self.config)
            if index == self.config.source_layer:
                next_source = hidden
        hidden = decoder.norm(hidden)
        if self.model.args.tie_word_embeddings:
            logits = decoder.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits, next_source

    def prefill(self, tokens: Sequence[int] | mx.array, *, cache=None, pending_source=None):
        """Serially prefill tokens, preserving the paper's recurrence across prompt steps."""

        if isinstance(tokens, mx.array):
            tokens = tokens.reshape(-1).tolist()
        cache = self.make_cache() if cache is None else cache
        logits = None
        collected = []
        for token in tokens:
            logits, pending_source = self.step(mx.array([[int(token)]], dtype=mx.int32), cache, pending_source)
            collected.append(logits)
        if logits is None:
            raise ValueError("prefill requires at least one token")
        mx.eval(logits, pending_source)
        return logits, cache, pending_source, mx.concatenate(collected, axis=1)


__all__ = [
    "MAX_FORWARD_ERROR",
    "ForwardError",
    "MLXRecirculator",
    "measure_forward_error",
    "mix_reference",
]
