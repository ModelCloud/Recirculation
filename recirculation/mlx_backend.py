# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful MLX same-token recirculation with upper-KV replacement."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache

from .controller import RecirculationConfig, _mixing_coefficients
from .mlx_kernels import DualGemvMetal, Qwen3DualTokenLayer

MAX_FORWARD_ERROR = 2e-3


@dataclass(frozen=True)
class ForwardError:
    max_absolute: float
    mean_absolute: float
    relative_l2: float
    normalized_max: float

    @property
    def rate(self) -> float:
        return max(self.max_absolute, self.relative_l2, self.normalized_max)

    def require(self, limit: float = MAX_FORWARD_ERROR) -> None:
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("forward error limit must be finite and non-negative")
        metrics = (self.max_absolute, self.mean_absolute, self.relative_l2, self.normalized_max)
        if not all(math.isfinite(value) for value in metrics) or self.rate > limit:
            raise RuntimeError(
                "forward accumulation error exceeds limit "
                f"{limit:.6g}: max_absolute={self.max_absolute:.6g}, "
                f"mean_absolute={self.mean_absolute:.6g}, "
                f"relative_l2={self.relative_l2:.6g}, normalized_max={self.normalized_max:.6g}"
            )


@dataclass(frozen=True)
class PendingRecirculation:
    destination: mx.array
    source: mx.array
    token_position: int


@dataclass(frozen=True)
class MLXPrefillSnapshot:
    cache_states: tuple
    pending: PendingRecirculation


@dataclass(frozen=True)
class MLXCandidateGroupSnapshot:
    snapshots: tuple[MLXPrefillSnapshot, ...]


def measure_forward_error(reference: mx.array, candidate: mx.array) -> ForwardError:
    """Measure an optimized MLX forward against values from the Torch oracle."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if math.prod(reference.shape) == 0:
        raise ValueError("forward error requires non-empty arrays")
    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    difference = mx.abs(reference - candidate)
    max_absolute = float(mx.max(difference).item())
    mean_absolute = float(mx.mean(difference).item())
    reference_l2 = float(mx.linalg.norm(reference).item())
    relative_l2 = float(mx.linalg.norm(difference).item()) / max(reference_l2, 1e-12)
    reference_max = float(mx.max(mx.abs(reference)).item())
    normalized_max = max_absolute / max(reference_max, 1e-12)
    return ForwardError(max_absolute, mean_absolute, relative_l2, normalized_max)


def _mix_expression(
    destination: mx.array,
    source: mx.array,
    alpha: float,
    beta: float,
    normalize_source: bool,
) -> mx.array:
    source = source.astype(destination.dtype)
    if normalize_source:
        destination_norm = mx.linalg.norm(destination.astype(mx.float32), axis=-1, keepdims=True).astype(
            destination.dtype
        )
        source_norm = mx.linalg.norm(source.astype(mx.float32), axis=-1, keepdims=True).astype(source.dtype)
        scale = mx.where(source_norm > 0, destination_norm / source_norm, mx.zeros_like(source_norm))
        source = source * scale
    # Torch applies Python scalar coefficients in FP32, rounds each product to
    # the residual dtype, and then performs the residual addition. Express the
    # same boundaries explicitly because MLX otherwise casts scalars to FP16.
    destination_term = (destination.astype(mx.float32) * beta).astype(destination.dtype)
    source_term = (source.astype(mx.float32) * alpha).astype(destination.dtype)
    return (destination_term + source_term).astype(destination.dtype)


def _validate_mix_inputs(destination: mx.array, source: mx.array) -> None:
    if destination.ndim < 1 or destination.shape != source.shape:
        raise ValueError("source and destination must have the same non-scalar shape")


def mix_reference(
    destination: mx.array,
    source: mx.array,
    config: RecirculationConfig,
    token_position: int | None = None,
) -> mx.array:
    """Eager MLX mirror of :func:`torch_mix_reference`."""

    _validate_mix_inputs(destination, source)
    alpha, beta = _mixing_coefficients(config, token_position)
    return _mix_expression(destination, source, alpha, beta, config.normalize_source)


class CompiledNormMix:
    """MLX-compiled mirror of the normative Torch norm-ratio expression."""

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
        _validate_mix_inputs(destination, source)
        alpha, beta = _mixing_coefficients(config, token_position)
        return self._compiled(destination, source, alpha, beta)


class MLXRecirculator:
    """Run a loaded MLX-LM decoder model with delayed deep-to-shallow feedback."""

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        mixer: Callable[[mx.array, mx.array, RecirculationConfig, int], mx.array] = mix_reference,
    ):
        if not hasattr(model, "model") or not hasattr(model.model, "layers"):
            raise TypeError("MLX recirculation requires an MLX-LM decoder model")
        if not 0 <= config.destination_layer < config.source_layer < len(model.model.layers):
            raise ValueError("recirculation layers are outside the decoder or are not ordered destination < source")
        self.model = model
        self.config = config
        self.mixer = mixer

    def make_cache(self):
        return make_prompt_cache(self.model)

    @staticmethod
    def _masks(decoder, hidden, cache, default_cache_index: int):
        """Build the full/sliding masks supported by the loaded MLX-LM decoder."""

        full_cache_index = getattr(decoder, "fa_idx", default_cache_index)
        full_mask = create_attention_mask(hidden, cache[full_cache_index])
        sliding_cache_index = getattr(decoder, "swa_idx", None)
        if sliding_cache_index is None:
            return full_mask, None
        sliding_mask = create_attention_mask(
            hidden,
            cache[sliding_cache_index],
            window_size=decoder.sliding_window,
        )
        return full_mask, sliding_mask

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
        full_mask, sliding_mask = self._masks(decoder, hidden, cache, first_upper_layer)
        for layer, layer_cache in zip(
            decoder.layers[first_upper_layer:], cache[first_upper_layer:]
        ):
            mask = sliding_mask if getattr(layer, "use_sliding", False) else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)

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
        full_cache_index = getattr(decoder, "fa_idx", 0)
        token_position = int(cache[full_cache_index].size())
        self._recirculate_pending(cache, pending)
        hidden = decoder.embed_tokens(token)
        full_mask, sliding_mask = self._masks(decoder, hidden, cache, full_cache_index)
        destination = source = None
        for index, (layer, layer_cache) in enumerate(zip(decoder.layers, cache)):
            mask = sliding_mask if getattr(layer, "use_sliding", False) else full_mask
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


class MLXQwen3DualGemvRecirculator(MLXRecirculator):
    """Use exact dual-GEMV projections for adjacent Qwen3 upper stacks."""

    @staticmethod
    def supports_model(model) -> bool:
        if getattr(model, "model_type", None) != "qwen3":
            return False
        projections = (
            projection.weight
            for layer in model.model.layers
            for projection in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            )
        )
        return all(DualGemvMetal.supports(weight) for weight in projections)

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        mixer: Callable[[mx.array, mx.array, RecirculationConfig, int], mx.array] = mix_reference,
    ):
        super().__init__(model, config, mixer)
        if getattr(model, "model_type", None) != "qwen3":
            raise TypeError("dual-GEMV recirculation currently supports Qwen3 only")
        if not self.supports_model(model):
            raise ValueError("Qwen3 projection shapes do not all use the supported BN=1 GEMV dispatch")
        self.paired_layer = Qwen3DualTokenLayer()

    def step(
        self,
        token: mx.array,
        cache,
        pending: PendingRecirculation | None = None,
        *,
        project_logits: bool = True,
    ):
        """Pair replay and current upper layers while preserving first-pass readout."""

        if pending is None:
            return super().step(token, cache, pending, project_logits=project_logits)
        if token.ndim == 1:
            token = token[None, :]
        if token.shape != (1, 1):
            raise ValueError("step requires one token with shape [1, 1]")

        decoder = self.model.model
        first_upper_layer = self.config.destination_layer + 1
        full_cache_index = getattr(decoder, "fa_idx", 0)
        token_position = int(cache[full_cache_index].size())
        current = decoder.embed_tokens(token)
        current_full_mask, current_sliding_mask = self._masks(decoder, current, cache, full_cache_index)
        for layer, layer_cache in zip(decoder.layers[:first_upper_layer], cache[:first_upper_layer]):
            mask = current_sliding_mask if getattr(layer, "use_sliding", False) else current_full_mask
            current = layer(current, mask, cache=layer_cache)
        return self._paired_upper(
            current,
            cache,
            pending,
            token_position,
            project_logits=project_logits,
        )

    def _paired_upper(
        self,
        destination: mx.array,
        cache,
        pending: PendingRecirculation,
        token_position: int,
        *,
        project_logits: bool,
    ):
        decoder = self.model.model
        first_upper_layer = self.config.destination_layer + 1
        for layer_cache in cache[first_upper_layer:]:
            if layer_cache.trim(1) != 1:
                raise RuntimeError("cannot rewind the preceding token for recirculation")
        replay = self.mixer(
            pending.destination,
            pending.source,
            self.config,
            pending.token_position,
        )
        replay_full_mask, replay_sliding_mask = self._masks(decoder, replay, cache, first_upper_layer)
        current = destination
        source = None
        for index, (layer, layer_cache) in enumerate(
            zip(decoder.layers[first_upper_layer:], cache[first_upper_layer:]),
            start=first_upper_layer,
        ):
            mask = replay_sliding_mask if getattr(layer, "use_sliding", False) else replay_full_mask
            replay, current = self.paired_layer(layer, replay, current, mask, layer_cache)
            if index == self.config.source_layer:
                source = current
        if source is None:
            raise RuntimeError("recirculation source activation was not captured")
        next_pending = PendingRecirculation(destination, source, token_position)
        if not project_logits:
            return None, next_pending
        hidden = decoder.norm(current)
        if self.model.args.tie_word_embeddings:
            logits = decoder.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits, next_pending


class MLXCandidateGroupRecirculator:
    """Share an exact lower stack across same-destination MLX candidates.

    Candidate upper stacks retain batch-one execution, which preserves the
    numerical behavior of independent :class:`MLXRecirculator` runs.  During a
    common teacher-forced stream, layers through the shared destination execute
    once.  Greedy decoding keeps sharing while candidate tokens agree and
    automatically clones the lower KV state when they diverge.
    """

    def __init__(
        self,
        model,
        configs: Sequence[RecirculationConfig],
        mixers: Sequence[Callable[[mx.array, mx.array, RecirculationConfig, int], mx.array]] | None = None,
    ):
        self.configs = tuple(configs)
        if not self.configs:
            raise ValueError("grouped MLX recirculation requires at least one candidate")
        destinations = {config.destination_layer for config in self.configs}
        if len(destinations) != 1:
            raise ValueError("grouped MLX candidates must share one destination layer")
        selected_mixers = tuple(mixers) if mixers is not None else tuple(mix_reference for _ in self.configs)
        if len(selected_mixers) != len(self.configs):
            raise ValueError("grouped MLX recirculation requires one mixer per candidate")
        runner_type = MLXRecirculator
        if MLXQwen3DualGemvRecirculator.supports_model(model):
            runner_type = MLXQwen3DualGemvRecirculator
        self.runners = tuple(runner_type(model, config, mixer) for config, mixer in zip(self.configs, selected_mixers))
        self.model = model
        self.destination_layer = self.configs[0].destination_layer

    @property
    def candidate_count(self) -> int:
        return len(self.runners)

    def make_caches(self):
        return [runner.make_cache() for runner in self.runners]

    def _share_lower_cache(self, caches) -> None:
        reference = caches[0]
        for layer_index in range(self.destination_layer + 1):
            state = reference[layer_index].state
            for cache in caches[1:]:
                cache[layer_index].state = state

    def _detach_lower_caches(self, caches) -> None:
        reference = caches[0]
        copied = []
        for cache in caches[1:]:
            for layer_index in range(self.destination_layer + 1):
                cache[layer_index].state = tuple(
                    value + mx.zeros_like(value) for value in reference[layer_index].state
                )
                copied.extend(cache[layer_index].state)
        if copied:
            mx.eval(*copied)

    def snapshot(self, caches, pendings) -> MLXCandidateGroupSnapshot:
        return MLXCandidateGroupSnapshot(
            tuple(runner.snapshot(cache, pending) for runner, cache, pending in zip(self.runners, caches, pendings))
        )

    def restore(self, snapshot: MLXCandidateGroupSnapshot):
        if len(snapshot.snapshots) != self.candidate_count:
            raise ValueError("snapshot candidate count differs from the grouped runner")
        restored = [runner.restore(item) for runner, item in zip(self.runners, snapshot.snapshots)]
        caches = [item[0] for item in restored]
        pendings = [item[1] for item in restored]
        self._share_lower_cache(caches)
        return caches, pendings

    def _run_shared_lower(self, token: mx.array, cache):
        decoder = self.model.model
        hidden = decoder.embed_tokens(token)
        full_cache_index = getattr(decoder, "fa_idx", 0)
        full_mask, sliding_mask = self.runners[0]._masks(decoder, hidden, cache, full_cache_index)
        for layer, layer_cache in zip(
            decoder.layers[: self.destination_layer + 1], cache[: self.destination_layer + 1]
        ):
            mask = sliding_mask if getattr(layer, "use_sliding", False) else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)
        return hidden

    def _run_candidate_upper(
        self,
        row: int,
        destination: mx.array,
        cache,
        token_position: int,
        *,
        project_logits: bool,
    ):
        decoder = self.model.model
        config = self.configs[row]
        first_upper_layer = self.destination_layer + 1
        hidden = destination
        full_mask, sliding_mask = self.runners[row]._masks(decoder, hidden, cache, first_upper_layer)
        source = None
        for index, (layer, layer_cache) in enumerate(
            zip(decoder.layers[first_upper_layer:], cache[first_upper_layer:]),
            start=first_upper_layer,
        ):
            mask = sliding_mask if getattr(layer, "use_sliding", False) else full_mask
            hidden = layer(hidden, mask, cache=layer_cache)
            if index == config.source_layer:
                source = hidden
        if source is None:
            raise RuntimeError("recirculation source activation was not captured")
        pending = PendingRecirculation(destination, source, token_position)
        if not project_logits:
            return None, pending
        hidden = decoder.norm(hidden)
        if self.model.args.tie_word_embeddings:
            logits = decoder.embed_tokens.as_linear(hidden)
        else:
            logits = self.model.lm_head(hidden)
        return logits, pending

    def step_shared(self, token: mx.array, caches, pendings, *, project_logits: bool = True):
        """Advance a token shared by candidates without changing batch-one math."""

        if token.ndim == 1:
            token = token[None, :]
        if token.shape != (1, 1):
            raise ValueError("shared grouped step requires one token with shape [1, 1]")
        decoder = self.model.model
        full_cache_index = getattr(decoder, "fa_idx", 0)
        positions = [int(cache[full_cache_index].size()) for cache in caches]
        if len(set(positions)) != 1:
            raise ValueError("grouped lower-stack sharing requires equal candidate cache lengths")
        token_position = positions[0]
        use_dual_gemv = all(
            isinstance(runner, MLXQwen3DualGemvRecirculator) and pending is not None
            for runner, pending in zip(self.runners, pendings)
        )
        for runner, cache, pending in zip(self.runners, caches, pendings):
            if pending is not None and pending.token_position != token_position - 1:
                raise ValueError("pending token position does not precede the current cache position")
            if not use_dual_gemv:
                runner._recirculate_pending(cache, pending)
        destination = self._run_shared_lower(token.astype(mx.int32), caches[0])
        self._share_lower_cache(caches)
        if use_dual_gemv:
            outputs = [
                runner._paired_upper(
                    destination,
                    cache,
                    pending,
                    token_position,
                    project_logits=project_logits,
                )
                for runner, cache, pending in zip(self.runners, caches, pendings)
            ]
        else:
            outputs = [
                self._run_candidate_upper(
                    row,
                    destination,
                    cache,
                    token_position,
                    project_logits=project_logits,
                )
                for row, cache in enumerate(caches)
            ]
        logits = [item[0] for item in outputs]
        next_pendings = [item[1] for item in outputs]
        mx.eval(
            *(value for value in logits if value is not None),
            *(pending.destination for pending in next_pendings),
            *(pending.source for pending in next_pendings),
        )
        return logits, next_pendings

    def prefill(
        self,
        tokens: Sequence[int] | mx.array,
        *,
        caches=None,
        pendings=None,
        collect_logits: bool = False,
    ):
        if isinstance(tokens, mx.array):
            tokens = tokens.reshape(-1).tolist()
        tokens = list(tokens)
        if not tokens:
            raise ValueError("prefill requires at least one token")
        caches = self.make_caches() if caches is None else caches
        pendings = [None] * self.candidate_count if pendings is None else pendings
        collected = [[] for _ in self.runners]
        logits = None
        for token_index, token in enumerate(tokens):
            project_logits = collect_logits or token_index == len(tokens) - 1
            logits, pendings = self.step_shared(
                mx.array([[int(token)]], dtype=mx.int32),
                caches,
                pendings,
                project_logits=project_logits,
            )
            if project_logits:
                for row, value in enumerate(logits):
                    collected[row].append(value)
        return (
            logits,
            caches,
            pendings,
            [mx.concatenate(values, axis=1) for values in collected],
        )

    def prefill_from_snapshot(
        self,
        tokens: Sequence[int] | mx.array,
        snapshot: MLXCandidateGroupSnapshot,
        *,
        collect_logits: bool = False,
    ):
        caches, pendings = self.restore(snapshot)
        return self.prefill(
            tokens,
            caches=caches,
            pendings=pendings,
            collect_logits=collect_logits,
        )

    def generate(
        self,
        prompt: Sequence[int] | mx.array,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return [[] for _ in self.runners]
        logits, caches, pendings, _ = self.prefill(prompt)
        return self._generate_from_state(
            logits,
            caches,
            pendings,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    def generate_from_snapshot(
        self,
        suffix: Sequence[int] | mx.array,
        snapshot: MLXCandidateGroupSnapshot,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> list[list[int]]:
        """Generate after an exact shared-prefix snapshot without recomputing it."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return [[] for _ in self.runners]
        logits, caches, pendings, _ = self.prefill_from_snapshot(suffix, snapshot)
        return self._generate_from_state(
            logits,
            caches,
            pendings,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    def _generate_from_state(
        self,
        logits,
        caches,
        pendings,
        *,
        max_new_tokens: int,
        eos_token_id: int | None,
    ) -> list[list[int]]:
        """Continue greedy generation from materialized per-candidate states."""

        continuations: list[list[int]] = [[] for _ in self.runners]
        active = [True] * self.candidate_count
        sharing_lower = True
        for step_index in range(max_new_tokens):
            token_values = [int(mx.argmax(value[:, -1, :], axis=-1).item()) for value in logits]
            for row, value in enumerate(token_values):
                if active[row]:
                    continuations[row].append(value)
                    if eos_token_id is not None and value == eos_token_id:
                        active[row] = False
            if step_index + 1 == max_new_tokens or not any(active):
                break
            can_share = all(active) and len(set(token_values)) == 1
            if sharing_lower and can_share:
                logits, pendings = self.step_shared(
                    mx.array([[token_values[0]]], dtype=mx.int32), caches, pendings
                )
                continue
            if sharing_lower:
                self._detach_lower_caches(caches)
                sharing_lower = False
            next_logits = list(logits)
            for row, runner in enumerate(self.runners):
                if not active[row]:
                    continue
                next_logits[row], pendings[row] = runner.step(
                    mx.array([[token_values[row]]], dtype=mx.int32),
                    caches[row],
                    pendings[row],
                )
            logits = next_logits
            mx.eval(
                *(logits[row] for row in range(self.candidate_count) if active[row]),
                *(pendings[row].destination for row in range(self.candidate_count) if active[row]),
                *(pendings[row].source for row in range(self.candidate_count) if active[row]),
            )
        return continuations


__all__ = [
    "MAX_FORWARD_ERROR",
    "CompiledNormMix",
    "ForwardError",
    "MLXCandidateGroupRecirculator",
    "MLXCandidateGroupSnapshot",
    "MLXPrefillSnapshot",
    "MLXQwen3DualGemvRecirculator",
    "MLXRecirculator",
    "PendingRecirculation",
    "measure_forward_error",
    "mix_reference",
]
