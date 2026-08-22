# SPDX-License-Identifier: Apache-2.0

"""CUDA recirculation kernels and serial/concurrent inference schedulers."""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from logbar import LogBar
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - depends on the CUDA installation
    raise ImportError("The CUDA backend requires Triton. Install the 'cuda' extra.") from error

from .controller import (
    RecirculationConfig,
    RecirculationController,
    _decoder_position_embeddings,
    _layer_position_embeddings,
    _mixing_coefficients,
    _PendingFirstIterationState,
    project_causal_lm_logits,
    torch_mix_reference,
)

MAX_FORWARD_ERROR = 2e-3
MAX_FUSED_OR_BATCHED_FORWARD_ERROR = 4e-3
LOG = LogBar.shared()
_CUDA_GRAPH_CAPTURE_LOCK = threading.Lock()
# Preserve the existing CUDA-backend API while making the Torch implementation
# the sole eager reference and fallback.
mix_reference = torch_mix_reference


def _new_recirculation_cache(model, data=None):
    """Create a cache that can roll back Gemma 3 sliding-window upper layers."""

    return (
        DynamicCache(data, config=model.config)
        if data is not None
        else DynamicCache(config=model.config)
    )


def _activate_upper_cache_recording(cache, top_stack_start: int) -> None:
    """Preserve rollback state only where recirculation replaces KV entries."""

    for cache_layer in cache.layers[top_stack_start:]:
        activate = getattr(cache_layer, "activate_past_recording", None)
        if callable(activate):
            activate()


def _trim_recorded_upper_cache(cache, top_stack_start: int) -> None:
    """Restrict recorded Gemma sliding caches after the current-token update."""

    for cache_layer in cache.layers[top_stack_start:]:
        if getattr(cache_layer, "record_past", False):
            cache_layer.crop(0)


def log_concurrency_mode() -> bool:
    """Log and return the GIL mode used by the concurrent CUDA scheduler."""

    gil_enabled = bool(getattr(sys, "_is_gil_enabled", lambda: True)())
    if gil_enabled:
        LOG.info.once(
            "GIL=1 detected: CUDAConcurrentRunner is enabled because PyTorch CUDA operations can release the GIL. "
            "A free-threaded build such as CPython 3.14t with -X gil=0 or PYTHON_GIL=0 may further reduce host-side "
            "scheduling overhead."
        )
    else:
        LOG.info.once(
            "GIL=0 detected: CUDAConcurrentRunner is enabled for faster parallel inference with Python workers and "
            "CUDA streams."
        )
    return gil_enabled


@dataclass(frozen=True)
class ForwardError:
    max_absolute: float
    mean_absolute: float
    relative_l2: float
    normalized_max: float

    @property
    def rate(self) -> float:
        """Return the release-gated mean absolute forward error."""

        return self.mean_absolute

    def require(self, limit: float = MAX_FORWARD_ERROR) -> None:
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("forward error limit must be finite and non-negative")
        metrics = (self.max_absolute, self.mean_absolute, self.relative_l2, self.normalized_max)
        if not all(math.isfinite(value) for value in metrics) or self.rate > limit:
            raise RuntimeError(
                "forward accumulation error exceeds limit "
                f"{limit:.6g}: max_absolute={self.max_absolute:.6g}, "
                f"mean_absolute={self.mean_absolute:.6g} (gate metric), "
                f"relative_l2={self.relative_l2:.6g}, normalized_max={self.normalized_max:.6g}"
            )


@dataclass(frozen=True)
class CUDARecirculationState:
    """First-pass activations waiting for their same-token upper replay."""

    destination_residual: torch.Tensor
    source_residual: torch.Tensor
    input_step: int

    # Compatibility aliases for artifacts and callers created before the
    # controller adopted the paper's first/additional-iteration terminology.
    @property
    def destination(self) -> torch.Tensor:
        return self.destination_residual

    @property
    def source(self) -> torch.Tensor:
        return self.source_residual

    @property
    def token_position(self) -> int:
        return self.input_step


@dataclass(frozen=True)
class CUDAPrefillSnapshot:
    """Immutable CUDA KV and pending state for shared-prefix screening."""

    cache_data: tuple[tuple[torch.Tensor | None, ...], ...]
    pending: CUDARecirculationState


def _snapshot_cuda_state(cache, pending: CUDARecirculationState) -> CUDAPrefillSnapshot:
    cache_data = tuple(
        tuple(value.detach().clone() if torch.is_tensor(value) else value for value in layer_data)
        for layer_data in cache
    )
    frozen_pending = CUDARecirculationState(
        pending.destination_residual.detach().clone(),
        pending.source_residual.detach().clone(),
        pending.input_step,
    )
    return CUDAPrefillSnapshot(cache_data, frozen_pending)


def measure_forward_error(reference: torch.Tensor, candidate: torch.Tensor) -> ForwardError:
    """Measure an optimized accumulated forward against the PyTorch oracle."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have the same shape")
    if reference.numel() == 0:
        raise ValueError("forward error requires non-empty tensors")
    reference = reference.float()
    candidate = candidate.float()
    difference = (reference - candidate).abs()
    max_absolute = difference.max().item()
    mean_absolute = difference.mean().item()
    reference_l2 = torch.linalg.vector_norm(reference).item()
    relative_l2 = torch.linalg.vector_norm(difference).item() / max(reference_l2, 1e-12)
    reference_max = reference.abs().max().item()
    normalized_max = max_absolute / max(reference_max, 1e-12)
    return ForwardError(max_absolute, mean_absolute, relative_l2, normalized_max)


@triton.jit
def _norm_mix_kernel(
    destination_pointer,
    source_pointer,
    output_pointer,
    hidden_size: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    normalize_source: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    indices = row * hidden_size + offsets
    destination = tl.load(destination_pointer + indices, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_pointer + indices, mask=mask, other=0.0).to(tl.float32)
    element_type: tl.constexpr = output_pointer.dtype.element_ty
    if normalize_source:
        destination_norm = tl.sqrt(tl.sum(destination * destination, axis=0))
        source_norm = tl.sqrt(tl.sum(source * source, axis=0))
        # Match the Torch oracle's dtype boundaries. Each FP32 norm is cast
        # back to the residual dtype before division and source scaling.
        destination_norm = destination_norm.to(element_type)
        source_norm = source_norm.to(element_type)
        scale = tl.where(source_norm > 0.0, destination_norm / source_norm, 0.0).to(element_type)
        source = (source * scale).to(element_type)
        # Torch materializes the normalized source in the residual dtype before
        # applying alpha.  A local cast is insufficient here because Triton can
        # fold the scale and alpha multiplies across that FP16/BF16 rounding
        # boundary.  Use the output buffer as row-local scratch to make the
        # oracle's dtype boundary observable without another kernel launch.
        tl.store(output_pointer + indices, source, mask=mask)
        tl.debug_barrier()
        source = tl.load(output_pointer + indices, mask=mask, other=0.0).to(tl.float32)
    destination_term = (beta * destination).to(element_type)
    source_term = (alpha * source).to(element_type)
    mixed = (destination_term + source_term).to(element_type)
    tl.store(output_pointer + indices, mixed, mask=mask)


class FusedNormMix:
    """Fuse both L2 reductions, source scaling, and residual mixing into one CUDA launch."""

    def __call__(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
        alpha: float,
        beta: float,
        normalize_source: bool,
    ) -> torch.Tensor:
        if not destination.is_cuda:
            raise ValueError("FusedNormMix requires a CUDA destination tensor")
        if destination.ndim < 1 or destination.shape != source.shape:
            raise ValueError("source and destination must have the same non-scalar shape")
        if destination.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError("FusedNormMix supports float16, bfloat16, and float32")
        destination = destination.contiguous()
        source = source.to(device=destination.device, dtype=destination.dtype).contiguous()
        hidden_size = destination.shape[-1]
        if hidden_size == 0:
            return torch.empty_like(destination)
        block_size = triton.next_power_of_2(hidden_size)
        if block_size > 65536:
            raise ValueError("FusedNormMix supports hidden sizes up to 65536")
        output = torch.empty_like(destination)
        rows = destination.numel() // hidden_size
        _norm_mix_kernel[(rows,)](
            destination,
            source,
            output,
            hidden_size=hidden_size,
            alpha=float(alpha),
            beta=float(beta),
            normalize_source=normalize_source,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output


class CUDAPrefillRunner:
    """Same-token replay prefill that skips intermediate vocabulary projections."""

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        *,
        fused: bool = True,
        skip_intermediate_logits: bool = True,
        allow_terminal_padding: bool = False,
        projection_chunk_tokens: int = 1,
        mask_free_unpadded: bool = False,
    ):
        if not hasattr(model, "get_decoder") or model.get_output_embeddings() is None:
            raise TypeError("CUDA prefill requires a Hugging Face causal language model")
        mixer = FusedNormMix() if fused else mix_reference
        self.model = model
        self.decoder = model.get_decoder()
        self.output_embeddings = model.get_output_embeddings()
        self.controller = RecirculationController(
            model,
            config,
            mixer=mixer,
            allow_terminal_padding=allow_terminal_padding,
        )
        self.skip_intermediate_logits = skip_intermediate_logits
        self.allow_terminal_padding = allow_terminal_padding
        if projection_chunk_tokens < 1:
            raise ValueError("projection_chunk_tokens must be positive")
        self.projection_chunk_tokens = projection_chunk_tokens
        self.mask_free_unpadded = mask_free_unpadded

    def _attach_controller_state(
        self,
        pending: CUDARecirculationState | None,
        next_input_step: int,
    ) -> None:
        """Bridge the public CUDA state into the Torch controller's oracle state."""

        self.controller.attach()
        self.controller._next_input_step = next_input_step
        self.controller._first_iteration_destination = None
        if pending is None:
            self.controller._pending_first_iteration = None
            return
        if pending.input_step != next_input_step - 1:
            raise ValueError("pending input step does not precede the next CUDA input step")
        self.controller._pending_first_iteration = _PendingFirstIterationState(
            destination_residual=pending.destination_residual,
            source_residual=pending.source_residual,
            input_step=pending.input_step,
        )

    def _export_controller_state(self) -> CUDARecirculationState:
        pending = self.controller._pending_first_iteration
        if pending is None:
            raise RuntimeError("prefill did not capture pending first-iteration state")
        return CUDARecirculationState(
            destination_residual=pending.destination_residual.detach(),
            source_residual=pending.source_residual.detach(),
            input_step=pending.input_step,
        )

    @torch.inference_mode()
    def prefill(
        self,
        tokens: Sequence[int] | torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        collect_logits: bool = False,
        cache=None,
        pending: CUDARecirculationState | None = None,
    ):
        """Return final logits, KV cache, pending same-token replay, and optionally every token's logits."""

        device = next(self.model.parameters()).device
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(list(tokens), dtype=torch.long, device=device)
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("prefill requires tokens with shape [sequence] or [batch, sequence]")
        prefix_length = 0 if cache is None else cache.get_seq_length()
        total_length = prefix_length + tokens.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((tokens.shape[0], total_length), dtype=torch.long, device=device)
        else:
            attention_mask = attention_mask.to(device=device)
        if attention_mask.shape != (tokens.shape[0], total_length):
            raise ValueError("attention_mask must cover the cached prefix and every new token")
        if self.allow_terminal_padding and bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
            raise ValueError("terminal padding masks cannot contain a real token after padding begins")

        logits = None
        collected = []
        self._attach_controller_state(pending, prefix_length)
        try:
            for position in range(tokens.shape[1]):
                prefix_mask = attention_mask[:, : prefix_length + position + 1]
                if cache is not None:
                    self.controller._run_pending_top_stack_iteration(cache, prefix_mask)
                output = self.decoder(
                    input_ids=tokens[:, position : position + 1],
                    attention_mask=prefix_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                self.controller._trim_recorded_upper_cache(cache)
                is_final = position == tokens.shape[1] - 1
                if collect_logits or not self.skip_intermediate_logits or is_final:
                    logits = project_causal_lm_logits(
                        self.model, output.last_hidden_state[:, -1:, :]
                    )
                if collect_logits or is_final:
                    collected.append(logits)
            if logits is None:  # pragma: no cover - guarded by model/config validation
                raise RuntimeError("prefill did not produce logits")
            state = self._export_controller_state()
            return logits, cache, state, torch.cat(collected, dim=1)
        finally:
            self.controller.detach()


    def snapshot(self, cache, pending: CUDARecirculationState) -> CUDAPrefillSnapshot:
        return _snapshot_cuda_state(cache, pending)

    def restore(self, snapshot: CUDAPrefillSnapshot, *, batch_size: int = 1):
        cache = _new_recirculation_cache(self.model, snapshot.cache_data)
        snapshot_batch = snapshot.pending.destination_residual.shape[0]
        if batch_size % snapshot_batch:
            raise ValueError("requested batch size must be a multiple of the snapshot batch size")
        repeats = batch_size // snapshot_batch
        pending = snapshot.pending
        if repeats > 1:
            cache.batch_repeat_interleave(repeats)
            pending = CUDARecirculationState(
                pending.destination_residual.repeat_interleave(repeats, dim=0),
                pending.source_residual.repeat_interleave(repeats, dim=0),
                pending.input_step,
            )
        return cache, pending

    def prefill_from_snapshot(
        self,
        tokens: Sequence[int] | torch.Tensor,
        snapshot: CUDAPrefillSnapshot,
        *,
        collect_logits: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size = tokens.shape[0] if isinstance(tokens, torch.Tensor) and tokens.ndim == 2 else 1
        cache, pending = self.restore(snapshot, batch_size=batch_size)
        return self.prefill(
            tokens,
            cache=cache,
            pending=pending,
            collect_logits=collect_logits,
            attention_mask=attention_mask,
        )

    @torch.inference_mode()
    def score(
        self,
        tokens: torch.Tensor,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        return_per_row: bool = False,
    ) -> tuple[float, int] | tuple[list[float], list[int]]:
        """Score sparse targets from an empty cache in one terminally padded batch.

        This no-prefix entry point is required for tokenizer contracts such as
        Qwen3's, where adding a synthetic BOS token would change the language-
        modeling objective.
        """

        return self._score(
            tokens,
            targets_by_position,
            attention_mask=attention_mask,
            return_per_row=return_per_row,
        )

    @torch.inference_mode()
    def score_target_losses(
        self,
        tokens: torch.Tensor,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
    ) -> list[float]:
        """Return one NLL per sparse target without replaying duplicate input rows.

        Targets are returned in ascending position order and retain their list order
        within each position.  This lets multiple-choice evaluators score several
        one-token answers from one shared prompt state.
        """

        return self._score(
            tokens,
            targets_by_position,
            attention_mask=attention_mask,
            return_per_row=False,
            return_target_losses=True,
        )

    @torch.inference_mode()
    def score_from_snapshot(
        self,
        tokens: torch.Tensor,
        snapshot: CUDAPrefillSnapshot,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        return_per_row: bool = False,
    ) -> tuple[float, int] | tuple[list[float], list[int]]:
        """Score sparse teacher-forced targets after a shared prefix snapshot."""

        return self._score(
            tokens,
            targets_by_position,
            attention_mask=attention_mask,
            return_per_row=return_per_row,
            snapshot=snapshot,
        )

    def _score(
        self,
        tokens: torch.Tensor,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        return_per_row: bool,
        snapshot: CUDAPrefillSnapshot | None = None,
        return_target_losses: bool = False,
    ) -> tuple[float, int] | tuple[list[float], list[int]] | list[float]:
        """Shared implementation for prefixed and true no-BOS scoring."""

        if return_per_row and return_target_losses:
            raise ValueError("per-row and per-target scoring outputs are mutually exclusive")

        device = next(self.model.parameters()).device
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("batched scoring requires tokens with shape [batch, sequence]")
        if snapshot is None:
            cache = pending = None
            prefix_length = 0
        else:
            cache, pending = self.restore(snapshot, batch_size=tokens.shape[0])
            prefix_length = cache.get_seq_length()
        attention_mask = attention_mask.to(device=device)
        if attention_mask.shape != (tokens.shape[0], prefix_length + tokens.shape[1]):
            raise ValueError("scoring attention mask must cover the cached prefix and continuation")
        if bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
            raise ValueError("scoring masks cannot contain a real token after terminal padding begins")
        use_mask_free_attention = self.mask_free_unpadded and bool(attention_mask.all())

        total_nll = torch.zeros((), dtype=torch.float32, device=device)
        target_count = 0
        row_nll = torch.zeros(tokens.shape[0], dtype=torch.float32, device=device)
        row_counts = [0] * tokens.shape[0]
        dense_rows = list(range(tokens.shape[0]))
        dense_targets = None
        if len(targets_by_position) == tokens.shape[1] and all(
            position in targets_by_position and targets_by_position[position][0] == dense_rows
            for position in range(tokens.shape[1])
        ):
            dense_targets = torch.tensor(
                [targets_by_position[position][1] for position in range(tokens.shape[1])],
                dtype=torch.long,
                device=device,
            )
            dense_row_indices = torch.arange(tokens.shape[0], dtype=torch.long, device=device)
        projection_hidden = []
        projection_rows = []
        projection_targets = []
        target_loss_batches = []

        def flush_projection():
            nonlocal target_count
            if not projection_hidden:
                return
            hidden = torch.cat(projection_hidden, dim=0)
            rows = torch.cat(projection_rows, dim=0)
            targets = torch.cat(projection_targets, dim=0)
            logits = project_causal_lm_logits(self.model, hidden).float()
            losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]
            if return_target_losses:
                target_loss_batches.append(losses)
            total_nll.add_(losses.sum())
            row_nll.index_add_(0, rows, losses)
            target_count += targets.numel()
            projection_hidden.clear()
            projection_rows.clear()
            projection_targets.clear()

        self._attach_controller_state(pending, prefix_length)
        try:
            for position in range(tokens.shape[1]):
                prefix_mask = attention_mask[:, : prefix_length + position + 1]
                if cache is not None:
                    self.controller._run_pending_top_stack_iteration(cache, prefix_mask)
                output = self.decoder(
                    input_ids=tokens[:, position : position + 1],
                    attention_mask=None if use_mask_free_attention else prefix_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                self.controller._trim_recorded_upper_cache(cache)
                selected = targets_by_position.get(position)
                if selected is not None:
                    if dense_targets is not None:
                        rows = dense_row_indices
                        targets = dense_targets[position]
                        row_values = dense_rows
                    else:
                        row_values, target_values = selected
                        rows = torch.tensor(row_values, dtype=torch.long, device=device)
                        targets = torch.tensor(target_values, dtype=torch.long, device=device)
                    projection_hidden.append(output.last_hidden_state[rows, -1])
                    projection_rows.append(rows)
                    projection_targets.append(targets)
                    for row in row_values:
                        row_counts[row] += 1
                    if len(projection_hidden) == self.projection_chunk_tokens:
                        flush_projection()
            flush_projection()
            if return_target_losses:
                if not target_loss_batches:
                    return []
                return torch.cat(target_loss_batches).tolist()
            if return_per_row:
                return row_nll.tolist(), row_counts
            return float(total_nll), target_count
        finally:
            self.controller.detach()


class Qwen3DualTokenLayer:
    """Run accuracy-gated eager Qwen3, Llama, or Gemma 3 replay/current layers."""

    @staticmethod
    def supports(model) -> bool:
        config = getattr(model, "config", None)
        return (
            getattr(config, "model_type", None) in {"qwen3", "llama", "gemma3_text"}
            and getattr(config, "_attn_implementation", "eager") == "eager"
        )

    def __call__(
        self,
        layer,
        replay: torch.Tensor,
        current: torch.Tensor,
        replay_position_embeddings,
        current_position_embeddings,
        cache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if replay.shape != current.shape or replay.ndim != 3 or replay.shape[1] != 1:
            raise ValueError("paired eager layer inputs must have shape [batch, 1, hidden_size]")
        is_gemma3 = getattr(getattr(layer, "config", None), "model_type", None) == "gemma3_text"
        if is_gemma3:
            # Gemma 3 1B is natively BF16. Concatenating replay/current before
            # its large projections changes GEMM tiling enough to exceed the
            # repository's 2e-3 accumulated-forward gate on the real model.
            # Keep candidate rows batched, but preserve the checkpoint's exact
            # decoder-layer call boundaries for the two token positions.
            replay = layer(
                replay,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=replay_position_embeddings,
            )
            current = layer(
                current,
                attention_mask=None,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=current_position_embeddings,
            )
            return replay, current
        batch_size = replay.shape[0]
        attention = layer.self_attn
        paired_norm = torch.cat((layer.input_layernorm(replay), layer.input_layernorm(current)), dim=0)

        def paired_projection(projection):
            values = projection(paired_norm)
            return values[:batch_size], values[batch_size:]

        replay_q, current_q = paired_projection(attention.q_proj)
        replay_k, current_k = paired_projection(attention.k_proj)
        replay_v, current_v = paired_projection(attention.v_proj)
        input_shape = replay.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)

        def prepare(q, k, v, position_embeddings):
            q = q.view(hidden_shape)
            k = k.view(hidden_shape)
            if hasattr(attention, "q_norm"):
                q = attention.q_norm(q)
            if hasattr(attention, "k_norm"):
                k = attention.k_norm(k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.view(hidden_shape).transpose(1, 2)
            return (*apply_rotary_pos_emb(q, k, *position_embeddings), v)

        replay_q, replay_k, replay_v = prepare(
            replay_q, replay_k, replay_v, replay_position_embeddings
        )
        current_q, current_k, current_v = prepare(
            current_q, current_k, current_v, current_position_embeddings
        )
        replay_keys, replay_values = cache.update(replay_k, replay_v, attention.layer_idx)
        attention_kwargs = {
            "scaling": attention.scaling,
            "dropout": 0.0,
        }
        if getattr(attention, "attn_logit_softcapping", None) is not None:
            attention_kwargs["softcap"] = attention.attn_logit_softcapping
        replay_attention, _ = eager_attention_forward(
            attention,
            replay_q,
            replay_keys,
            replay_values,
            None,
            **attention_kwargs,
        )
        current_keys, current_values = cache.update(current_k, current_v, attention.layer_idx)
        current_attention, _ = eager_attention_forward(
            attention,
            current_q,
            current_keys,
            current_values,
            None,
            **attention_kwargs,
        )
        paired_attention = torch.cat(
            (
                replay_attention.reshape(*input_shape, -1),
                current_attention.reshape(*input_shape, -1),
            ),
            dim=0,
        )
        paired_attention = attention.o_proj(paired_attention)
        replay_residual = replay + paired_attention[:batch_size]
        current_residual = current + paired_attention[batch_size:]
        paired_mlp_norm = torch.cat(
            (
                layer.post_attention_layernorm(replay_residual),
                layer.post_attention_layernorm(current_residual),
            ),
            dim=0,
        )
        paired_mlp = layer.mlp.down_proj(
            F.silu(layer.mlp.gate_proj(paired_mlp_norm)) * layer.mlp.up_proj(paired_mlp_norm)
        )
        return replay_residual + paired_mlp[:batch_size], current_residual + paired_mlp[batch_size:]


class CUDAConcurrentRunner:
    """Overlap the previous upper replay with the current token's lower stack.

    The two branches use disjoint decoder layers and KV-cache entries. Persistent
    Python workers enqueue them onto separate CUDA streams, then the streams join
    before the current token enters the upper stack. This preserves first-pass
    readout while implementing the paper's two-stack decode schedule.
    """

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        *,
        fused: bool = True,
        stream_priority: int = -3,
        use_python_threads: bool = True,
        dual_gemm: bool = False,
    ):
        if not hasattr(model, "get_decoder") or model.get_output_embeddings() is None:
            raise TypeError("concurrent CUDA inference requires a Hugging Face causal language model")
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise ValueError("concurrent CUDA inference requires a CUDA model")
        self.model = model
        self.decoder = model.get_decoder()
        self.output_embeddings = model.get_output_embeddings()
        self.config = config
        if not 0 <= config.destination_layer < config.source_layer < len(self.decoder.layers):
            raise ValueError("recirculation layers are outside the decoder or are not ordered destination < source")
        self.mixer = FusedNormMix() if fused else mix_reference
        self.device = device
        self.stream_priority = stream_priority
        self.use_python_threads = use_python_threads
        if dual_gemm and not Qwen3DualTokenLayer.supports(model):
            raise ValueError("dual-GEMM upper stacks require eager Qwen3, Llama, or Gemma 3")
        self.dual_token_layer = Qwen3DualTokenLayer() if dual_gemm else None
        self.lower_stream = torch.cuda.Stream(device=device, priority=stream_priority)
        self.replay_stream = torch.cuda.Stream(device=device, priority=stream_priority)
        self.dependency_event = torch.cuda.Event()
        self.lower_done_event = torch.cuda.Event()
        self.replay_done_event = torch.cuda.Event()
        self.executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="recirculation-cuda")
            if use_python_threads
            else None
        )
        self.gil_enabled = log_concurrency_mode()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _mix(self, pending: CUDARecirculationState) -> torch.Tensor:
        alpha, beta = _mixing_coefficients(self.config, pending.input_step)
        return self.mixer(
            pending.destination_residual,
            pending.source_residual,
            alpha,
            beta,
            self.config.normalize_source,
        )

    def _position(self, hidden: torch.Tensor, token_position: int):
        position_ids = torch.full(
            (hidden.shape[0], 1), token_position, dtype=torch.long, device=self.device
        )
        return position_ids, _decoder_position_embeddings(self.decoder, hidden, position_ids)

    def _run_lower(self, token: torch.Tensor, cache, token_position: int):
        hidden = self.decoder.embed_tokens(token)
        position_ids, position_embeddings = self._position(hidden, token_position)
        for index, layer in enumerate(self.decoder.layers[: self.config.destination_layer + 1]):
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=_layer_position_embeddings(
                    self.decoder, position_embeddings, index
                ),
            )
        return hidden, position_ids, position_embeddings

    def _run_upper(self, hidden, cache, position_ids, position_embeddings, *, capture_source: bool):
        source = None
        first_upper_layer = self.config.destination_layer + 1
        for index, layer in enumerate(self.decoder.layers[first_upper_layer:], start=first_upper_layer):
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=_layer_position_embeddings(
                    self.decoder, position_embeddings, index
                ),
            )
            if capture_source and index == self.config.source_layer:
                source = hidden
        if capture_source and source is None:
            raise RuntimeError("source activation was not captured in the current upper stack")
        return hidden, source

    def _run_paired_upper(
        self,
        replay,
        current,
        cache,
        replay_position_embeddings,
        current_position_embeddings,
    ):
        source = None
        first_upper_layer = self.config.destination_layer + 1
        for index, layer in enumerate(self.decoder.layers[first_upper_layer:], start=first_upper_layer):
            replay, current = self.dual_token_layer(
                layer,
                replay,
                current,
                _layer_position_embeddings(self.decoder, replay_position_embeddings, index),
                _layer_position_embeddings(self.decoder, current_position_embeddings, index),
                cache,
            )
            if index == self.config.source_layer:
                source = current
        if source is None:
            raise RuntimeError("source activation was not captured in the paired upper stack")
        return current, source

    def _replay_branch(self, pending, cache, dependency):
        with torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(self.replay_stream):
            self.replay_stream.wait_event(dependency)
            first_upper_layer = self.config.destination_layer + 1
            for layer_cache in cache.layers[first_upper_layer:]:
                layer_cache.crop(-1)
            hidden = self._mix(pending)
            position_ids, position_embeddings = self._position(hidden, pending.input_step)
            self._run_upper(
                hidden,
                cache,
                position_ids,
                position_embeddings,
                capture_source=False,
            )

    def _lower_branch(self, token, cache, token_position, dependency):
        with torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(self.lower_stream):
            self.lower_stream.wait_event(dependency)
            return self._run_lower(token, cache, token_position)

    @torch.inference_mode()
    def step(
        self,
        token: torch.Tensor,
        cache=None,
        pending: CUDARecirculationState | None = None,
        *,
        project_logits: bool = True,
        return_hidden: bool = False,
    ):
        """Process one token, overlapping its lower stack with the pending upper replay."""

        token = token.to(device=self.device, dtype=torch.long)
        if token.ndim == 1:
            token = token.unsqueeze(0)
        if token.ndim != 2 or token.shape[1] != 1 or token.shape[0] < 1:
            raise ValueError("step requires tokens with shape [batch, 1]")
        cache = _new_recirculation_cache(self.model) if cache is None else cache
        first_upper_layer = self.config.destination_layer + 1
        _activate_upper_cache_recording(cache, first_upper_layer)
        token_position = cache.get_seq_length()
        main_stream = torch.cuda.current_stream(self.device)

        if pending is None:
            if token_position != 0:
                raise ValueError("a non-empty cache requires pending recirculation state")
            destination, position_ids, position_embeddings = self._run_lower(token, cache, token_position)
        elif self.dual_token_layer is not None:
            if pending.input_step != token_position - 1:
                raise ValueError("pending input step does not precede the current cache position")
            destination, position_ids, position_embeddings = self._run_lower(token, cache, token_position)
            first_upper_layer = self.config.destination_layer + 1
            for layer_cache in cache.layers[first_upper_layer:]:
                layer_cache.crop(-1)
            replay = self._mix(pending)
            _, replay_position_embeddings = self._position(replay, pending.input_step)
            hidden, source = self._run_paired_upper(
                replay,
                destination,
                cache,
                replay_position_embeddings,
                position_embeddings,
            )
        else:
            if pending.input_step != token_position - 1:
                raise ValueError("pending input step does not precede the current cache position")
            self.dependency_event.record(main_stream)
            if self.executor is None:
                self._replay_branch(pending, cache, self.dependency_event)
                destination, position_ids, position_embeddings = self._lower_branch(
                    token, cache, token_position, self.dependency_event
                )
            else:
                replay_future = self.executor.submit(self._replay_branch, pending, cache, self.dependency_event)
                lower_future = self.executor.submit(
                    self._lower_branch, token, cache, token_position, self.dependency_event
                )
                destination, position_ids, position_embeddings = lower_future.result()
                replay_future.result()
            self.lower_done_event.record(self.lower_stream)
            self.replay_done_event.record(self.replay_stream)
            main_stream.wait_event(self.lower_done_event)
            main_stream.wait_event(self.replay_done_event)

        if pending is None or self.dual_token_layer is None:
            hidden, source = self._run_upper(
                destination,
                cache,
                position_ids,
                position_embeddings,
                capture_source=True,
            )
        next_pending = CUDARecirculationState(destination.detach(), source.detach(), token_position)
        _trim_recorded_upper_cache(cache, first_upper_layer)
        if return_hidden:
            return self.decoder.norm(hidden), cache, next_pending
        if not project_logits:
            return None, cache, next_pending
        logits = project_causal_lm_logits(self.model, self.decoder.norm(hidden))
        return logits, cache, next_pending

    @torch.inference_mode()
    def score(
        self,
        tokens: torch.Tensor,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        return_per_row: bool = False,
        projection_chunk_tokens: int = 1,
    ) -> tuple[float, int] | tuple[list[float], list[int]]:
        """Score an unpadded batch through the explicit two-stream scheduler."""

        tokens = tokens.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("batched scoring requires tokens with shape [batch, sequence]")
        if attention_mask.shape != tokens.shape or not bool(attention_mask.all()):
            raise ValueError("concurrent batched scoring requires a fully unpadded attention mask")
        if projection_chunk_tokens < 1:
            raise ValueError("projection_chunk_tokens must be positive")

        batch_size = tokens.shape[0]
        dense_rows = list(range(batch_size))
        dense_targets = None
        if len(targets_by_position) == tokens.shape[1] and all(
            position in targets_by_position and targets_by_position[position][0] == dense_rows
            for position in range(tokens.shape[1])
        ):
            dense_targets = torch.tensor(
                [targets_by_position[position][1] for position in range(tokens.shape[1])],
                dtype=torch.long,
                device=self.device,
            )
            dense_row_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)

        total_nll = torch.zeros((), dtype=torch.float32, device=self.device)
        target_count = 0
        row_nll = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        row_counts = [0] * batch_size
        projection_hidden = []
        projection_rows = []
        projection_targets = []

        def flush_projection():
            nonlocal target_count
            if not projection_hidden:
                return
            hidden = torch.cat(projection_hidden, dim=0)
            rows = torch.cat(projection_rows, dim=0)
            targets = torch.cat(projection_targets, dim=0)
            logits = project_causal_lm_logits(self.model, hidden).float()
            losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]
            total_nll.add_(losses.sum())
            row_nll.index_add_(0, rows, losses)
            target_count += targets.numel()
            projection_hidden.clear()
            projection_rows.clear()
            projection_targets.clear()

        cache = pending = None
        for position in range(tokens.shape[1]):
            hidden, cache, pending = self.step(
                tokens[:, position : position + 1],
                cache,
                pending,
                project_logits=False,
                return_hidden=True,
            )
            selected = targets_by_position.get(position)
            if selected is None:
                continue
            if dense_targets is not None:
                rows = dense_row_indices
                targets = dense_targets[position]
                row_values = dense_rows
            else:
                row_values, target_values = selected
                rows = torch.tensor(row_values, dtype=torch.long, device=self.device)
                targets = torch.tensor(target_values, dtype=torch.long, device=self.device)
            projection_hidden.append(hidden[rows, -1])
            projection_rows.append(rows)
            projection_targets.append(targets)
            for row in row_values:
                row_counts[row] += 1
            if len(projection_hidden) == projection_chunk_tokens:
                flush_projection()
        flush_projection()
        if return_per_row:
            return row_nll.tolist(), row_counts
        return float(total_nll), target_count

    @torch.inference_mode()
    def prefill(
        self,
        tokens: Sequence[int] | torch.Tensor,
        *,
        collect_logits: bool = False,
        cache=None,
        pending: CUDARecirculationState | None = None,
    ):
        """Serially prefill while overlapping the two independent stacks at each step."""

        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(list(tokens), dtype=torch.long, device=self.device)
        tokens = tokens.to(device=self.device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] == 0:
            raise ValueError("prefill requires tokens with shape [sequence] or [batch, sequence]")
        logits = None
        collected = []
        for position in range(tokens.shape[1]):
            project_logits = collect_logits or position == tokens.shape[1] - 1
            logits, cache, pending = self.step(
                tokens[:, position : position + 1],
                cache,
                pending,
                project_logits=project_logits,
            )
            if logits is not None:
                collected.append(logits)
        return logits, cache, pending, torch.cat(collected, dim=1)

    def snapshot(self, cache, pending: CUDARecirculationState) -> CUDAPrefillSnapshot:
        return _snapshot_cuda_state(cache, pending)

    def restore(self, snapshot: CUDAPrefillSnapshot):
        cache = _new_recirculation_cache(self.model, snapshot.cache_data)
        return cache, snapshot.pending

    def prefill_from_snapshot(
        self,
        tokens: Sequence[int] | torch.Tensor,
        snapshot: CUDAPrefillSnapshot,
        *,
        collect_logits: bool = False,
    ):
        cache, pending = self.restore(snapshot)
        return self.prefill(tokens, cache=cache, pending=pending, collect_logits=collect_logits)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate with first-pass readout and overlapped two-stack decode."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        logits, cache, pending, _ = self.prefill(input_ids)
        prompt = input_ids.to(device=self.device, dtype=torch.long)
        batch_size, prompt_length = prompt.shape
        # Keep the decode buffer on CUDA and fill in place.  Repeated cat()
        # creates and copies an ever-growing tensor at every token.
        generated = torch.empty(
            (batch_size, prompt_length + max_new_tokens),
            dtype=torch.long,
            device=self.device,
        )
        generated[:, :prompt_length] = prompt
        if max_new_tokens == 0:
            return generated[:, :prompt_length]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        for offset in range(max_new_tokens):
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if eos_token_id is not None:
                # Keep completed rows inert while the remaining rows continue. Repeating EOS is
                # removed by special-token decoding and preserves each row's single-item output.
                token = torch.where(finished[:, None], eos_token_id, token)
                finished |= token[:, 0] == eos_token_id
            generated[:, prompt_length + offset : prompt_length + offset + 1] = token
            if eos_token_id is not None and bool(finished.all()):
                break
            logits, cache, pending = self.step(token, cache, pending)
        return generated[:, : prompt_length + offset + 1]


class CUDABatchedPathRunner:
    """Score several same-destination recirculation paths in one CUDA batch.

    Candidate paths occupy contiguous slices of the model batch.  Requiring a
    shared destination gives every candidate the same lower/upper stack split,
    so weights are dispatched once per layer while source residuals and KV
    cache rows remain independent.  This is mathematically the same operation
    as running one :class:`CUDAConcurrentRunner` per candidate with dual GEMMs.
    """

    def __init__(self, model, configs: Sequence[RecirculationConfig]):
        configs = tuple(configs)
        if not configs:
            raise ValueError("candidate batching requires at least one configuration")
        destination = configs[0].destination_layer
        mixing = (
            configs[0].alpha,
            configs[0].beta,
            configs[0].normalize_source,
            configs[0].ramp_tokens,
        )
        if any(config.destination_layer != destination for config in configs):
            raise ValueError("batched candidates must share a destination layer")
        if any(
            (config.alpha, config.beta, config.normalize_source, config.ramp_tokens) != mixing
            for config in configs
        ):
            raise ValueError("batched candidates must share alpha, beta, normalization, and ramp")
        if configs[0].ramp_tokens:
            raise ValueError("candidate batching does not yet support alpha ramping")
        if not Qwen3DualTokenLayer.supports(model):
            raise ValueError("candidate batching requires eager Qwen3, Llama, or Gemma 3")
        self.model = model
        self.decoder = model.get_decoder()
        self.output_embeddings = model.get_output_embeddings()
        self.configs = configs
        self.destination_layer = destination
        self.device = next(model.parameters()).device
        self.mixer = FusedNormMix()
        self.dual_token_layer = Qwen3DualTokenLayer()

    def _position(self, hidden: torch.Tensor, token_position: int):
        position_ids = torch.full(
            (hidden.shape[0], 1), token_position, dtype=torch.long, device=self.device
        )
        return position_ids, _decoder_position_embeddings(self.decoder, hidden, position_ids)

    def _run_lower(self, token: torch.Tensor, cache, token_position: int):
        hidden = self.decoder.embed_tokens(token)
        position_ids, position_embeddings = self._position(hidden, token_position)
        for index, layer in enumerate(self.decoder.layers[: self.destination_layer + 1]):
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=_layer_position_embeddings(
                    self.decoder, position_embeddings, index
                ),
            )
        return hidden, position_ids, position_embeddings

    def _capture_sources(self, hidden, layer_index: int, row_batch_size: int, sources):
        for candidate_index, config in enumerate(self.configs):
            if config.source_layer == layer_index:
                start = candidate_index * row_batch_size
                sources[candidate_index] = hidden[start : start + row_batch_size]

    def _run_upper(
        self,
        hidden,
        cache,
        position_ids,
        position_embeddings,
        row_batch_size: int,
    ):
        sources = [None] * len(self.configs)
        first_upper_layer = self.destination_layer + 1
        for index, layer in enumerate(self.decoder.layers[first_upper_layer:], start=first_upper_layer):
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=_layer_position_embeddings(
                    self.decoder, position_embeddings, index
                ),
            )
            self._capture_sources(hidden, index, row_batch_size, sources)
        if any(source is None for source in sources):
            raise RuntimeError("one or more batched source activations were not captured")
        return hidden, torch.cat(sources, dim=0)

    def _run_paired_upper(
        self,
        replay,
        current,
        cache,
        replay_position_embeddings,
        current_position_embeddings,
        row_batch_size: int,
    ):
        sources = [None] * len(self.configs)
        first_upper_layer = self.destination_layer + 1
        for index, layer in enumerate(self.decoder.layers[first_upper_layer:], start=first_upper_layer):
            replay, current = self.dual_token_layer(
                layer,
                replay,
                current,
                _layer_position_embeddings(self.decoder, replay_position_embeddings, index),
                _layer_position_embeddings(self.decoder, current_position_embeddings, index),
                cache,
            )
            self._capture_sources(current, index, row_batch_size, sources)
        if any(source is None for source in sources):
            raise RuntimeError("one or more batched source activations were not captured")
        return current, torch.cat(sources, dim=0)

    @torch.inference_mode()
    def step(self, token, cache=None, pending: CUDARecirculationState | None = None):
        token = token.to(device=self.device, dtype=torch.long)
        if token.ndim != 2 or token.shape[1] != 1:
            raise ValueError("batched path step requires tokens with shape [candidate*row, 1]")
        if token.shape[0] % len(self.configs):
            raise ValueError("token batch must be divisible by the candidate count")
        row_batch_size = token.shape[0] // len(self.configs)
        cache = _new_recirculation_cache(self.model) if cache is None else cache
        first_upper_layer = self.destination_layer + 1
        _activate_upper_cache_recording(cache, first_upper_layer)
        token_position = cache.get_seq_length()
        destination, position_ids, position_embeddings = self._run_lower(token, cache, token_position)
        if pending is None:
            if token_position != 0:
                raise ValueError("a non-empty cache requires pending recirculation state")
            hidden, source = self._run_upper(
                destination,
                cache,
                position_ids,
                position_embeddings,
                row_batch_size,
            )
        else:
            if pending.input_step != token_position - 1:
                raise ValueError("pending input step does not precede the current cache position")
            for layer_cache in cache.layers[self.destination_layer + 1 :]:
                layer_cache.crop(-1)
            config = self.configs[0]
            replay = self.mixer(
                pending.destination_residual,
                pending.source_residual,
                config.alpha,
                config.beta,
                config.normalize_source,
            )
            _, replay_position_embeddings = self._position(replay, pending.input_step)
            hidden, source = self._run_paired_upper(
                replay,
                destination,
                cache,
                replay_position_embeddings,
                position_embeddings,
                row_batch_size,
            )
        next_pending = CUDARecirculationState(destination.detach(), source.detach(), token_position)
        _trim_recorded_upper_cache(cache, first_upper_layer)
        return self.decoder.norm(hidden), cache, next_pending

    @torch.inference_mode()
    def score(
        self,
        tokens: torch.Tensor,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        projection_chunk_tokens: int = 1,
    ) -> tuple[list[list[float]], list[list[int]]]:
        """Return per-candidate, per-row NLL totals and target counts."""

        tokens = tokens.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("batched scoring requires tokens with shape [row, sequence]")
        if attention_mask.shape != tokens.shape or not bool(attention_mask.all()):
            raise ValueError("candidate-batched scoring requires a fully unpadded attention mask")
        if projection_chunk_tokens < 1:
            raise ValueError("projection_chunk_tokens must be positive")

        if (
            getattr(self.model.config, "model_type", None) == "gemma3_text"
            and len(self.configs) > 1
        ):
            # Expanding candidate paths in Gemma 3's BF16 batch dimension
            # changes cuBLAS tiling enough to fail the real-checkpoint 2e-3
            # accumulated-forward gate. Preserve the batched API and exact
            # search results until a shape-stable Gemma kernel is available.
            LOG.info.once(
                "Gemma 3 candidate width >1 uses the accuracy-gated serial fallback; "
                "Llama and Qwen3 retain fused in-process candidate batching."
            )
            candidate_nll = []
            candidate_counts = []
            for config in self.configs:
                runner = CUDAPrefillRunner(
                    self.model,
                    config,
                    fused=True,
                    allow_terminal_padding=False,
                    projection_chunk_tokens=projection_chunk_tokens,
                    mask_free_unpadded=True,
                )
                nll, counts = runner.score(
                    tokens,
                    targets_by_position,
                    attention_mask=attention_mask,
                    return_per_row=True,
                )
                candidate_nll.append(nll)
                candidate_counts.append(counts)
            return candidate_nll, candidate_counts

        candidate_count = len(self.configs)
        row_batch_size, sequence_length = tokens.shape
        expanded_tokens = tokens.repeat(candidate_count, 1)
        expanded_rows = candidate_count * row_batch_size
        row_nll = torch.zeros(expanded_rows, dtype=torch.float32, device=self.device)
        row_counts = [0] * expanded_rows
        projection_hidden = []
        projection_rows = []
        projection_targets = []

        def flush_projection():
            if not projection_hidden:
                return
            hidden = torch.cat(projection_hidden, dim=0)
            rows = torch.cat(projection_rows, dim=0)
            targets = torch.cat(projection_targets, dim=0)
            logits = project_causal_lm_logits(self.model, hidden).float()
            losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]
            row_nll.index_add_(0, rows, losses)
            projection_hidden.clear()
            projection_rows.clear()
            projection_targets.clear()

        cache = pending = None
        for position in range(sequence_length):
            hidden, cache, pending = self.step(
                expanded_tokens[:, position : position + 1], cache, pending
            )
            selected = targets_by_position.get(position)
            if selected is None:
                continue
            row_values, target_values = selected
            base_rows = torch.tensor(row_values, dtype=torch.long, device=self.device)
            base_targets = torch.tensor(target_values, dtype=torch.long, device=self.device)
            rows = torch.cat(
                [base_rows + candidate_index * row_batch_size for candidate_index in range(candidate_count)]
            )
            targets = base_targets.repeat(candidate_count)
            projection_hidden.append(hidden[rows, -1])
            projection_rows.append(rows)
            projection_targets.append(targets)
            for candidate_index in range(candidate_count):
                offset = candidate_index * row_batch_size
                for row in row_values:
                    row_counts[offset + row] += 1
            if len(projection_hidden) == projection_chunk_tokens:
                flush_projection()
        flush_projection()
        nll_values = row_nll.view(candidate_count, row_batch_size).tolist()
        count_values = [
            row_counts[start : start + row_batch_size]
            for start in range(0, expanded_rows, row_batch_size)
        ]
        return nll_values, count_values

class CUDAGraphedConcurrentPrefill:
    """Capture the two-stream concurrent prefill as one fixed-shape CUDA Graph.

    CUDA Graph capture is process-global with respect to unsafe CUDA activity.
    A process-wide lock serializes warmup and capture across instances. The
    runner's lower and replay streams fork from the capture stream and rejoin it
    at every token boundary, so both paths are included in the same graph.
    """

    def __init__(
        self,
        runner: CUDAConcurrentRunner,
        example_tokens: torch.Tensor,
        *,
        warmups: int = 3,
        capture_error_mode: str = "global",
        keep_graph: bool = False,
        capture_stream_priority: int = 0,
        max_tokens: int = 256,
    ):
        if warmups < 0:
            raise ValueError("concurrent CUDA graph warmups must be non-negative")
        if capture_error_mode not in {"global", "thread_local", "relaxed"}:
            raise ValueError("capture_error_mode must be global, thread_local, or relaxed")
        if runner.config.ramp_tokens:
            raise ValueError(
                "concurrent CUDA graph replay with ramp_tokens is disabled until changed-input accuracy is gated"
            )
        if example_tokens.ndim == 1:
            example_tokens = example_tokens.unsqueeze(0)
        if example_tokens.ndim != 2 or example_tokens.shape[0] != 1 or example_tokens.shape[1] == 0:
            raise ValueError("concurrent graph capture requires tokens with shape [sequence] or [1, sequence]")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if example_tokens.shape[1] > max_tokens:
            raise ValueError(
                f"concurrent graph capture is limited to {max_tokens} tokens; use eager or a larger bounded graph limit"
            )
        self.runner = runner
        self.static_tokens = example_tokens.to(device=runner.device, dtype=torch.long).clone()
        self.capture_error_mode = capture_error_mode
        self.keep_graph = keep_graph
        self.capture_stream_priority = capture_stream_priority
        self.capture_stream = torch.cuda.Stream(device=runner.device, priority=capture_stream_priority)
        self.graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)

        LOG.info.once(
            "Serializing two-stream CUDA Graph warmup and capture with the process-wide capture lock."
        )
        with _CUDA_GRAPH_CAPTURE_LOCK:
            current_stream = torch.cuda.current_stream(runner.device)
            self.capture_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.capture_stream):
                for _ in range(warmups):
                    self.outputs = runner.prefill(self.static_tokens)
            current_stream.wait_stream(self.capture_stream)
            torch.cuda.synchronize(runner.device)
            with torch.cuda.graph(
                self.graph,
                stream=self.capture_stream,
                capture_error_mode=capture_error_mode,
            ):
                self.outputs = runner.prefill(self.static_tokens)
            if keep_graph:
                self.graph.instantiate()
            torch.cuda.synchronize(runner.device)

    @torch.inference_mode()
    def prefill(self, tokens: Sequence[int] | torch.Tensor):
        """Copy fixed-shape token values and replay both captured CUDA streams."""

        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(list(tokens), dtype=torch.long, device=self.runner.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape != self.static_tokens.shape:
            raise ValueError(f"captured concurrent prefill requires tokens with shape {tuple(self.static_tokens.shape)}")
        self.static_tokens.copy_(tokens.to(device=self.runner.device, dtype=torch.long))
        self.graph.replay()
        return self.outputs


class CUDAGraphedPrefill:
    """Replay a captured fixed-length prefill without per-token CPU dispatch.

    Capture is specific to the model, sequence length, tensor shapes, and CUDA
    device. The static input buffers are updated before every replay, so token
    values and attention-mask values may change without recapturing.
    """

    def __init__(
        self,
        runner: CUDAPrefillRunner,
        example_tokens: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        warmups: int = 3,
    ):
        if warmups < 1:
            raise ValueError("CUDA graph capture requires at least one warmup")
        if runner.controller.config.ramp_tokens:
            raise ValueError(
                "CUDA graph replay with ramp_tokens is disabled because changed-input error exceeds the release gate; "
                "use CUDAPrefillRunner directly"
            )
        if example_tokens.ndim == 1:
            example_tokens = example_tokens.unsqueeze(0)
        if example_tokens.ndim != 2 or example_tokens.shape[0] == 0 or example_tokens.shape[1] == 0:
            raise ValueError("graph capture requires tokens with shape [sequence] or [batch, sequence]")
        device = next(runner.model.parameters()).device
        if device.type != "cuda":
            raise ValueError("CUDA graph capture requires a CUDA model")
        self.runner = runner
        self.static_tokens = example_tokens.to(device=device, dtype=torch.long).clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(self.static_tokens)
        if attention_mask.shape != self.static_tokens.shape:
            raise ValueError("attention_mask must have the same shape as tokens")
        self.static_attention_mask = attention_mask.to(device=device).clone()
        if not bool(self.static_attention_mask.all()):
            raise ValueError("graphed same-token replay currently requires an unpadded batch")

        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmups):
                self.outputs = runner.prefill(self.static_tokens, attention_mask=self.static_attention_mask)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = runner.prefill(self.static_tokens, attention_mask=self.static_attention_mask)

    @torch.inference_mode()
    def prefill(self, tokens: Sequence[int] | torch.Tensor, *, attention_mask: torch.Tensor | None = None):
        """Copy new values into the static buffers and replay the captured prefill."""

        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(list(tokens), dtype=torch.long, device=self.static_tokens.device)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape != self.static_tokens.shape:
            raise ValueError(f"captured prefill requires tokens with shape {tuple(self.static_tokens.shape)}")
        self.static_tokens.copy_(tokens.to(device=self.static_tokens.device, dtype=torch.long))
        if attention_mask is None:
            self.static_attention_mask.fill_(1)
        else:
            if attention_mask.shape != self.static_attention_mask.shape:
                raise ValueError("attention_mask must have the captured token shape")
            if not bool(attention_mask.all()):
                raise ValueError("graphed same-token replay currently requires an unpadded batch")
            self.static_attention_mask.copy_(attention_mask.to(device=self.static_attention_mask.device))
        self.graph.replay()
        return self.outputs


__all__ = [
    "MAX_FORWARD_ERROR",
    "MAX_FUSED_OR_BATCHED_FORWARD_ERROR",
    "CUDABatchedPathRunner",
    "CUDAConcurrentRunner",
    "CUDAGraphedConcurrentPrefill",
    "CUDAGraphedPrefill",
    "CUDAPrefillRunner",
    "CUDAPrefillSnapshot",
    "CUDARecirculationState",
    "ForwardError",
    "FusedNormMix",
    "Qwen3DualTokenLayer",
    "log_concurrency_mode",
    "measure_forward_error",
    "mix_reference",
]
