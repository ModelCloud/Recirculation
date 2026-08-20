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
from logbar import LogBar
from transformers import DynamicCache

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - depends on the CUDA installation
    raise ImportError("The CUDA backend requires Triton. Install the 'cuda' extra.") from error

from .controller import RecirculationConfig, RecirculationController, _mixing_coefficients, torch_mix_reference

MAX_FORWARD_ERROR = 2e-3
LOG = LogBar.shared()
_CUDA_GRAPH_CAPTURE_LOCK = threading.Lock()
# Preserve the existing CUDA-backend API while making the Torch implementation
# the sole eager reference and fallback.
mix_reference = torch_mix_reference


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
class CUDARecirculationState:
    """First-pass activations waiting for their same-token upper replay."""

    destination: torch.Tensor
    source: torch.Tensor
    token_position: int


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
        pending.destination.detach().clone(),
        pending.source.detach().clone(),
        pending.token_position,
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
        self.controller.attach()
        self.controller._pending = pending
        self.controller._position = prefix_length
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
                is_final = position == tokens.shape[1] - 1
                if collect_logits or not self.skip_intermediate_logits or is_final:
                    logits = self.output_embeddings(output.last_hidden_state[:, -1:, :])
                if collect_logits or is_final:
                    collected.append(logits)
            pending = self.controller._pending
            if logits is None or pending is None:  # pragma: no cover - guarded by model/config validation
                raise RuntimeError("prefill did not produce logits and pending replay state")
            state = CUDARecirculationState(
                pending.destination.detach(), pending.source.detach(), pending.token_position
            )
            return logits, cache, state, torch.cat(collected, dim=1)
        finally:
            self.controller.detach()

    def snapshot(self, cache, pending: CUDARecirculationState) -> CUDAPrefillSnapshot:
        return _snapshot_cuda_state(cache, pending)

    def restore(self, snapshot: CUDAPrefillSnapshot, *, batch_size: int = 1):
        cache = DynamicCache(snapshot.cache_data, config=self.model.config)
        snapshot_batch = snapshot.pending.destination.shape[0]
        if batch_size % snapshot_batch:
            raise ValueError("requested batch size must be a multiple of the snapshot batch size")
        repeats = batch_size // snapshot_batch
        pending = snapshot.pending
        if repeats > 1:
            cache.batch_repeat_interleave(repeats)
            pending = CUDARecirculationState(
                pending.destination.repeat_interleave(repeats, dim=0),
                pending.source.repeat_interleave(repeats, dim=0),
                pending.token_position,
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
    def score_from_snapshot(
        self,
        tokens: torch.Tensor,
        snapshot: CUDAPrefillSnapshot,
        targets_by_position: dict[int, tuple[list[int], list[int]]],
        *,
        attention_mask: torch.Tensor,
        return_per_row: bool = False,
    ) -> tuple[float, int] | tuple[list[float], list[int]]:
        """Score sparse teacher-forced targets in one terminally padded batch."""

        device = next(self.model.parameters()).device
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] == 0:
            raise ValueError("batched scoring requires tokens with shape [batch, sequence]")
        cache, pending = self.restore(snapshot, batch_size=tokens.shape[0])
        prefix_length = cache.get_seq_length()
        attention_mask = attention_mask.to(device=device)
        if attention_mask.shape != (tokens.shape[0], prefix_length + tokens.shape[1]):
            raise ValueError("scoring attention mask must cover the cached prefix and continuation")
        if bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
            raise ValueError("scoring masks cannot contain a real token after terminal padding begins")

        total_nll = torch.zeros((), dtype=torch.float32, device=device)
        target_count = 0
        row_nll = torch.zeros(tokens.shape[0], dtype=torch.float32, device=device)
        row_counts = [0] * tokens.shape[0]
        self.controller.attach()
        self.controller._pending = pending
        self.controller._position = prefix_length
        try:
            for position in range(tokens.shape[1]):
                prefix_mask = attention_mask[:, : prefix_length + position + 1]
                self.controller._run_pending_top_stack_iteration(cache, prefix_mask)
                output = self.decoder(
                    input_ids=tokens[:, position : position + 1],
                    attention_mask=prefix_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = output.past_key_values
                selected = targets_by_position.get(position)
                if selected is not None:
                    row_values, target_values = selected
                    rows = torch.tensor(row_values, dtype=torch.long, device=device)
                    targets = torch.tensor(target_values, dtype=torch.long, device=device)
                    logits = self.output_embeddings(output.last_hidden_state[rows, -1]).float()
                    losses = torch.logsumexp(logits, dim=-1) - logits.gather(1, targets[:, None])[:, 0]
                    total_nll += losses.sum()
                    row_nll.index_add_(0, rows, losses)
                    target_count += len(target_values)
                    for row in row_values:
                        row_counts[row] += 1
            if return_per_row:
                return row_nll.tolist(), row_counts
            return float(total_nll), target_count
        finally:
            self.controller.detach()


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
        alpha, beta = _mixing_coefficients(self.config, pending.token_position)
        return self.mixer(
            pending.destination,
            pending.source,
            alpha,
            beta,
            self.config.normalize_source,
        )

    def _position(self, hidden: torch.Tensor, token_position: int):
        position_ids = torch.full(
            (hidden.shape[0], 1), token_position, dtype=torch.long, device=self.device
        )
        return position_ids, self.decoder.rotary_emb(hidden, position_ids=position_ids)

    def _run_lower(self, token: torch.Tensor, cache, token_position: int):
        hidden = self.decoder.embed_tokens(token)
        position_ids, position_embeddings = self._position(hidden, token_position)
        for layer in self.decoder.layers[: self.config.destination_layer + 1]:
            hidden = layer(
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=position_embeddings,
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
                position_embeddings=position_embeddings,
            )
            if capture_source and index == self.config.source_layer:
                source = hidden
        if capture_source and source is None:
            raise RuntimeError("source activation was not captured in the current upper stack")
        return hidden, source

    def _replay_branch(self, pending, cache, dependency):
        with torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(self.replay_stream):
            self.replay_stream.wait_event(dependency)
            first_upper_layer = self.config.destination_layer + 1
            for layer_cache in cache.layers[first_upper_layer:]:
                layer_cache.crop(-1)
            hidden = self._mix(pending)
            position_ids, position_embeddings = self._position(hidden, pending.token_position)
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
    ):
        """Process one token, overlapping its lower stack with the pending upper replay."""

        token = token.to(device=self.device, dtype=torch.long)
        if token.ndim == 1:
            token = token.unsqueeze(0)
        if token.ndim != 2 or token.shape[1] != 1 or token.shape[0] < 1:
            raise ValueError("step requires tokens with shape [batch, 1]")
        cache = DynamicCache(config=self.model.config) if cache is None else cache
        token_position = cache.get_seq_length()
        main_stream = torch.cuda.current_stream(self.device)

        if pending is None:
            if token_position != 0:
                raise ValueError("a non-empty cache requires pending recirculation state")
            destination, position_ids, position_embeddings = self._run_lower(token, cache, token_position)
        else:
            if pending.token_position != token_position - 1:
                raise ValueError("pending token position does not precede the current cache position")
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

        hidden, source = self._run_upper(
            destination,
            cache,
            position_ids,
            position_embeddings,
            capture_source=True,
        )
        next_pending = CUDARecirculationState(destination.detach(), source.detach(), token_position)
        if not project_logits:
            return None, cache, next_pending
        logits = self.output_embeddings(self.decoder.norm(hidden))
        return logits, cache, next_pending

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
        cache = DynamicCache(snapshot.cache_data, config=self.model.config)
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
        for offset in range(max_new_tokens):
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated[:, prompt_length + offset : prompt_length + offset + 1] = token
            if eos_token_id is not None and bool((token == eos_token_id).all()):
                break
            logits, cache, pending = self.step(token, cache, pending)
        return generated[:, : prompt_length + offset + 1]


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
    "CUDAConcurrentRunner",
    "CUDAGraphedConcurrentPrefill",
    "CUDAGraphedPrefill",
    "CUDAPrefillRunner",
    "CUDAPrefillSnapshot",
    "CUDARecirculationState",
    "ForwardError",
    "FusedNormMix",
    "log_concurrency_mode",
    "measure_forward_error",
    "mix_reference",
]
