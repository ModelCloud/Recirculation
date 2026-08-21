# SPDX-License-Identifier: Apache-2.0

"""Repository-owned Transformers continuous-batching patch for recirculation.

The module contains both the initial fixed-slot manager and a recurrent-state
extension for Transformers' native paged cache.  The latter stores destination
and source residuals beside physical KV pages, extends cache copy-on-write, and
drives the model through packed paged-FlashAttention batches.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import MethodType

import torch
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus
from transformers.modeling_outputs import CausalLMOutputWithPast

from .controller import (
    RecirculationConfig,
    _decoder_position_embeddings,
    _layer_position_embeddings,
    project_causal_lm_logits,
)

_TRANSFORMERS_CONFIG_PATCH_LOCK = threading.RLock()


@contextmanager
def patch_transformers_paged_cache_defaults(*, num_blocks: int, block_size: int = 256):
    """Bound native paged-cache allocation when a caller omits cache sizing.

    Evalution exposes the useful scheduler limits but does not currently expose
    ``num_blocks``. Transformers otherwise consumes most free VRAM, even for a
    small fixed-width evaluator. This scoped patch keeps explicit caller values
    intact and restores the class before returning.
    """

    if min(num_blocks, block_size) < 1:
        raise ValueError("paged cache block count and size must be positive")
    from transformers import ContinuousBatchingConfig

    with _TRANSFORMERS_CONFIG_PATCH_LOCK:
        original_init = ContinuousBatchingConfig.__init__

        @wraps(original_init)
        def bounded_init(self, *args, **kwargs):
            kwargs.setdefault("num_blocks", num_blocks)
            kwargs.setdefault("block_size", block_size)
            original_init(self, *args, **kwargs)

        ContinuousBatchingConfig.__init__ = bounded_init
        try:
            yield
        finally:
            ContinuousBatchingConfig.__init__ = original_init


class RecirculationPagedState:
    """Store recurrent residuals at the same physical pages as paged KV state."""

    def __init__(self, cache, hidden_size: int, dtype: torch.dtype) -> None:
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.cache = cache
        self.hidden_size = hidden_size
        physical_pages = int(cache.key_cache[0].shape[0])
        self.destination = torch.empty(
            (physical_pages, hidden_size), dtype=dtype, device=cache.device
        )
        self.source = torch.empty_like(self.destination)
        self.input_steps = torch.full(
            (physical_pages,), -1, dtype=torch.int64, device=cache.device
        )

    def store(
        self,
        physical_indices: torch.Tensor,
        destination: torch.Tensor,
        source: torch.Tensor,
        input_steps: torch.Tensor | int,
    ) -> None:
        indices = physical_indices.to(device=self.destination.device, dtype=torch.long)
        destination = destination.reshape(len(indices), self.hidden_size)
        source = source.reshape(len(indices), self.hidden_size)
        self.destination.index_copy_(0, indices, destination)
        self.source.index_copy_(0, indices, source)
        if isinstance(input_steps, int):
            steps = torch.full_like(indices, input_steps)
        else:
            steps = input_steps.to(device=indices.device, dtype=torch.int64).reshape(-1)
        self.input_steps.index_copy_(0, indices, steps)

    def load(self, physical_indices: torch.Tensor):
        indices = physical_indices.to(device=self.destination.device, dtype=torch.long)
        return (
            self.destination.index_select(0, indices),
            self.source.index_select(0, indices),
            self.input_steps.index_select(0, indices),
        )

    def copy_blocks(self, source_blocks: list[int], destination_blocks: list[int]) -> None:
        if not source_blocks:
            return
        block_size = int(self.cache.block_size)
        source = torch.tensor(source_blocks, device=self.destination.device, dtype=torch.long)
        destination = torch.tensor(destination_blocks, device=self.destination.device, dtype=torch.long)
        source_pages = source[:, None] * block_size + torch.arange(
            block_size, device=source.device
        )[None, :]
        destination_pages = destination[:, None] * block_size + torch.arange(
            block_size, device=destination.device
        )[None, :]
        source_pages = source_pages.reshape(-1)
        destination_pages = destination_pages.reshape(-1)
        self.destination.index_copy_(
            0, destination_pages, self.destination.index_select(0, source_pages)
        )
        self.source.index_copy_(0, destination_pages, self.source.index_select(0, source_pages))
        self.input_steps.index_copy_(
            0, destination_pages, self.input_steps.index_select(0, source_pages)
        )

    def physical_token_indices(self, request_ids: list[str], positions: list[int]) -> torch.Tensor:
        if len(request_ids) != len(positions):
            raise ValueError("request_ids and positions must have equal lengths")
        allocator = self.cache.group_cache_managers[0]
        indices = []
        for request_id, position in zip(request_ids, positions, strict=True):
            if position < 0:
                raise ValueError("logical cache positions must be non-negative")
            block = allocator.block_table[request_id][position // self.cache.block_size]
            indices.append(block * self.cache.block_size + position % self.cache.block_size)
        return torch.tensor(indices, dtype=torch.long, device=self.destination.device)


def make_paged_cache_recirculation_aware(cache, *, hidden_size: int, dtype: torch.dtype):
    """Attach recurrent page state and extend cache copy-on-write to include it."""

    if hasattr(cache, "recirculation_state"):
        return cache.recirculation_state
    if len(cache.group_cache_managers) != 1:
        raise ValueError(
            "paged recirculation currently requires one full-attention cache group"
        )
    state = RecirculationPagedState(cache, hidden_size, dtype)
    original_copy_cache = cache.copy_cache

    def copy_cache_with_recirculation(self, source_blocks, destination_blocks):
        original_copy_cache(source_blocks, destination_blocks)
        state.copy_blocks(source_blocks, destination_blocks)

    cache.copy_cache = MethodType(copy_cache_with_recirculation, cache)
    cache.recirculation_state = state
    return state


def seed_paged_cache_from_snapshot(
    cache,
    *,
    request_id: str,
    prompt_ids: list[int],
    snapshot,
) -> int:
    """Import a block-aligned CUDA-runner prefix into shareable paged storage."""

    if not prompt_ids or len(prompt_ids) % cache.block_size:
        raise ValueError("seeded paged prefixes must contain complete blocks")
    block_count = len(prompt_ids) // cache.block_size
    allocated = cache.allocate_blocks(block_count, request_id, allocated_blocks=0)
    if allocated != block_count:
        raise RuntimeError("paged cache could not allocate the shared prefix")
    allocator = cache.group_cache_managers[0]
    blocks = allocator.block_table[request_id]
    physical = torch.cat(
        [
            torch.arange(
                block * cache.block_size,
                (block + 1) * cache.block_size,
                device=cache.device,
                dtype=torch.long,
            )
            for block in blocks
        ]
    )
    if len(snapshot.cache_data) != len(cache.layer_index_to_group_indices):
        raise ValueError("snapshot and paged cache layer counts differ")
    for layer_index, layer_data in enumerate(snapshot.cache_data):
        key, value = layer_data[:2]
        if key is None or value is None or key.shape[2] != len(prompt_ids):
            raise ValueError("snapshot does not contain a complete fixed-length KV prefix")
        _group, layer_in_group = cache.layer_index_to_group_indices[layer_index]
        cache.key_cache[layer_in_group].index_copy_(0, physical, key[0].transpose(0, 1))
        cache.value_cache[layer_in_group].index_copy_(0, physical, value[0].transpose(0, 1))

    state = make_paged_cache_recirculation_aware(
        cache,
        hidden_size=snapshot.pending.destination_residual.shape[-1],
        dtype=snapshot.pending.destination_residual.dtype,
    )
    state.store(
        physical[-1:],
        snapshot.pending.destination_residual,
        snapshot.pending.source_residual,
        snapshot.pending.input_step,
    )
    cache._block_manager.mark_shareable_blocks_as_complete(
        num_complete_blocks=block_count,
        allocated_blocks=blocks,
        prompt_ids=prompt_ids,
    )
    cache.free_blocks(request_id)
    return block_count


class PagedRecirculationForward:
    """Run tokenwise recirculation over Transformers packed paged-attention batches."""

    def __init__(self, model, config: RecirculationConfig, mixer) -> None:
        if config.ramp_tokens:
            raise ValueError("paged recirculation initially requires ramp_tokens=0")
        self.model = model
        self.decoder = model.get_decoder()
        self.config = config
        self.mixer = mixer
        self.last_logits: torch.Tensor | None = None

    @staticmethod
    def _single_group(value):
        return next(iter(value.values())) if isinstance(value, dict) else value

    def _wave_kwargs(
        self,
        kwargs: dict,
        sequence_indices: list[int],
        positions: torch.Tensor,
        physical_write_indices: torch.Tensor,
        physical_read_indices: list[torch.Tensor] | None,
    ) -> dict:
        device = positions.device
        wave = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "cache",
                "cu_seq_lens_q",
                "cu_seq_lens_k",
                "max_seqlen_q",
                "max_seqlen_k",
                "read_index",
                "write_index",
                "block_table",
                "logits_indices",
                "logits_processor_args",
            }
        }
        count = len(sequence_indices)
        cu_dtype = kwargs["cu_seq_lens_q"].dtype
        wave["cache"] = kwargs["cache"]
        wave["cu_seq_lens_q"] = torch.arange(count + 1, device=device, dtype=cu_dtype)
        wave["max_seqlen_q"] = 1
        kv_lengths = (positions + 1).to(dtype=cu_dtype)
        wave["cu_seq_lens_k"] = torch.cat(
            (torch.zeros(1, device=device, dtype=cu_dtype), kv_lengths.cumsum(0))
        )
        wave["max_seqlen_k"] = int(kv_lengths.max().item())
        wave["write_index"] = [physical_write_indices]

        original_reads = kwargs["read_index"]
        wave_reads = []
        for group_reads in original_reads:
            if physical_read_indices is None:
                wave_reads.append(group_reads[:0])
            else:
                wave_reads.append(torch.cat(physical_read_indices))
        wave["read_index"] = wave_reads
        block_table = kwargs.get("block_table")
        if block_table is not None:
            wave["block_table"] = block_table[:, sequence_indices]
        else:
            wave["block_table"] = None
        return wave

    def _run_layers(self, hidden, position_ids, wave_kwargs, start: int, stop: int):
        position_embeddings = _decoder_position_embeddings(self.decoder, hidden, position_ids)
        for index in range(start, stop):
            hidden = self.decoder.layers[index](
                hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                position_embeddings=_layer_position_embeddings(
                    self.decoder, position_embeddings, index
                ),
                **wave_kwargs,
            )
        return hidden

    @torch.inference_mode()
    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        del attention_mask, past_key_values, labels, use_cache
        if input_ids is None or inputs_embeds is not None:
            raise ValueError("paged recirculation requires packed input_ids")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("paged recirculation expects Transformers packed shape [1, tokens]")
        if position_ids is None or "cache" not in kwargs or "cu_seq_lens_q" not in kwargs:
            raise ValueError("paged recirculation requires cache, position_ids, and cu_seq_lens_q")

        cache = kwargs["cache"]
        state = make_paged_cache_recirculation_aware(
            cache,
            hidden_size=self.model.config.hidden_size,
            dtype=next(self.model.parameters()).dtype,
        )
        cu_q = kwargs["cu_seq_lens_q"]
        sequence_count = len(cu_q) - 1
        lengths = [int((cu_q[index + 1] - cu_q[index]).item()) for index in range(sequence_count)]
        output_hidden = torch.empty(
            (1, input_ids.shape[1], self.model.config.hidden_size),
            dtype=next(self.model.parameters()).dtype,
            device=input_ids.device,
        )
        original_write = kwargs["write_index"][0]
        original_cu_k = self._single_group(kwargs["cu_seq_lens_k"])
        original_read = kwargs["read_index"][0]

        def read_pieces(active_sequences: list[int], through_offsets: list[int]):
            if kwargs.get("block_table") is not None:
                return None
            pieces = []
            for sequence_index, through_offset in zip(
                active_sequences, through_offsets, strict=True
            ):
                query_start = int(cu_q[sequence_index].item())
                prefix_length = int(position_ids[0, query_start].item())
                read_start = int(original_cu_k[sequence_index].item())
                prefix = original_read[read_start : read_start + prefix_length]
                current = original_write[query_start : query_start + through_offset + 1]
                pieces.append(torch.cat((prefix, current)))
            return pieces

        for offset in range(max(lengths)):
            sequence_indices = [index for index, length in enumerate(lengths) if offset < length]
            query_indices = [int(cu_q[index].item()) + offset for index in sequence_indices]
            query_tensor = torch.tensor(query_indices, device=input_ids.device, dtype=torch.long)
            current_positions = position_ids[0].index_select(0, query_tensor)
            block_table = kwargs.get("block_table")
            if block_table is None:
                current_writes = original_write.index_select(0, query_tensor)
            else:
                group_table = block_table[0]
                rows = torch.tensor(sequence_indices, device=input_ids.device, dtype=torch.long)
                blocks = group_table[rows, current_positions // cache.block_size]
                current_writes = blocks * cache.block_size + current_positions % cache.block_size
            current_kwargs = self._wave_kwargs(
                kwargs,
                sequence_indices,
                current_positions,
                current_writes,
                read_pieces(sequence_indices, [offset] * len(sequence_indices)),
            )
            current = self.decoder.embed_tokens(input_ids[:, query_tensor])
            destination = self._run_layers(
                current,
                current_positions.unsqueeze(0),
                current_kwargs,
                0,
                self.config.destination_layer + 1,
            )

            replay_sequences = []
            replay_physical = []
            replay_offsets = []
            for sequence_index, position in zip(
                sequence_indices, current_positions.tolist(), strict=True
            ):
                if position <= 0:
                    continue
                replay_sequences.append(sequence_index)
                replay_offsets.append(offset - 1)
                if block_table is None:
                    if offset > 0:
                        previous_query = int(cu_q[sequence_index].item()) + offset - 1
                        replay_physical.append(original_write[previous_query])
                    else:
                        read_start = int(original_cu_k[sequence_index].item())
                        replay_physical.append(original_read[read_start + position - 1])
                else:
                    block = block_table[0, sequence_index, (position - 1) // cache.block_size]
                    replay_physical.append(block * cache.block_size + (position - 1) % cache.block_size)
            if replay_sequences:
                replay_indices = torch.stack(replay_physical).to(dtype=torch.long)
                replay_destination, replay_source, replay_steps = state.load(replay_indices)
                if not torch.equal(replay_steps, current_positions[current_positions > 0] - 1):
                    raise RuntimeError("paged recurrent state does not precede the current token")
                replay = self.mixer(
                    replay_destination.unsqueeze(0),
                    replay_source.unsqueeze(0),
                    self.config.alpha,
                    self.config.beta,
                    self.config.normalize_source,
                )
                replay_positions = replay_steps
                replay_kwargs = self._wave_kwargs(
                    kwargs,
                    replay_sequences,
                    replay_positions,
                    replay_indices,
                    read_pieces(replay_sequences, replay_offsets),
                )
                self._run_layers(
                    replay,
                    replay_positions.unsqueeze(0),
                    replay_kwargs,
                    self.config.destination_layer + 1,
                    len(self.decoder.layers),
                )

            hidden = destination
            source = None
            position_embeddings = _decoder_position_embeddings(
                self.decoder, hidden, current_positions.unsqueeze(0)
            )
            for index in range(self.config.destination_layer + 1, len(self.decoder.layers)):
                hidden = self.decoder.layers[index](
                    hidden,
                    attention_mask=None,
                    position_ids=current_positions.unsqueeze(0),
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=_layer_position_embeddings(
                        self.decoder, position_embeddings, index
                    ),
                    **current_kwargs,
                )
                if index == self.config.source_layer:
                    source = hidden
            if source is None:
                raise RuntimeError("paged current pass did not capture the source layer")
            state.store(current_writes, destination, source, current_positions)
            output_hidden[:, query_tensor] = self.decoder.norm(hidden)

        if isinstance(logits_to_keep, torch.Tensor):
            output_hidden = output_hidden[:, logits_to_keep]
        elif isinstance(logits_to_keep, int) and logits_to_keep > 0:
            output_hidden = output_hidden[:, -logits_to_keep:]
        logits = project_causal_lm_logits(self.model, output_hidden)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError("paged recirculation produced non-finite logits")
        # Retain the most recent raw projection for the repository's numerical
        # gate.  This is detached and replaced on every scheduler iteration, so
        # it neither preserves an autograd graph nor grows with decode length.
        self.last_logits = logits.detach()
        return CausalLMOutputWithPast(logits=logits)


@contextmanager
def patch_model_paged_recirculation(model, config: RecirculationConfig, mixer):
    """Temporarily replace a causal LM forward with packed paged recirculation."""

    original_forward = model.forward
    implementation = PagedRecirculationForward(model, config, mixer)
    model.forward = implementation
    try:
        yield implementation
    finally:
        model.forward = original_forward


class RecirculationContinuousBatchingManager:
    """Implement the Transformers manager API with fixed recirculation slots."""

    def __init__(
        self,
        model,
        generation_config,
        continuous_batching_config=None,
        workload_hints=None,
        **_kwargs,
    ) -> None:
        del workload_hints
        runner = getattr(model, "_recirculation_runner", None)
        if runner is None:
            raise ValueError("model has no _recirculation_runner attached")
        self.model = model
        self.runner = runner
        self.generation_config = generation_config
        self.config = continuous_batching_config
        self.max_requests = int(
            getattr(continuous_batching_config, "max_requests_per_batch", None) or 32
        )
        self._inputs: queue.Queue[tuple[str, list[int], int]] = queue.Queue()
        self._outputs: queue.Queue[GenerationOutput] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add_request(
        self,
        input_ids: list[int],
        *,
        request_id: str,
        max_new_tokens: int | None = None,
        streaming: bool = False,
        **_kwargs,
    ) -> str:
        if streaming:
            raise ValueError("recirculation fixed-slot manager does not support streaming")
        limit = max_new_tokens
        if limit is None:
            limit = getattr(self.generation_config, "max_new_tokens", None) or 20
        self._inputs.put((request_id, list(map(int, input_ids)), int(limit)))
        return request_id

    def add_requests(self, inputs: list[list[int]], *, max_new_tokens=None, **kwargs) -> list[str]:
        request_ids = []
        for index, input_ids in enumerate(inputs):
            request_id = f"req_{index}"
            self.add_request(
                input_ids,
                request_id=request_id,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            request_ids.append(request_id)
        return request_ids

    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_generation_loop,
            name="recirculation-continuous-batching",
            daemon=True,
        )
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_result(self, timeout: float | None = None):
        try:
            return self._outputs.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(
        self,
        block: bool = True,
        timeout: float | None = None,
        keep_for_next_session: bool = False,
        hard_stop: bool = False,
    ) -> None:
        del keep_for_next_session
        if hard_stop:
            while True:
                try:
                    self._inputs.get_nowait()
                except queue.Empty:
                    break
        self._stop.set()
        if block and self._thread is not None:
            self._thread.join(timeout=timeout)
            if not self._thread.is_alive():
                self._thread = None

    def destroy(self) -> None:
        self.stop(block=True)

    def evict_request_from_cache(self, _request_id: str) -> None:
        return None

    def _drain_ready(self) -> list[tuple[str, list[int], int]]:
        pending = []
        try:
            pending.append(self._inputs.get(timeout=0.05))
        except queue.Empty:
            return pending
        # Give the producer a short opportunity to fill the initial batch.
        deadline = time.perf_counter() + 0.01
        while len(pending) < self.max_requests:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                pending.append(self._inputs.get(timeout=remaining))
            except queue.Empty:
                break
        return pending

    def _run_generation_loop(self) -> None:
        device_context = (
            torch.cuda.device(self.runner.device)
            if torch.device(self.runner.device).type == "cuda"
            else nullcontext()
        )
        with device_context:
            while not self._stop.is_set() or not self._inputs.empty():
                pending = self._drain_ready()
                if not pending:
                    continue
                groups = defaultdict(list)
                for request in pending:
                    groups[(len(request[1]), request[2])].append(request)
                for (_prompt_length, max_new_tokens), requests in groups.items():
                    for start in range(0, len(requests), self.max_requests):
                        batch = requests[start : start + self.max_requests]
                        tokens = torch.tensor(
                            [request[1] for request in batch],
                            dtype=torch.long,
                            device=self.runner.device,
                        )
                        try:
                            generated = self.runner.generate(
                                tokens,
                                max_new_tokens=max_new_tokens,
                                eos_token_id=getattr(self.generation_config, "eos_token_id", None),
                            )
                            continuations = generated[:, tokens.shape[1] :].detach().cpu().tolist()
                            for (request_id, prompt_ids, _limit), continuation in zip(
                                batch, continuations, strict=True
                            ):
                                self._outputs.put(
                                    GenerationOutput(
                                        request_id=request_id,
                                        prompt_ids=prompt_ids,
                                        generated_tokens=continuation,
                                        status=RequestStatus.FINISHED,
                                    )
                                )
                        except (RuntimeError, ValueError, torch.AcceleratorError) as error:
                            for request_id, prompt_ids, _limit in batch:
                                self._outputs.put(
                                    GenerationOutput(
                                        request_id=request_id,
                                        prompt_ids=prompt_ids,
                                        error=str(error),
                                        status=RequestStatus.FAILED,
                                    )
                                )


@contextmanager
def patch_transformers_continuous_batching(model, runner):
    """Temporarily route Transformers continuous batching through recirculation."""

    import transformers

    original_manager = transformers.ContinuousBatchingManager
    previous_runner = getattr(model, "_recirculation_runner", None)
    model._recirculation_runner = runner
    transformers.ContinuousBatchingManager = RecirculationContinuousBatchingManager
    try:
        yield
    finally:
        transformers.ContinuousBatchingManager = original_manager
        if previous_runner is None:
            delattr(model, "_recirculation_runner")
        else:
            model._recirculation_runner = previous_runner
