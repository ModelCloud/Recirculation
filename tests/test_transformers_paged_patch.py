# SPDX-License-Identifier: Apache-2.0

import queue
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from transformers.generation.continuous_batching.requests import RequestStatus

import recirculation.transformers_paged_patch as paged_patch
from recirculation import RecirculationConfig
from recirculation.cuda_backend import CUDAPrefillSnapshot, CUDARecirculationState
from recirculation.transformers_paged_patch import (
    PagedRecirculationForward,
    RecirculationContinuousBatchingManager,
    RecirculationPagedState,
    make_paged_cache_recirculation_aware,
    patch_model_paged_recirculation,
    patch_transformers_continuous_batching,
    patch_transformers_paged_cache_defaults,
    seed_paged_cache_from_snapshot,
)


class _FakeRunner:
    device = torch.device("cpu")

    def generate(self, tokens, *, max_new_tokens, eos_token_id):
        del eos_token_id
        continuation = torch.full(
            (tokens.shape[0], max_new_tokens),
            42,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        return torch.cat((tokens, continuation), dim=1)


class _RecordingRunner(_FakeRunner):
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def generate(self, tokens, *, max_new_tokens, eos_token_id):
        self.calls.append((tokens.clone(), max_new_tokens, eos_token_id))
        if self.error is not None:
            raise self.error
        return super().generate(
            tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )


class _FakeLayer:
    def __init__(self, index):
        self.index = index
        self.calls = []

    def __call__(self, hidden, **kwargs):
        self.calls.append(kwargs)
        return hidden + self.index + 1


class _FakeDecoder:
    def __init__(self, layer_count=3, hidden_size=2):
        self.layers = [_FakeLayer(index) for index in range(layer_count)]
        self.hidden_size = hidden_size

    def embed_tokens(self, input_ids):
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, self.hidden_size)

    @staticmethod
    def norm(hidden):
        return hidden


class _FakeModel(torch.nn.Module):
    def __init__(self, layer_count=3, hidden_size=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.decoder = _FakeDecoder(layer_count, hidden_size)
        self.forward = self._original_forward

    def _original_forward(self, *_args, **_kwargs):
        return "original"

    def get_decoder(self):
        return self.decoder


class _BlockManager:
    def __init__(self):
        self.completed = []

    def mark_shareable_blocks_as_complete(self, **kwargs):
        self.completed.append(kwargs)


def _fake_cache(*, pages=32, block_size=4, block_table=None, layer_count=1):
    allocator = SimpleNamespace(block_table=block_table or {})
    copy_calls = []
    freed = []
    cache = SimpleNamespace(
        key_cache=[torch.zeros(pages, 1, 2) for _ in range(layer_count)],
        value_cache=[torch.zeros(pages, 1, 2) for _ in range(layer_count)],
        device=torch.device("cpu"),
        block_size=block_size,
        group_cache_managers=[allocator],
        layer_index_to_group_indices=[(0, index) for index in range(layer_count)],
        _block_manager=_BlockManager(),
        copy_cache=lambda source, destination: copy_calls.append((source, destination)),
        free_blocks=lambda request_id: freed.append(request_id),
    )
    cache.copy_calls = copy_calls
    cache.freed = freed

    def allocate_blocks(count, request_id, allocated_blocks=0):
        del allocated_blocks
        allocator.block_table[request_id] = list(range(1, count + 1))
        return count

    cache.allocate_blocks = allocate_blocks
    return cache


@pytest.fixture
def simple_forward(monkeypatch):
    monkeypatch.setattr(
        paged_patch,
        "_decoder_position_embeddings",
        lambda _decoder, _hidden, _positions: None,
    )
    monkeypatch.setattr(
        paged_patch,
        "_layer_position_embeddings",
        lambda _decoder, embeddings, _index: embeddings,
    )
    monkeypatch.setattr(
        paged_patch,
        "project_causal_lm_logits",
        lambda _model, hidden: hidden,
    )


def test_fixed_slot_manager_batches_shape_compatible_requests():
    model = SimpleNamespace(_recirculation_runner=_FakeRunner())
    generation = SimpleNamespace(max_new_tokens=3, eos_token_id=2)
    batching = SimpleNamespace(max_requests_per_batch=4)
    manager = RecirculationContinuousBatchingManager(model, generation, batching)
    manager.start()
    try:
        manager.add_request([1, 2, 3], request_id="a", max_new_tokens=2)
        manager.add_request([4, 5, 6], request_id="b", max_new_tokens=2)
        outputs = {output.request_id: output for output in (manager.get_result(1), manager.get_result(1))}
        assert outputs["a"].generated_tokens == [42, 42]
        assert outputs["b"].generated_tokens == [42, 42]
        assert all(output.is_finished() for output in outputs.values())
    finally:
        manager.stop()


def test_paged_recurrent_state_tracks_physical_pages_and_copies_blocks():
    allocator = SimpleNamespace(block_table={"a": [1, 3], "b": [4]})
    cache = SimpleNamespace(
        key_cache=[torch.empty(24, 2, 2)],
        device=torch.device("cpu"),
        block_size=4,
        group_cache_managers=[allocator],
    )
    state = RecirculationPagedState(cache, hidden_size=3, dtype=torch.float32)
    indices = state.physical_token_indices(["a", "a", "b"], [0, 5, 2])
    assert indices.tolist() == [4, 13, 18]
    destination = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    source = destination + 20
    state.store(indices, destination, source, torch.tensor([0, 5, 2]))
    loaded_destination, loaded_source, loaded_steps = state.load(indices)
    assert torch.equal(loaded_destination, destination)
    assert torch.equal(loaded_source, source)
    assert loaded_steps.tolist() == [0, 5, 2]

    block_pages = torch.arange(4, 8)
    state.store(
        block_pages,
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100,
        torch.arange(4),
    )
    state.copy_blocks([1], [2])
    copied = torch.arange(8, 12)
    assert torch.equal(state.destination[copied], state.destination[block_pages])
    assert torch.equal(state.source[copied], state.source[block_pages])
    assert torch.equal(state.input_steps[copied], state.input_steps[block_pages])


def test_paged_cache_copy_on_write_copies_recurrent_pages():
    copied_kv = []
    cache = SimpleNamespace(
        key_cache=[torch.empty(16, 2, 2)],
        device=torch.device("cpu"),
        block_size=4,
        group_cache_managers=[SimpleNamespace()],
        copy_cache=lambda source, destination: copied_kv.append((source, destination)),
    )
    state = make_paged_cache_recirculation_aware(
        cache,
        hidden_size=3,
        dtype=torch.float32,
    )
    source_pages = torch.arange(4, 8)
    state.store(
        source_pages,
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100,
        torch.arange(4),
    )

    cache.copy_cache([1], [2])

    assert copied_kv == [([1], [2])]
    destination_pages = torch.arange(8, 12)
    assert torch.equal(state.destination[destination_pages], state.destination[source_pages])
    assert torch.equal(state.source[destination_pages], state.source[source_pages])
    assert torch.equal(state.input_steps[destination_pages], state.input_steps[source_pages])


def test_paged_cache_default_patch_is_scoped_and_preserves_explicit_values():
    import inspect

    from transformers import ContinuousBatchingConfig

    original_init = ContinuousBatchingConfig.__init__
    original_signature = inspect.signature(original_init)
    with patch_transformers_paged_cache_defaults(num_blocks=17, block_size=8):
        assert inspect.signature(ContinuousBatchingConfig.__init__) == original_signature
        automatic = ContinuousBatchingConfig()
        explicit = ContinuousBatchingConfig(num_blocks=9, block_size=4)
        assert automatic.num_blocks == 17
        assert automatic.block_size == 8
        assert explicit.num_blocks == 9
        assert explicit.block_size == 4
    assert ContinuousBatchingConfig.__init__ is original_init


@pytest.mark.parametrize("num_blocks,block_size", [(0, 4), (4, 0), (-1, 4)])
def test_paged_cache_default_patch_rejects_invalid_sizes(num_blocks, block_size):
    with (
        pytest.raises(ValueError, match="must be positive"),
        patch_transformers_paged_cache_defaults(
            num_blocks=num_blocks,
            block_size=block_size,
        ),
    ):
        pass


def test_paged_state_validation_integer_steps_empty_copy_and_existing_attachment():
    cache = _fake_cache(block_table={"a": [1]})
    with pytest.raises(ValueError, match="hidden_size"):
        RecirculationPagedState(cache, hidden_size=0, dtype=torch.float32)

    state = RecirculationPagedState(cache, hidden_size=2, dtype=torch.float32)
    state.store(
        torch.tensor([4, 5]),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
        7,
    )
    assert state.input_steps[4:6].tolist() == [7, 7]
    before = state.destination.clone()
    state.copy_blocks([], [])
    assert torch.equal(before, state.destination)
    with pytest.raises(ValueError, match="equal lengths"):
        state.physical_token_indices(["a"], [0, 1])
    with pytest.raises(ValueError, match="non-negative"):
        state.physical_token_indices(["a"], [-1])

    cache.recirculation_state = state
    assert (
        make_paged_cache_recirculation_aware(
            cache,
            hidden_size=2,
            dtype=torch.float32,
        )
        is state
    )

    cache_two_groups = _fake_cache()
    cache_two_groups.group_cache_managers.append(SimpleNamespace())
    with pytest.raises(ValueError, match="one full-attention"):
        make_paged_cache_recirculation_aware(
            cache_two_groups,
            hidden_size=2,
            dtype=torch.float32,
        )


def _snapshot(prompt_tokens=4, layer_count=2):
    layers = []
    for index in range(layer_count):
        key = torch.arange(prompt_tokens * 2, dtype=torch.float32).reshape(1, 1, prompt_tokens, 2) + index * 100
        value = key + 50
        layers.append((key, value))
    pending = CUDARecirculationState(
        destination_residual=torch.tensor([[[11.0, 12.0]]]),
        source_residual=torch.tensor([[[21.0, 22.0]]]),
        input_step=prompt_tokens - 1,
    )
    return CUDAPrefillSnapshot(tuple(layers), pending)


def test_seed_paged_cache_imports_kv_state_marks_and_frees_prefix():
    cache = _fake_cache(pages=16, block_size=2, layer_count=2)
    snapshot = _snapshot()
    blocks = seed_paged_cache_from_snapshot(
        cache,
        request_id="seed",
        prompt_ids=[1, 2, 3, 4],
        snapshot=snapshot,
    )

    assert blocks == 2
    physical = torch.tensor([2, 3, 4, 5])
    for index, (key, value) in enumerate(snapshot.cache_data):
        assert torch.equal(cache.key_cache[index][physical], key[0].transpose(0, 1))
        assert torch.equal(cache.value_cache[index][physical], value[0].transpose(0, 1))
    state = cache.recirculation_state
    assert state.input_steps[5].item() == 3
    assert state.destination[5].tolist() == [11.0, 12.0]
    assert state.source[5].tolist() == [21.0, 22.0]
    assert cache._block_manager.completed == [
        {
            "num_complete_blocks": 2,
            "allocated_blocks": [1, 2],
            "prompt_ids": [1, 2, 3, 4],
        }
    ]
    assert cache.freed == ["seed"]


@pytest.mark.parametrize("prompt_ids", [[], [1], [1, 2, 3]])
def test_seed_paged_cache_requires_complete_blocks(prompt_ids):
    with pytest.raises(ValueError, match="complete blocks"):
        seed_paged_cache_from_snapshot(
            _fake_cache(block_size=2),
            request_id="seed",
            prompt_ids=prompt_ids,
            snapshot=_snapshot(),
        )


def test_seed_paged_cache_validates_allocation_layers_and_kv_shape():
    cache = _fake_cache(block_size=2, layer_count=2)
    cache.allocate_blocks = lambda *_args, **_kwargs: 0
    with pytest.raises(RuntimeError, match="could not allocate"):
        seed_paged_cache_from_snapshot(
            cache,
            request_id="seed",
            prompt_ids=[1, 2, 3, 4],
            snapshot=_snapshot(),
        )

    cache = _fake_cache(block_size=2, layer_count=1)
    with pytest.raises(ValueError, match="layer counts"):
        seed_paged_cache_from_snapshot(
            cache,
            request_id="seed",
            prompt_ids=[1, 2, 3, 4],
            snapshot=_snapshot(layer_count=2),
        )

    for invalid_layer in (
        (None, torch.zeros(1, 1, 4, 2)),
        (torch.zeros(1, 1, 4, 2), None),
        (torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),
    ):
        cache = _fake_cache(block_size=2, layer_count=1)
        snapshot = SimpleNamespace(cache_data=(invalid_layer,), pending=_snapshot().pending)
        with pytest.raises(ValueError, match="complete fixed-length"):
            seed_paged_cache_from_snapshot(
                cache,
                request_id="seed",
                prompt_ids=[1, 2, 3, 4],
                snapshot=snapshot,
            )


def _forward_kwargs(cache, *, positions, writes, cu_q, reads=None, cu_k=None, block_table=None):
    reads = torch.tensor([], dtype=torch.long) if reads is None else torch.tensor(reads)
    cu_k = [0] * len(cu_q) if cu_k is None else cu_k
    return {
        "position_ids": torch.tensor([positions]),
        "cache": cache,
        "cu_seq_lens_q": torch.tensor(cu_q, dtype=torch.int32),
        "cu_seq_lens_k": {"full": torch.tensor(cu_k, dtype=torch.int32)},
        "max_seqlen_q": max(right - left for left, right in pairwise(cu_q)),
        "max_seqlen_k": max(positions) + 1,
        "read_index": [reads],
        "write_index": [torch.tensor(writes)],
        "block_table": block_table,
        "logits_indices": torch.tensor([0]),
        "logits_processor_args": {"ignored": True},
        "custom_argument": "preserved",
    }


def test_paged_forward_runs_varlen_current_and_replay_paths(simple_forward):
    model = _FakeModel()
    cache = _fake_cache()
    mixer_calls = []

    def mixer(destination, source, *_args):
        mixer_calls.append((destination.clone(), source.clone()))
        return destination + source

    implementation = PagedRecirculationForward(
        model,
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        mixer,
    )
    output = implementation(
        input_ids=torch.tensor([[1, 2]]),
        logits_to_keep=torch.tensor([1]),
        **_forward_kwargs(cache, positions=[0, 1], writes=[0, 1], cu_q=[0, 2], cu_k=[0, 0]),
    )

    assert output.logits.tolist() == [[[8.0, 8.0]]]
    assert torch.equal(implementation.last_logits, output.logits)
    assert len(mixer_calls) == 1
    state = cache.recirculation_state
    assert state.input_steps[:2].tolist() == [0, 1]
    assert state.destination[:2].tolist() == [[2.0, 2.0], [3.0, 3.0]]
    assert state.source[:2].tolist() == [[4.0, 4.0], [5.0, 5.0]]
    assert model.decoder.layers[0].calls[0]["custom_argument"] == "preserved"


def test_paged_forward_reads_prefix_state_without_block_table(simple_forward):
    model = _FakeModel()
    cache = _fake_cache()
    state = make_paged_cache_recirculation_aware(
        cache,
        hidden_size=2,
        dtype=torch.float32,
    )
    state.store(
        torch.tensor([5]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        0,
    )
    implementation = PagedRecirculationForward(
        model,
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        lambda destination, _source, *_args: destination,
    )
    output = implementation(
        input_ids=torch.tensor([[2]]),
        logits_to_keep=1,
        **_forward_kwargs(
            cache,
            positions=[1],
            writes=[6],
            reads=[5],
            cu_q=[0, 1],
            cu_k=[0, 1],
        ),
    )
    assert output.logits.tolist() == [[[8.0, 8.0]]]
    assert cache.recirculation_state.input_steps[6].item() == 1


def test_paged_forward_uses_block_table_physical_addresses(simple_forward):
    model = _FakeModel()
    cache = _fake_cache(block_size=4)
    state = make_paged_cache_recirculation_aware(
        cache,
        hidden_size=2,
        dtype=torch.float32,
    )
    state.store(
        torch.tensor([8]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        0,
    )
    implementation = PagedRecirculationForward(
        model,
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.05),
        lambda destination, _source, *_args: destination,
    )
    output = implementation(
        input_ids=torch.tensor([[2]]),
        **_forward_kwargs(
            cache,
            positions=[1],
            writes=[31],
            cu_q=[0, 1],
            cu_k=[0, 0],
            block_table=torch.tensor([[[2, 3]]]),
        ),
    )
    assert output.logits.tolist() == [[[8.0, 8.0]]]
    assert cache.recirculation_state.input_steps[9].item() == 1
    assert model.decoder.layers[0].calls[-1]["block_table"].tolist() == [[[2, 3]]]


@pytest.mark.parametrize(
    "call_kwargs,message",
    [
        ({"input_ids": None}, "requires packed input_ids"),
        (
            {"input_ids": torch.tensor([[1]]), "inputs_embeds": torch.zeros(1, 1, 2)},
            "requires packed input_ids",
        ),
        ({"input_ids": torch.tensor([1])}, "packed shape"),
        ({"input_ids": torch.tensor([[1], [2]])}, "packed shape"),
        ({"input_ids": torch.tensor([[1]]), "position_ids": torch.tensor([[0]])}, "requires cache"),
    ],
)
def test_paged_forward_validates_packed_inputs(simple_forward, call_kwargs, message):
    implementation = PagedRecirculationForward(
        _FakeModel(),
        RecirculationConfig(source_layer=1, destination_layer=0),
        lambda destination, *_args: destination,
    )
    with pytest.raises(ValueError, match=message):
        implementation(**call_kwargs)


def test_paged_forward_detects_state_source_and_logit_failures(simple_forward, monkeypatch):
    model = _FakeModel()
    config = RecirculationConfig(source_layer=1, destination_layer=0)
    kwargs = _forward_kwargs(
        _fake_cache(),
        positions=[1],
        writes=[1],
        reads=[0],
        cu_q=[0, 1],
        cu_k=[0, 1],
    )
    state = make_paged_cache_recirculation_aware(kwargs["cache"], hidden_size=2, dtype=torch.float32)
    state.store(torch.tensor([0]), torch.ones(1, 2), torch.ones(1, 2), 9)
    with pytest.raises(RuntimeError, match="does not precede"):
        PagedRecirculationForward(model, config, lambda destination, *_args: destination)(
            input_ids=torch.tensor([[1]]), **kwargs
        )

    model = _FakeModel(layer_count=1)
    kwargs = _forward_kwargs(_fake_cache(), positions=[0], writes=[0], cu_q=[0, 1])
    with pytest.raises(RuntimeError, match="did not capture"):
        PagedRecirculationForward(model, config, lambda destination, *_args: destination)(
            input_ids=torch.tensor([[1]]), **kwargs
        )

    model = _FakeModel()
    kwargs = _forward_kwargs(_fake_cache(), positions=[0], writes=[0], cu_q=[0, 1])
    monkeypatch.setattr(
        paged_patch,
        "project_causal_lm_logits",
        lambda _model, hidden: torch.full_like(hidden, torch.nan),
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        PagedRecirculationForward(model, config, lambda destination, *_args: destination)(
            input_ids=torch.tensor([[1]]), **kwargs
        )


def test_paged_forward_and_model_patch_validate_ramp_and_restore(simple_forward):
    model = _FakeModel()
    ramp = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        ramp_tokens=1,
    )
    with pytest.raises(ValueError, match="ramp_tokens"):
        PagedRecirculationForward(model, ramp, lambda destination, *_args: destination)

    original = model.forward
    with (
        pytest.raises(RuntimeError, match="body failure"),
        patch_model_paged_recirculation(
            model,
            RecirculationConfig(source_layer=1, destination_layer=0),
            lambda destination, *_args: destination,
        ) as implementation,
    ):
        assert model.forward is implementation
        raise RuntimeError("body failure")
    assert model.forward == original


def test_fixed_manager_defaults_add_requests_lifecycle_and_timeout(monkeypatch):
    runner = _RecordingRunner()
    model = SimpleNamespace(_recirculation_runner=runner)
    generation = SimpleNamespace(max_new_tokens=None, eos_token_id=2)
    manager = RecirculationContinuousBatchingManager(model, generation)
    assert manager.max_requests == 32
    assert manager.add_request([1], request_id="default") == "default"
    assert manager._inputs.get_nowait() == ("default", [1], 20)
    assert manager.add_requests([[1, 2], [3, 4]], max_new_tokens=1) == ["req_0", "req_1"]
    manager.start()
    first_thread = manager._thread
    manager.start()
    assert manager._thread is first_thread
    outputs = {manager.get_result(1).request_id, manager.get_result(1).request_id}
    assert outputs == {"req_0", "req_1"}
    manager.stop(block=False, keep_for_next_session=True)
    if manager._thread is not None:
        manager.stop(block=True)
    assert manager.get_result(timeout=0.001) is None
    assert manager.evict_request_from_cache("anything") is None
    manager.destroy()

    with pytest.raises(ValueError, match="no _recirculation_runner"):
        RecirculationContinuousBatchingManager(SimpleNamespace(), generation)
    with pytest.raises(ValueError, match="streaming"):
        RecirculationContinuousBatchingManager(model, generation).add_request([1], request_id="stream", streaming=True)

    idle = RecirculationContinuousBatchingManager(model, generation)
    assert idle._drain_ready() == []
    idle.add_request([1], request_id="queued")
    monkeypatch.setattr(paged_patch.time, "perf_counter", iter([0.0, 1.0]).__next__)
    assert idle._drain_ready() == [("queued", [1], 20)]


def test_fixed_manager_hard_stop_and_failed_generation():
    generation = SimpleNamespace(max_new_tokens=1, eos_token_id=2)
    failing = RecirculationContinuousBatchingManager(
        SimpleNamespace(_recirculation_runner=_RecordingRunner(RuntimeError("boom"))),
        generation,
    )
    failing.add_request([1], request_id="bad")
    failing.start()
    output = failing.get_result(1)
    failing.stop()
    assert output.status == RequestStatus.FAILED
    assert output.error == "boom"

    manager = RecirculationContinuousBatchingManager(
        SimpleNamespace(_recirculation_runner=_RecordingRunner()), generation
    )
    manager.add_request([1], request_id="one")
    manager.add_request([2], request_id="two")
    manager.stop(block=False, hard_stop=True)
    with pytest.raises(queue.Empty):
        manager._inputs.get_nowait()


def test_fixed_manager_enters_cuda_device_context(monkeypatch):
    entered = []

    class FakeCUDAContext:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *_args):
            entered.append(False)

    original_device = torch.device("cpu")
    runner = _RecordingRunner()
    runner.device = original_device
    manager = RecirculationContinuousBatchingManager(
        SimpleNamespace(_recirculation_runner=runner),
        SimpleNamespace(max_new_tokens=1, eos_token_id=2),
    )
    monkeypatch.setattr(
        paged_patch.torch,
        "device",
        lambda _value: SimpleNamespace(type="cuda"),
    )
    monkeypatch.setattr(
        paged_patch.torch.cuda,
        "device",
        lambda _value: FakeCUDAContext(),
    )
    manager.add_request([1], request_id="cuda-context")
    manager.start()
    output = manager.get_result(1)
    manager.stop()
    assert output.status == RequestStatus.FINISHED
    assert entered == [True, False]


def test_transformers_manager_patch_restores_missing_and_previous_runner():
    import transformers

    original_manager = transformers.ContinuousBatchingManager
    model = SimpleNamespace()
    runner = _FakeRunner()
    with patch_transformers_continuous_batching(model, runner):
        assert model._recirculation_runner is runner
        assert transformers.ContinuousBatchingManager is RecirculationContinuousBatchingManager
    assert not hasattr(model, "_recirculation_runner")
    assert transformers.ContinuousBatchingManager is original_manager

    previous = object()
    model._recirculation_runner = previous
    with pytest.raises(RuntimeError, match="body failure"), patch_transformers_continuous_batching(model, runner):
        raise RuntimeError("body failure")
    assert model._recirculation_runner is previous
    assert transformers.ContinuousBatchingManager is original_manager
