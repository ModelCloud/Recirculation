# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from recirculation.inference_defaults import (
    resolve_dense_cuda_defaults,
    resolve_recirculation_cuda_defaults,
)


def _recirculation_args(**overrides):
    values = {
        "device": "cuda",
        "cuda_paged_continuous": None,
        "cuda_batch_size": None,
        "attention_backend": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _dense_args(**overrides):
    values = {
        "batch_size": None,
        "attention_backend": None,
        "continuous_batching": None,
        "paged_attention": None,
        "cuda_graph": None,
        "max_batch_tokens": None,
        "max_blocks_per_request": 8,
        "paged_num_blocks": 256,
        "paged_block_size": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
def test_recirculation_cuda_defaults_enable_fast_path(device):
    args = _recirculation_args(device=device)

    cuda_device, automatic = resolve_recirculation_cuda_defaults(args, flash_available=True)

    assert cuda_device is True
    assert automatic is True
    assert args.cuda_paged_continuous is True
    assert args.cuda_batch_size == 32
    assert args.attention_backend == "flash_attention_2"


def test_recirculation_cuda_defaults_fall_back_without_flash_attention():
    args = _recirculation_args()

    cuda_device, automatic = resolve_recirculation_cuda_defaults(args, flash_available=False)

    assert cuda_device is True
    assert automatic is False
    assert args.cuda_paged_continuous is False
    assert args.cuda_batch_size == 32
    assert args.attention_backend == "sdpa"


def test_recirculation_non_cuda_defaults_remain_portable():
    args = _recirculation_args(device="mps")

    cuda_device, automatic = resolve_recirculation_cuda_defaults(args, flash_available=True)

    assert cuda_device is False
    assert automatic is False
    assert args.cuda_paged_continuous is False
    assert args.cuda_batch_size == 4
    assert args.attention_backend == "eager"


def test_recirculation_explicit_settings_are_preserved():
    args = _recirculation_args(
        cuda_paged_continuous=False,
        cuda_batch_size=7,
        attention_backend="eager",
    )

    cuda_device, automatic = resolve_recirculation_cuda_defaults(args, flash_available=True)

    assert cuda_device is True
    assert automatic is False
    assert args.cuda_paged_continuous is False
    assert args.cuda_batch_size == 7
    assert args.attention_backend == "eager"


def test_recirculation_paged_mode_requires_cuda_and_flash_attention():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        resolve_recirculation_cuda_defaults(
            _recirculation_args(device="cpu", cuda_paged_continuous=True),
            flash_available=True,
        )
    with pytest.raises(ValueError, match="requires FlashAttention 2"):
        resolve_recirculation_cuda_defaults(
            _recirculation_args(cuda_paged_continuous=True),
            flash_available=False,
        )


def test_dense_cuda_defaults_enable_full_fast_path():
    args = _dense_args()

    automatic = resolve_dense_cuda_defaults(args, flash_available=True)

    assert automatic is True
    assert args.batch_size == 32
    assert args.attention_backend == "flash_attention_2"
    assert args.continuous_batching is True
    assert args.paged_attention is True
    assert args.cuda_graph is False
    assert args.max_batch_tokens == 65_536


def test_dense_cuda_defaults_fall_back_without_flash_attention():
    args = _dense_args()

    automatic = resolve_dense_cuda_defaults(args, flash_available=False)

    assert automatic is False
    assert args.batch_size == 32
    assert args.attention_backend == "sdpa"
    assert args.continuous_batching is False
    assert args.paged_attention is False
    assert args.cuda_graph is False
    assert args.max_batch_tokens == 65_536


def test_dense_explicit_settings_are_preserved():
    args = _dense_args(
        batch_size=9,
        attention_backend="eager",
        continuous_batching=True,
        paged_attention=False,
        cuda_graph=False,
        max_batch_tokens=12_345,
    )

    automatic = resolve_dense_cuda_defaults(args, flash_available=True)

    assert automatic is False
    assert args.batch_size == 9
    assert args.attention_backend == "eager"
    assert args.continuous_batching is True
    assert args.paged_attention is False
    assert args.cuda_graph is False
    assert args.max_batch_tokens == 12_345


def test_dense_cuda_token_budget_is_bounded_by_request_capacity():
    args = _dense_args(
        batch_size=4,
        paged_num_blocks=256,
        paged_block_size=256,
        max_blocks_per_request=8,
    )

    resolve_dense_cuda_defaults(args, flash_available=True)

    assert args.max_batch_tokens == 8_192


def test_dense_paged_attention_requires_flash_and_continuous_batching():
    with pytest.raises(ValueError, match="requires FlashAttention 2"):
        resolve_dense_cuda_defaults(
            _dense_args(paged_attention=True),
            flash_available=False,
        )
    with pytest.raises(ValueError, match="requires --continuous-batching"):
        resolve_dense_cuda_defaults(
            _dense_args(paged_attention=True, continuous_batching=False),
            flash_available=True,
        )
    with pytest.raises(ValueError, match="requires --paged-attention"):
        resolve_dense_cuda_defaults(
            _dense_args(paged_attention=False, cuda_graph=True),
            flash_available=True,
        )
