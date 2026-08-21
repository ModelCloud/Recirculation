# SPDX-License-Identifier: Apache-2.0

"""Resolve reproducible inference defaults shared by CUDA evaluation entrypoints."""

from __future__ import annotations


def resolve_recirculation_cuda_defaults(args, *, flash_available: bool) -> tuple[bool, bool]:
    """Mutate a recirculation-eval namespace and return CUDA/automatic flags."""

    cuda_device = str(args.device).split(":", 1)[0] == "cuda"
    paged_auto = args.cuda_paged_continuous is None
    if args.cuda_paged_continuous is None:
        args.cuda_paged_continuous = cuda_device and flash_available
    if args.cuda_paged_continuous and not cuda_device:
        raise ValueError("--cuda-paged-continuous requires a CUDA device")
    if args.cuda_paged_continuous and not flash_available:
        raise ValueError(
            "CUDA paged continuous batching requires FlashAttention 2; install flash-attn "
            "or pass --no-cuda-paged-continuous"
        )
    if args.cuda_batch_size is None:
        args.cuda_batch_size = 32 if cuda_device else 4
    if args.attention_backend is None:
        args.attention_backend = (
            "flash_attention_2"
            if args.cuda_paged_continuous
            else "sdpa"
            if cuda_device
            else "eager"
        )
    return cuda_device, cuda_device and paged_auto and args.cuda_paged_continuous


def resolve_dense_cuda_defaults(args, *, flash_available: bool) -> bool:
    """Mutate a dense-Evalution namespace and return whether all CUDA defaults were automatic."""

    auto_requested = all(
        value is None
        for value in (
            args.batch_size,
            args.attention_backend,
            args.continuous_batching,
            args.paged_attention,
        )
    )
    if args.batch_size is None:
        args.batch_size = 32
    if args.attention_backend is None:
        args.attention_backend = "flash_attention_2" if flash_available else "sdpa"
    if args.paged_attention is None:
        args.paged_attention = flash_available
    if args.continuous_batching is None:
        args.continuous_batching = args.paged_attention
    if args.paged_attention and not flash_available:
        raise ValueError(
            "paged attention requires FlashAttention 2; install flash-attn or pass --no-paged-attention"
        )
    if args.paged_attention and not args.continuous_batching:
        raise ValueError("--paged-attention requires --continuous-batching")
    return auto_requested and flash_available
