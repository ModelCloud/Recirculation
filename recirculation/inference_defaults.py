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

    max_batch_tokens = getattr(args, "max_batch_tokens", None)
    cuda_graph = getattr(args, "cuda_graph", None)
    auto_requested = all(
        value is None
        for value in (
            args.batch_size,
            args.attention_backend,
            args.continuous_batching,
            args.paged_attention,
            max_batch_tokens,
            cuda_graph,
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
    if cuda_graph is None:
        # The repository-owned packed recirculation forward still consumes
        # dynamic CUDA metadata on the host and is not graph-capture safe.
        args.cuda_graph = False
    if max_batch_tokens is None:
        cache_capacity = int(getattr(args, "paged_num_blocks", 256)) * int(
            getattr(args, "paged_block_size", 256)
        )
        request_capacity = (
            int(args.batch_size)
            * int(getattr(args, "max_blocks_per_request", 8))
            * int(getattr(args, "paged_block_size", 256))
        )
        args.max_batch_tokens = min(cache_capacity, request_capacity)
    if args.paged_attention and not flash_available:
        raise ValueError(
            "paged attention requires FlashAttention 2; install flash-attn or pass --no-paged-attention"
        )
    if args.paged_attention and not args.continuous_batching:
        raise ValueError("--paged-attention requires --continuous-batching")
    if args.cuda_graph and not args.paged_attention:
        raise ValueError("--cuda-graph requires --paged-attention")
    return auto_requested and flash_available
