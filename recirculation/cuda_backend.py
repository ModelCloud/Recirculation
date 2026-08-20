# SPDX-License-Identifier: Apache-2.0

"""CUDA serial prefill with a fused recirculation norm-and-mix kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:  # pragma: no cover - depends on the CUDA installation
    raise ImportError("The CUDA backend requires Triton. Install the 'cuda' extra.") from error

from .controller import RecirculationConfig, RecirculationController

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


def measure_forward_error(reference: torch.Tensor, candidate: torch.Tensor) -> ForwardError:
    """Measure an optimized accumulated forward against the PyTorch oracle."""

    reference = reference.float()
    candidate = candidate.float()
    difference = (reference - candidate).abs()
    max_absolute = difference.max().item()
    reference_l2 = torch.linalg.vector_norm(reference).item()
    relative_l2 = torch.linalg.vector_norm(difference).item() / max(reference_l2, 1e-12)
    reference_max = reference.abs().max().item()
    normalized_max = max_absolute / max(reference_max, 1e-12)
    return ForwardError(max_absolute, relative_l2, normalized_max)


def mix_reference(
    destination: torch.Tensor,
    source: torch.Tensor,
    alpha: float,
    beta: float,
    normalize_source: bool,
) -> torch.Tensor:
    """Published norm-ratio mixture expressed with ordinary PyTorch ops."""

    source = source.to(device=destination.device, dtype=destination.dtype)
    if normalize_source:
        destination_norm = destination.float().norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        source_norm = source.float().norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).eps)
        source = source * (destination_norm.to(source.dtype) / source_norm.to(source.dtype))
    return beta * destination + alpha * source


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
        # Match the eager oracle's dtype boundaries. PyTorch casts each norm
        # back to the residual dtype before division and rounds the scaled
        # source before applying alpha.
        destination_norm = destination_norm.to(element_type)
        source_norm = source_norm.to(element_type)
        scale = (destination_norm / tl.maximum(source_norm, 1.0e-12)).to(element_type)
        source = (source * scale).to(element_type)
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
    """Paper-faithful cached prefill that skips intermediate vocabulary projections."""

    def __init__(
        self,
        model,
        config: RecirculationConfig,
        *,
        fused: bool = True,
        skip_intermediate_logits: bool = True,
    ):
        if not hasattr(model, "get_decoder") or model.get_output_embeddings() is None:
            raise TypeError("CUDA prefill requires a Hugging Face causal language model")
        mixer = FusedNormMix() if fused else mix_reference
        self.model = model
        self.decoder = model.get_decoder()
        self.output_embeddings = model.get_output_embeddings()
        self.controller = RecirculationController(model, config, mixer=mixer)
        self.skip_intermediate_logits = skip_intermediate_logits

    @torch.inference_mode()
    def prefill(
        self,
        tokens: Sequence[int] | torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        collect_logits: bool = False,
    ):
        """Return final logits, KV cache, delayed source, and optionally every token's logits."""

        device = next(self.model.parameters()).device
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(list(tokens), dtype=torch.long, device=device)
        tokens = tokens.to(device=device, dtype=torch.long)
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
            raise ValueError("prefill requires tokens with shape [sequence] or [1, sequence]")
        if attention_mask is None:
            attention_mask = torch.ones_like(tokens)
        else:
            attention_mask = attention_mask.to(device=device)
        if attention_mask.shape != tokens.shape:
            raise ValueError("attention_mask must have the same shape as tokens")

        cache = None
        logits = None
        collected = []
        self.controller.attach()
        try:
            for position in range(tokens.shape[1]):
                output = self.decoder(
                    input_ids=tokens[:, position : position + 1],
                    attention_mask=attention_mask[:, : position + 1],
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
            pending_source = self.controller._pending_source
            if logits is None or pending_source is None:  # pragma: no cover - guarded by model/config validation
                raise RuntimeError("prefill did not produce logits and delayed source state")
            return logits, cache, pending_source.detach(), torch.cat(collected, dim=1)
        finally:
            self.controller.detach()


__all__ = [
    "MAX_FORWARD_ERROR",
    "CUDAPrefillRunner",
    "ForwardError",
    "FusedNormMix",
    "measure_forward_error",
    "mix_reference",
]
