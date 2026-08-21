# SPDX-License-Identifier: Apache-2.0

"""Recirculation-specific MLX kernels."""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention

_DUAL_GEMV_SOURCE = r"""
    constexpr int TM = 4;
    constexpr int TN = 4;
    constexpr int SN = 32;
    constexpr int block_m = BM * TM;
    constexpr int block_n = SN * TN;

    const int in_size = weight_shape[1];
    const int out_size = weight_shape[0];
    const int simd_lane = int(thread_index_in_simdgroup);
    int out_row = int(threadgroup_position_in_grid.x) * block_m
        + int(simdgroup_index_in_threadgroup) * TM;
    if (out_row >= out_size) {
        return;
    }
    out_row = out_row + TM <= out_size ? out_row : out_size - TM;

    float result0[TM] = {0.0f};
    float result1[TM] = {0.0f};
    T matrix_values[TN];
    float vector0[TN];
    float vector1[TN];
    int input_col = simd_lane * TN;
    const int full_blocks = in_size / block_n;

    for (int block = 0; block < full_blocks; ++block) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) {
            vector0[tn] = static_cast<float>(input0[input_col + tn]);
            vector1[tn] = static_cast<float>(input1[input_col + tn]);
        }
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) {
            const int matrix_offset = (out_row + tm) * in_size + input_col;
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                matrix_values[tn] = weight[matrix_offset + tn];
            }
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                result0[tm] += matrix_values[tn] * vector0[tn];
                result1[tm] += matrix_values[tn] * vector1[tn];
            }
        }
        input_col += block_n;
    }

    if (input_col < in_size) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) {
            const int column = input_col + tn;
            vector0[tn] = column < in_size ? static_cast<float>(input0[column]) : 0.0f;
            vector1[tn] = column < in_size ? static_cast<float>(input1[column]) : 0.0f;
        }
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) {
            const int matrix_offset = (out_row + tm) * in_size + input_col;
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                const int column = input_col + tn;
                matrix_values[tn] = column < in_size ? weight[matrix_offset + tn] : T(0);
            }
            #pragma clang loop unroll(full)
            for (int tn = 0; tn < TN; ++tn) {
                result0[tm] += matrix_values[tn] * vector0[tn];
                result1[tm] += matrix_values[tn] * vector1[tn];
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {
        #pragma clang loop unroll(full)
        for (ushort offset = SN / 2; offset >= 1; offset >>= 1) {
            result0[tm] += simd_shuffle_down(result0[tm], offset);
            result1[tm] += simd_shuffle_down(result1[tm], offset);
        }
    }

    if (simd_lane == 0) {
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) {
            output0[out_row + tm] = static_cast<T>(result0[tm]);
            output1[out_row + tm] = static_cast<T>(result1[tm]);
        }
    }
"""


class DualGemvMetal:
    """Evaluate two batch-one dense projections while loading each weight once.

    The launch geometry and FP32 reduction order mirror MLX's non-transposed
    GEMV dispatch for the dense Qwen3 projection shapes used by recirculation.
    """

    def __init__(self):
        self._kernel = mx.fast.metal_kernel(
            name="recirculation_dual_gemv",
            input_names=["weight", "input0", "input1"],
            output_names=["output0", "output1"],
            source=_DUAL_GEMV_SOURCE,
            compile_options={"math_mode": "safe"},
        )

    @staticmethod
    def supports(weight: mx.array) -> bool:
        if weight.ndim != 2:
            return False
        out_features, in_features = weight.shape
        return (
            weight.dtype in (mx.bfloat16, mx.float16, mx.float32)
            and in_features > 64
            and in_features < 16 * out_features
            and out_features >= 4
        )

    def __call__(self, weight: mx.array, input0: mx.array, input1: mx.array):
        if weight.ndim != 2:
            raise ValueError("dual GEMV weight must have shape [out_features, in_features]")
        if input0.shape != input1.shape or input0.ndim < 1:
            raise ValueError("dual GEMV inputs must have the same non-scalar shape")
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        if input0.size != in_features:
            raise ValueError("dual GEMV inputs must each contain exactly in_features elements")
        if weight.dtype != input0.dtype or weight.dtype != input1.dtype:
            raise TypeError("dual GEMV weight and inputs must share one dtype")
        if weight.dtype not in (mx.bfloat16, mx.float16, mx.float32):
            raise TypeError("dual GEMV supports bfloat16, float16, and float32")
        if not self.supports(weight):
            raise ValueError("shape does not use MLX's supported BN=1 GEMV dispatch")

        block_rows = 8 if out_features >= 4096 else 4
        rows_per_threadgroup = block_rows * 4
        threadgroup_count = (out_features + rows_per_threadgroup - 1) // rows_per_threadgroup
        output_shape = (*input0.shape[:-1], out_features)
        return tuple(
            self._kernel(
                inputs=[weight, input0, input1],
                template=[("T", weight.dtype), ("BM", block_rows)],
                grid=(threadgroup_count * 32, 1, block_rows),
                threadgroup=(32, 1, block_rows),
                output_shapes=[output_shape, output_shape],
                output_dtypes=[weight.dtype, weight.dtype],
                stream=mx.gpu,
            )
        )


class Qwen3DualTokenLayer:
    """Run adjacent Qwen3 token stacks with exact dual-GEMV projections."""

    def __init__(self, projection: DualGemvMetal | None = None):
        self.projection = DualGemvMetal() if projection is None else projection

    def __call__(self, layer, replay: mx.array, current: mx.array, mask, cache):
        if replay.shape != current.shape or replay.shape[0:2] != (1, 1):
            raise ValueError("paired Qwen3 layer inputs must both have shape [1, 1, hidden_size]")
        attention = layer.self_attn
        replay_norm = layer.input_layernorm(replay)
        current_norm = layer.input_layernorm(current)
        replay_q, current_q = self.projection(attention.q_proj.weight, replay_norm, current_norm)
        replay_k, current_k = self.projection(attention.k_proj.weight, replay_norm, current_norm)
        replay_v, current_v = self.projection(attention.v_proj.weight, replay_norm, current_norm)

        batch, length, _ = replay.shape
        replay_q = attention.q_norm(replay_q.reshape(batch, length, attention.n_heads, -1)).transpose(0, 2, 1, 3)
        replay_k = attention.k_norm(replay_k.reshape(batch, length, attention.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        replay_v = replay_v.reshape(batch, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)
        current_q = attention.q_norm(current_q.reshape(batch, length, attention.n_heads, -1)).transpose(0, 2, 1, 3)
        current_k = attention.k_norm(current_k.reshape(batch, length, attention.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        current_v = current_v.reshape(batch, length, attention.n_kv_heads, -1).transpose(0, 2, 1, 3)

        replay_q = attention.rope(replay_q, offset=cache.offset)
        replay_k = attention.rope(replay_k, offset=cache.offset)
        replay_keys, replay_values = cache.update_and_fetch(replay_k, replay_v)
        replay_attention = scaled_dot_product_attention(
            replay_q,
            replay_keys,
            replay_values,
            cache=cache,
            scale=attention.scale,
            mask=mask,
        )

        current_q = attention.rope(current_q, offset=cache.offset)
        current_k = attention.rope(current_k, offset=cache.offset)
        current_keys, current_values = cache.update_and_fetch(current_k, current_v)
        current_attention = scaled_dot_product_attention(
            current_q,
            current_keys,
            current_values,
            cache=cache,
            scale=attention.scale,
            mask=mask,
        )
        replay_attention = replay_attention.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        current_attention = current_attention.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        replay_attention, current_attention = self.projection(
            attention.o_proj.weight,
            replay_attention,
            current_attention,
        )
        replay_residual = replay + replay_attention
        current_residual = current + current_attention

        replay_norm = layer.post_attention_layernorm(replay_residual)
        current_norm = layer.post_attention_layernorm(current_residual)
        replay_gate, current_gate = self.projection(layer.mlp.gate_proj.weight, replay_norm, current_norm)
        replay_up, current_up = self.projection(layer.mlp.up_proj.weight, replay_norm, current_norm)
        replay_mlp = swiglu(replay_gate, replay_up)
        current_mlp = swiglu(current_gate, current_up)
        replay_mlp, current_mlp = self.projection(layer.mlp.down_proj.weight, replay_mlp, current_mlp)
        return replay_residual + replay_mlp, current_residual + current_mlp
