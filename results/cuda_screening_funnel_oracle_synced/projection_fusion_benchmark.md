# Projection fusion experiment

Workload: Llama 3.2 1B Instruct, CUDA SDPA, 128-token prompt, 16-token
warm decode, `8→2, α=0.20`.

| Variant | Accuracy gate | Throughput | Decision |
|---|---:|---:|---|
| Current eager runner | pass | ~5.0 tok/s | reference |
| Packed gate/up GEMM | exact (zero error) | 4.63 tok/s | reject: slower |
| Packed QKV GEMM | fail: max abs 3.906e-2 | not promoted | reject: accuracy |

Packed QKV had normalized maximum error 1.484e-3 but exceeded the repository's
strict absolute forward-error gate of 2e-3. Packing gate/up was bit-exact but
slower for batch-one decode. `down_proj` cannot be folded into the input
projections algebraically because it consumes the nonlinear SwiGLU product; it
requires a purpose-built fused SwiGLU/down kernel.
