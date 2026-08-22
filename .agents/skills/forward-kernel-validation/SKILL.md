---
name: forward-kernel-validation
description: Validate optimized Recirculation forward kernels against an unfused, unbatched oracle when changing batching, fusion, attention, CUDA graphs, or accelerator inference code.
---

# Forward kernel validation

Compare optimized forward outputs with the unfused, unbatched implementation using identical weights, dtype, tokens,
positions, masks, cache state, and Recirculation configuration.

Use mean absolute forward error as the release gate:

- Fused or batched kernels: mean absolute error must be at most `4e-3`.
- Kernels that are both non-fused and non-batched: mean absolute error must be at most `2e-3`.

A path qualifies for the `4e-3` limit if it uses fusion, batching, or both. Record max absolute error, relative L2, and
normalized max for diagnostics, but do not use those metrics to accept or reject the kernel. A stricter explicitly
requested gate may still be used.

Benchmark accuracy, task scores, and greedy-token agreement are outcome diagnostics, not substitutes for the forward
gate. Do not reject an otherwise passing optimization solely because a benchmark score or greedy output changes;
report those changes separately. Save the selected gate, kernel classification, oracle configuration, and all error
metrics with benchmark artifacts.
