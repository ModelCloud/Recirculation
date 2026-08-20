# Result validity

Artifacts dated before the 2026-08-20 same-token replay correction are retained for historical comparison only.
They measured a delayed cross-token activation intervention and must not be cited as recirculation results.

New path tuning and accuracy evaluation are pending for the corrected implementation. The CUDA prefill artifact was
rerun after the correction and identifies the same-token replay scheduler explicitly. The CUDA concurrency artifact
compares mathematically equivalent sequential and two-stream schedules on the corrected implementation.
It was collected on a GIL-enabled runtime, which is now supported by `CUDAConcurrentRunner`.

The three-mode latency comparison in
[`cuda_dense_vs_recirculation_128_tokens.md`](cuda_dense_vs_recirculation_128_tokens.md) records tested commit IDs,
fresh dense/sequential timings, and the provenance and limitations of the historical parallel measurement.
