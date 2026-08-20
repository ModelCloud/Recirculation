# Result validity

Artifacts dated before the 2026-08-20 same-token replay correction are retained for historical comparison only.
They measured a delayed cross-token activation intervention and must not be cited as recirculation results.

New path tuning and accuracy evaluation are pending for the corrected implementation. The CUDA prefill artifact was
rerun after the correction and identifies the same-token replay scheduler explicitly. The CUDA concurrency artifact
compares mathematically equivalent sequential and two-stream schedules on the corrected implementation.
It was collected on a GIL-enabled development runtime before `CUDAConcurrentRunner` was restricted to `GIL=0`.
