# Result validity

The completed Qwen3-8B C4+PG19 staged CUDA path search at alpha 0.05 is recorded in
[`qwen3_8b_c4_pg19_path_search_alpha005_b03a904.md`](qwen3_8b_c4_pg19_path_search_alpha005_b03a904.md). It preserves
the implementation commit, run configuration, stage leaders, both final selection objectives, and every Stage 3 score.

Artifacts dated before the 2026-08-20 same-token replay correction are retained for historical comparison only.
They measured a delayed cross-token activation intervention and must not be cited as recirculation results.

The corrected path and alpha sweep, exact dataset partition, configuration, commands, full rankings, and locked
evaluation record are documented in [same_token_gsm8k_platinum.md](same_token_gsm8k_platinum.md). The CUDA prefill
artifact was rerun after the correction and identifies the same-token replay scheduler explicitly. The CUDA
concurrency artifact compares mathematically equivalent sequential and two-stream schedules on the corrected
implementation. It was collected on a GIL-enabled runtime, which is now supported by `CUDAConcurrentRunner`.

The three-mode latency comparison in
[`cuda_dense_vs_recirculation_128_tokens.md`](cuda_dense_vs_recirculation_128_tokens.md) records tested commit IDs,
fresh dense/sequential timings, and the provenance and limitations of the historical parallel measurement.

The GIL-enabled eager and CUDA Graph dual-stream comparison is recorded in
[`cuda_concurrent_graph_gil1_128_tokens.md`](cuda_concurrent_graph_gil1_128_tokens.md), including its implementation
commit, raw JSON artifact, environment, and accuracy standards.

The post-capture mode/priority sweep and its higher-repetition confirmation are summarized in
[`cuda_graph_mode_sweep_gil1_128_tokens.md`](cuda_graph_mode_sweep_gil1_128_tokens.md). The nominal winner preserves
global capture safety and changes only the lower/replay stream priority.
