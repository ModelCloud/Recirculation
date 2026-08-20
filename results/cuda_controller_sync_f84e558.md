# CUDA controller synchronization validation

- Implementation commit: `f84e5588675154275e2deac18c9b40d47bc57821`
- Model: `/local-models/Llama-3.2-1B-Instruct`
- Configuration: source layer 12, destination layer 5, alpha 0.1, 32 tokens, 2 repetitions
- Accuracy gate: error rate <= `2e-3`
- Result: pass; every measured eager and graphed logits/state error rate was exactly `0.0`
- Torch controller versus CUDA generation regression test: exact token equality

| Execution path | Median latency | Speedup vs sequential |
|---|---:|---:|
| Sequential | 696.263 ms | 1.00x |
| Eager dual-stream | 635.109 ms | 1.10x |
| CUDA graph dual-stream | 127.387 ms | 5.47x |

The benchmark used CPython with the GIL enabled, two Python worker threads, and two CUDA streams. CUDA graph warmup and capture were protected by the process-wide capture lock.

Machine-readable measurements are in `cuda_controller_sync_f84e558.json`.
