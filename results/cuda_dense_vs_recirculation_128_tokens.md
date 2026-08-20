# CUDA dense vs. recirculation latency

Benchmark date: 2026-08-20 UTC

This record compares token-wise dense inference with sequential recirculation and preserves the existing paired
sequential/concurrent recirculation result. Commit IDs are recorded so later kernel changes can be compared with the
exact implementations measured here.

## Result

| Mode | Median latency for 128 tokens | Relative to dense | Measurement commit |
| --- | ---: | ---: | --- |
| Dense, no recirculation | 1,750.060 ms | 1.000x | `42b08dc981d97a9775558c9889d92722d530f5ed` |
| Sequential recirculation | 2,711.806 ms | 1.550x latency | `42b08dc981d97a9775558c9889d92722d530f5ed` |
| Parallel recirculation | 2,481.095 ms | 1.418x latency[^cross-run] | `a6a040c2e58f5e1206d2979d262c74647b43436e` |

Dense inference was **1.550x faster** than sequential recirculation in the fresh paired run. Sequential
recirculation added **55.0% latency**.

The historical paired recirculation run measured 2,643.354 ms sequential and 2,481.095 ms parallel. Parallel
recirculation was therefore **1.065x faster** than its paired sequential run, reducing latency by **6.14%**. Against
the fresh dense median, its recorded latency is 41.8% higher, but that cross-run comparison is indicative only.[^cross-run]

## Fresh dense/sequential run

- Model: `/local-models/Llama-3.2-1B-Instruct`, FP16
- Input: 128 tokens, batch size 1
- Schedule: token-wise inference with a growing KV cache; only the final vocabulary projection is computed
- Recirculation: source layer 12, destination layer 5, alpha 0.1, no ramp
- Warmups: 1 per mode
- Timed repetitions: 3 per mode; wall-clock time around each complete prefill with CUDA synchronization
- Dense samples: 1,589.508 ms, 1,750.060 ms, 1,841.809 ms
- Sequential recirculation samples: 2,804.630 ms, 2,711.806 ms, 2,665.545 ms
- Tested commit: `42b08dc981d97a9775558c9889d92722d530f5ed`

## Environment

- GPU: NVIDIA PG506-230, 98,304 MiB
- NVIDIA driver: 610.43.02
- CUDA reported by PyTorch: 13.0
- PyTorch: 2.15.0.dev20260817+cu130
- Python: 3.14.6, Clang 22.1.3
- GIL state: enabled
- OS kernel: Linux 6.18.28 x86_64

## Parallel-result provenance and limitation

The parallel result comes from [`cuda_concurrent_128_tokens.json`](cuda_concurrent_128_tokens.json), measured on the
same GPU/runtime family at commit `a6a040c2e58f5e1206d2979d262c74647b43436e`. That paired run used five repetitions
and reported zero logits and pending-state error between sequential and parallel recirculation.

It predates the `GIL=0` enforcement added by commit `42b08dc981d97a9775558c9889d92722d530f5ed`. The current Python
binary reports `GIL=1` and rejects `-X gil=0`, so a fresh `CUDAConcurrentRunner` measurement cannot be made on this
interpreter. This historical number must not be represented as a free-threaded Python result. Rerun all three modes
under a supported GIL-free build before drawing a final conclusion about parallel speedup.

[^cross-run]: The dense and parallel values were collected in separate benchmark runs and at different commits.
    Only the fresh dense/sequential ratio and the historical sequential/parallel ratio are paired comparisons.
