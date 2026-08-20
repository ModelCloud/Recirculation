# Compiled recirculation runner benchmark

Workload: Llama 3.2 1B Instruct, SDPA, a prefetched 128-token prompt snapshot,
and 16 incremental decode tokens. Model loading and prompt prefill are excluded.

| Runner | Median throughput |
|---|---:|
| Eager CUDAConcurrentRunner | 42.95 tok/s |
| Compiled lower/upper stacks (`triton.cudagraphs=False`) | 109.65 tok/s |

The compiled runner is 2.55× faster in warm decode. Its first measured decode
call took 12.96 seconds while Inductor compiled the dynamic shapes; subsequent
calls took 0.146 and 0.141 seconds. CUDA sweeps enable it by default because the
one-time cost is amortized over many rows. Use `--no-cuda-compile-runner` for
short smoke/debug runs.
