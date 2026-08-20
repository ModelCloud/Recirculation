# Compiled recirculation runner benchmark

Workload: Llama 3.2 1B Instruct, SDPA, a prefetched 128-token prompt snapshot,
and 16 incremental decode tokens. Model loading and prompt prefill are excluded.

| Runner | Median throughput |
|---|---:|
| Eager CUDAConcurrentRunner | 42.95 tok/s |
| Compiled lower/upper stacks (`triton.cudagraphs=False`) | 109.65 tok/s |

The compiled runner is 2.55× faster in warm decode. Its first measured decode
call took 12.96 seconds while Inductor compiled the dynamic shapes; subsequent
calls took 0.146 and 0.141 seconds.

An end-to-end two-row, 32-token-cap sweep took 48 seconds with shared-prefix
eager execution and 114 seconds with compilation, showing that compiler startup
dominates small jobs. The measured break-even is approximately 4,700 generated
tokens per candidate. CUDA sweeps therefore enable runner compilation
automatically only when `rows × max_new_tokens >= 5000`. Explicit
`--cuda-compile-runner` and `--no-cuda-compile-runner` override the policy.
