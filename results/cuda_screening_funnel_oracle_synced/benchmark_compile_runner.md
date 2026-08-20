# Compiled recirculation runner benchmark

Warm workload: Llama 3.2 1B Instruct, SDPA, 128-token prompt, 16 generated
tokens, two repetitions. Model load and compiler warmup are excluded from the
reported decode timings.

| Runner | Median throughput |
|---|---:|
| Eager CUDAConcurrentRunner | ~5.0 tok/s |
| Compiled lower/upper stacks (`triton.cudagraphs=False`) | 11.17 tok/s |

The compiled runner is approximately 2.2× faster in warm decode. Compilation
has substantial one-time overhead, so it remains opt-in via
`--cuda-compile-runner` and should be used for sufficiently large sweeps.
