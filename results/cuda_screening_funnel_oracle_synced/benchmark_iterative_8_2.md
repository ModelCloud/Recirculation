# Iterative CUDA benchmark: `8→2`, α=0.20

Workload: Llama 3.2 1B Instruct, 128-token prefill, two timed repetitions.

| Path | Median latency | Relative speed |
|---|---:|---:|
| Sequential CUDA | 3041.38 ms | 1.00× |
| Concurrent CUDA | 2941.96 ms | 1.03× |
| CUDA graph prefill | 620.19 ms | 4.90× vs sequential |

The graph path passed all logits and pending-state accuracy checks with zero
error, including changed-input replay. This is a prefill result; autoregressive
decode still requires a static-cache decode graph to obtain the same gain.

Raw data: `benchmark_iterative_8_2.json`.
