# StaticCache benchmark

Identical workload: one GSM8K row, CUDA, `8→2, α=0.20`, eight generated
tokens, batch size 1. Timings include model and dataset startup.

| Cache | Wall time |
|---|---:|
| DynamicCache (default) | 42 s |
| StaticCache (`--cuda-static-cache`) | 58 s |

StaticCache is currently 38% slower on this workload, likely because its
initial allocation/shape setup dominates this short run. It remains opt-in and
must not be promoted as the default without a longer decode-only benchmark.
