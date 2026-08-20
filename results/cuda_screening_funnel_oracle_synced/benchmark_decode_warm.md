# Warm CUDA decode benchmark

Workload: Llama 3.2 1B Instruct, SDPA, 128-token prompt, 16 generated tokens,
two warmed repetitions. Timings exclude model loading.

| Scheduler | Median | Tokens/s |
|---|---:|---:|
| Python threaded | 3.229 s | 4.955 |
| Direct two-stream | 3.224 s | 4.963 |

The direct scheduler is only 0.16% faster, below measurement noise. No default
change is justified; both remain selectable with `--cuda-python-threads` and
`--no-cuda-python-threads`.
