# CUDA stream scheduler benchmark

Workload: Llama 3.2 1B Instruct, CUDA, GSM8K-Platinum row 272, one row,
`max_new_tokens=8`, candidate `8→2, α=0.20`, SDPA attention.

| Scheduler | Wall time |
|---|---:|
| `--cuda-python-threads` | 41 s |
| `--no-cuda-python-threads` | 43 s |

The threaded scheduler is currently retained as the default, with a small
measured advantage on this device. Results include model load and dataset
startup overhead; a longer decode-only benchmark is still needed before
claiming a stable percentage improvement.
