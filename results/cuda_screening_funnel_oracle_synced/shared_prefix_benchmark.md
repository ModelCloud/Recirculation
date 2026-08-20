# Shared-prefix CUDA evaluation benchmark

Workload: Llama 3.2 1B Instruct, GSM8K-Platinum rows 272–273, CUDA SDPA,
candidate `8→2, α=0.20`, 32 maximum generated tokens.

| Mode | Wall time | Relative speed |
|---|---:|---:|
| Full prompt prefill per row | 68 s | 1.00× |
| Shared prefix snapshot | 48 s | 1.42× |

The two prompts share 1,078 tokens. The optimized path prefills those tokens
once for the candidate, snapshots the KV/pending state, and restores it before
each row-specific suffix. Generated result objects were byte-for-byte identical
between modes. The benefit increases with row count because the common prefix
cost is amortized across the full evaluation.
