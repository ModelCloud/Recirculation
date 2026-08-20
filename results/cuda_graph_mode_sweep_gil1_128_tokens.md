# CUDA Graph mode and priority sweep

Benchmark date: 2026-08-20 UTC

Sweep implementation commit: `819567f3cab5341f61c11b9c69b419639f367817`

All variants use Llama 3.2 1B Instruct, FP16, 128 tokens, three capture warmups, seven timed post-capture replays,
and the same two-stream recirculation schedule on the NVIDIA PG506-230. Every successful variant produced zero
changed-input logits and pending-state error against the eager reference (`0.0 < 2e-3`).

| Variant | Median | Standard deviation | Difference from default |
| --- | ---: | ---: | ---: |
| Global, automatic, default priorities | 513.375 ms | 0.401 ms | baseline |
| Global, manual instantiation | 513.384 ms | 0.422 ms | +0.002% |
| Thread-local, automatic | 513.547 ms | 0.363 ms | +0.033% |
| Relaxed, automatic | 513.217 ms | 0.185 ms | -0.031% |
| Global, high-priority capture | 513.483 ms | 0.830 ms | +0.021% |
| Global, high-priority branches | **512.977 ms** | 0.293 ms | **-0.078%** |
| Global, high-priority launch | 513.578 ms | 0.475 ms | +0.039% |
| Global, manual, all high priority | 513.536 ms | 0.302 ms | +0.031% |

A reverse-order 21-repetition confirmation measured high-priority branches at 513.372 ms and default branches at
513.499 ms, a nominal `1.000249x` speedup (0.025%). The difference is smaller than run-to-run dispersion and is not
evidence of a material performance improvement. High-priority lower/replay streams were nevertheless promoted as the
default because they remained the fastest accuracy-gated configuration in both runs and do not weaken global capture
safety.

Capture safety modes and retained/manual graph instantiation affect capture semantics or first-replay latency, not the
steady-state graph topology, and showed no meaningful post-capture performance difference here. Production capture
therefore remains `global`, automatic, and protected by the process-wide capture lock.

Raw artifacts:

- [`cuda_graph_mode_sweep_gil1_128_tokens.json`](cuda_graph_mode_sweep_gil1_128_tokens.json)
- [`cuda_graph_priority_confirmation_gil1_128_tokens.json`](cuda_graph_priority_confirmation_gil1_128_tokens.json)
