# GIL-enabled dual-stream CUDA Graph benchmark

Benchmark date: 2026-08-20 UTC

Implementation commit: `b8777b57e94ae4f6e204c57f48e99b4675edbdd7`

## Result

| Recirculation mode | Median latency, 128 tokens | Throughput | Speedup vs. sequential |
| --- | ---: | ---: | ---: |
| Sequential eager CUDA | 2,700.498 ms | 47.40 tokens/s | 1.000x |
| Dual-stream eager CUDA | 2,561.898 ms | 49.96 tokens/s | 1.054x |
| Dual-stream CUDA Graph | 513.439 ms | 249.30 tokens/s | **5.260x** |

On this `GIL=1` runtime, eager dual-stream scheduling reduced latency by 5.13% relative to sequential recirculation.
Capturing the complete dual-stream schedule reduced latency by 80.99% relative to sequential and by 79.96% relative
to eager dual-stream execution. The graphed path was 4.990x faster than eager dual-stream execution.

## Accuracy gate

All measured error rates were `0.0`, below the release limit of `2e-3`:

- Sequential versus eager dual-stream logits: 0.0
- Sequential versus eager dual-stream pending state: 0.0
- Eager versus graphed dual-stream logits: 0.0
- Eager versus graphed dual-stream pending state: 0.0
- Changed-input eager versus graphed logits: 0.0
- Changed-input eager versus graphed pending state: 0.0

The changed-input check uses a one-token roll of the captured input to verify that replay consumes the updated static
token buffer instead of accidentally reusing capture-time token values.

## Method

- Model: `/local-models/Llama-3.2-1B-Instruct`, FP16
- Input: 128 tokens, batch size 1
- Recirculation: source layer 12, destination layer 5, alpha 0.1, no ramp
- Timed repetitions: 5 per mode, with CUDA synchronization around each complete prefill
- CUDA Graph capture: three warmups; warmup and global-mode capture protected by the process-wide capture lock
- Python worker threads: 2
- CUDA branch streams: 2
- GIL: enabled

Median samples are calculated from the raw values in
[`cuda_concurrent_graph_gil1_128_tokens.json`](cuda_concurrent_graph_gil1_128_tokens.json).

## Environment

- GPU: NVIDIA PG506-230, 98,304 MiB
- NVIDIA driver: 610.43.02
- CUDA reported by PyTorch: 13.0
- PyTorch: 2.15.0.dev20260817+cu130
- Python: 3.14.6, Clang 22.1.3
- Implementation commit: `b8777b57e94ae4f6e204c57f48e99b4675edbdd7`

CUDA Graph replay requires the captured token shape and stable virtual memory addresses. These numbers therefore
describe the fixed-shape 128-token path and do not include one-time graph warmup or capture latency.
