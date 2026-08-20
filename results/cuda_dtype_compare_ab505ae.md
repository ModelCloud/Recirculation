# CUDA FP16 versus BF16 perplexity

- Implementation commit: `ab505ae13505864c72c27c6c0420930c3be7159e`
- Model: `/local-models/Llama-3.2-1B-Instruct`
- Data: the same frozen 256 C4 + 256 PG19 windows
- Window length: 1,024 tokens
- Scored target tokens: 523,776
- Prompt formatting: raw corpus tokens with BOS; no chat template

| Dtype | NLL | Perplexity | Runtime |
|---|---:|---:|---:|
| FP16 | 2.9259035981 | 18.6510715100 | 232.970 s |
| BF16 | 2.9262037073 | 18.6566697095 | 231.127 s |

Relative to FP16, BF16 increased NLL by `0.0003001093` and perplexity by `0.0055981994` (`0.0300154%`). BF16 lowered row-average NLL on 201 windows and raised it on 311 windows. It was `0.791%` faster, but did not improve the path-search objective. The path search should therefore remain FP16 for this model, software stack, and GPU unless a broader accuracy evaluation establishes a BF16 advantage.

The paired raw baseline artifacts are `cuda_dtype_compare_ab505ae_fp16_baseline.json` and `cuda_dtype_compare_ab505ae_bf16_baseline.json`.
