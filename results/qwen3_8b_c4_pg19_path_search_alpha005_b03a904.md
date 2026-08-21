# Qwen3-8B CUDA path search at alpha 0.05

This report records the completed staged path search for the local Qwen3-8B checkpoint. It is a tuning result, not a
held-out generation evaluation. The run compared paths at the fixed alpha `0.05`; it did not sweep other alpha values.

## Provenance and configuration

- Status: complete (`16/16` Stage 3 finalists)
- Implementation commit: `b03a904d718adbca2469c7e5a6cb7403c0ce32dd`
- Model: `/local-models/Qwen3-8B` (36 decoder layers)
- Device: NVIDIA PG506-230
- Software: Python 3.14.7 with GIL disabled, PyTorch 2.13.0+cu130, CUDA 13.0
- Precision: FP16
- Data: 256 C4 windows plus 256 PG19 windows
- Stage sizes: 32 × 256-token windows, 128 × 512-token windows, then 512 × 1024-token windows
- Funnel: all 630 valid `source > destination` paths → 64 candidates → 16 finalists
- Alpha: `0.05`
- Stage 3 target tokens: 523,776
- Native Stage 3 baseline: NLL `2.5595603240`, perplexity `12.9301309977`
- Candidate order: randomized with seed `20260823`
- Attention backend: eager; dual GEMM enabled
- Requested/actual candidate batch size: 8/1 (adapted to memory constraints)
- Requested/actual row batch size: 128/64
- Stage 3 runtime: 14,937.93 seconds

The plain ranking minimizes target perplexity. The robust ranking minimizes `native_delta_nll + tail_harm_nll`, where
tail harm is the worst-tail positive per-row NLL delta at quantile 0.9 with weight 1.0. Promotion was recalculated from
the full completed population rather than only against the previously promoted candidate.

## Stage leaders

| Stage | Population | Rows | Tokens/window | Plain-PPL leader | PPL | Robust leader | Robust score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 630 | 32 | 256 | 35→31 | 15.3820828 | 35→31 | -0.0126862769 |
| 2 | 64 | 128 | 512 | 35→10 | 14.2541056 | 35→10 | -0.0118558467 |
| 3 | 16 | 512 | 1024 | **35→10** | **12.7821939** | **35→21** | **-0.00987030921** |

Absolute perplexities differ between stages because each stage uses a different sample size and context length. Only
candidates within the same stage should be compared directly.

## All Stage 3 scores

| PPL rank | Path | Alpha | PPL | Delta NLL | Tail harm | Robust score | Improved | Regressed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35→10 | 0.05 | **12.7821939** | -0.0115072222 | 0.00221874743 | -0.00928847481 | 504 | 8 |
| 2 | 35→17 | 0.05 | 12.7839511 | -0.0113697605 | 0.00185690709 | -0.00951285344 | 506 | 6 |
| 3 | 35→12 | 0.05 | 12.7842210 | -0.0113486471 | 0.00188124840 | -0.00946739870 | 504 | 8 |
| 4 | 35→23 | 0.05 | 12.7849319 | -0.0112930415 | 0.00171859928 | -0.00957444223 | 505 | 7 |
| 5 | 35→8 | 0.05 | 12.7874555 | -0.0110956703 | 0.00193906865 | -0.00915660161 | 502 | 10 |
| 6 | 35→13 | 0.05 | 12.7882429 | -0.0110340961 | 0.00170373404 | -0.00933036202 | 505 | 7 |
| 7 | 35→6 | 0.05 | 12.7882822 | -0.0110310281 | 0.00215887484 | -0.00887215322 | 501 | 11 |
| 8 | 35→19 | 0.05 | 12.7885319 | -0.0110115008 | 0.00195014013 | -0.00906136067 | 504 | 8 |
| 9 | 35→7 | 0.05 | 12.7894791 | -0.0109374399 | 0.00189208812 | -0.00904535181 | 499 | 13 |
| 10 | 35→22 | 0.05 | 12.7897770 | -0.0109141439 | 0.00119605087 | -0.00971809300 | 506 | 6 |
| 11 | 35→18 | 0.05 | 12.7911053 | -0.0108102966 | 0.00191308430 | -0.00889721231 | 504 | 8 |
| 12 | 35→21 | 0.05 | 12.7916903 | -0.0107645621 | 0.000894252927 | **-0.00987030921** | 505 | 7 |
| 13 | 35→20 | 0.05 | 12.7928963 | -0.0106702871 | 0.00211992643 | -0.00855036062 | 505 | 7 |
| 14 | 35→11 | 0.05 | 12.7945373 | -0.0105420210 | 0.00228610083 | -0.00825592017 | 503 | 9 |
| 15 | 35→28 | 0.05 | 12.7945554 | -0.0105406030 | 0.00146767774 | -0.00907292528 | 506 | 6 |
| 16 | 35→9 | 0.05 | 12.7973375 | -0.0103231805 | 0.00315598318 | -0.00716719736 | 501 | 11 |

## Selection outcome

- Plain perplexity selection: path `35→10`, alpha `0.05`.
- Harm-penalized selection: path `35→21`, alpha `0.05`.
- Every finalist improved aggregate NLL versus the shared native baseline.
- These candidates still require disjoint natural-generation evaluation before use as a default inference setting.

