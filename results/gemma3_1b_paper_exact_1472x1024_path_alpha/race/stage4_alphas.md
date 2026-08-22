# CUDA Recirculation Screening Ledger

- Status: complete
- Complete / active / pending / total: 7 / 0 / 0 / 7
- Implementation commit: `e5b6197962c038359728c3fac8cf8e2962eb279b`
- Promotion population: all 7 completed candidates in this ledger

## Current leaders

| Ranking | Path | Alpha | Metric |
|---|---:|---:|---:|
| language_modeling_perplexity | 25→20 | 0.12 | 27.2862217 |
| language_modeling_robust | 25→20 | 0.04 | -0.0030982008 |
| final_answer_perplexity | — | — | — |
| final_answer_robust | — | — | — |
| full_solution_perplexity | — | — | — |
| full_solution_robust | — | — | — |

## All completed candidates

The promotion shortlist is recalculated from this entire table, never only from the most recent result.

| Scan | Path | Alpha | Seconds | LM PPL | LM ΔNLL | LM robust | LM tail harm | LM I/R/N |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 25→20 | 0.04 | 673.640 | 27.399767 | -0.00634379373 | -0.0030982008 | 0.00324559293 | 1320/152/0 |
| 1 | 25→20 | 0.08 | 675.658 | 27.2884836 | -0.0104135392 | -0.00297224734 | 0.00744129189 | 1290/182/0 |
| 2 | 25→20 | 0.12 | 658.624 | 27.2862217 | -0.0104964292 | 0.00749709803 | 0.0179935272 | 1182/290/0 |
| 3 | 25→20 | 0.16 | 666.603 | 27.3827917 | -0.00696352858 | 0.0279593991 | 0.0349229277 | 1017/455/0 |
| 4 | 25→20 | 0.2 | 669.514 | 27.6198901 | 0.00165786575 | 0.0628905707 | 0.061232705 | 770/702/0 |
| 5 | 25→20 | 0.24 | 672.596 | 27.9467601 | 0.0134229718 | 0.107529978 | 0.0941070059 | 589/883/0 |
| 6 | 25→20 | 0.28 | 672.911 | 28.3785832 | 0.0287564447 | 0.161897663 | 0.133141218 | 425/1047/0 |

## Candidate schedule

Order: `sequential`; seed: `20260821`.

| Scan | Path | Alpha | State |
|---:|---:|---:|---:|
| 0 | 25→20 | 0.04 | complete |
| 1 | 25→20 | 0.08 | complete |
| 2 | 25→20 | 0.12 | complete |
| 3 | 25→20 | 0.16 | complete |
| 4 | 25→20 | 0.2 | complete |
| 5 | 25→20 | 0.24 | complete |
| 6 | 25→20 | 0.28 | complete |
