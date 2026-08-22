# CUDA Recirculation Screening Ledger

- Status: complete
- Complete / active / pending / total: 7 / 0 / 0 / 7
- Implementation commit: `c32e388d49975f7e1240306b84b606ad6a2b1bd9`
- Promotion population: all 7 completed candidates in this ledger

## Current leaders

| Ranking | Path | Alpha | Metric |
|---|---:|---:|---:|
| language_modeling_perplexity | 10→1 | 0.04 | 14.6784564 |
| language_modeling_robust | 10→1 | 0.04 | 0.0032032362 |
| final_answer_perplexity | — | — | — |
| final_answer_robust | — | — | — |
| full_solution_perplexity | — | — | — |
| full_solution_robust | — | — | — |

## All completed candidates

The promotion shortlist is recalculated from this entire table, never only from the most recent result.

| Scan | Path | Alpha | Seconds | LM PPL | LM ΔNLL | LM robust | LM tail harm | LM I/R/N |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10→1 | 0.04 | 710.067 | 14.6784564 | -0.00120021109 | 0.0032032362 | 0.00440344729 | 1013/459/0 |
| 1 | 10→1 | 0.08 | 709.963 | 14.6830577 | -0.000886788753 | 0.00952843373 | 0.0104152225 | 874/598/0 |
| 2 | 10→1 | 0.12 | 710.229 | 14.7085105 | 0.000845190987 | 0.0189657537 | 0.0181205627 | 713/759/0 |
| 3 | 10→1 | 0.16 | 710.562 | 14.7546428 | 0.00397671594 | 0.0315434391 | 0.0275667232 | 538/934/0 |
| 4 | 10→1 | 0.2 | 710.469 | 14.822115 | 0.00853924054 | 0.0474220362 | 0.0388827957 | 382/1090/0 |
| 5 | 10→1 | 0.24 | 710.456 | 14.9149207 | 0.0147810219 | 0.0674838636 | 0.0527028417 | 249/1223/0 |
| 6 | 10→1 | 0.28 | 709.125 | 15.038075 | 0.0230042358 | 0.0923478919 | 0.0693436561 | 143/1329/0 |

## Candidate schedule

Order: `sequential`; seed: `20260821`.

| Scan | Path | Alpha | State |
|---:|---:|---:|---:|
| 0 | 10→1 | 0.04 | complete |
| 1 | 10→1 | 0.08 | complete |
| 2 | 10→1 | 0.12 | complete |
| 3 | 10→1 | 0.16 | complete |
| 4 | 10→1 | 0.2 | complete |
| 5 | 10→1 | 0.24 | complete |
| 6 | 10→1 | 0.28 | complete |
