# Same-token GSM8K-Platinum sweep and evaluation

Date: 2026-08-20 (Asia/Shanghai)

This report records the corrected same-token recirculation search and the locked evaluation used to test the
selected configuration. Results produced before the same-token replay correction are not included as evidence.

## Reproducibility record

| Field | Value |
|---|---|
| Repository commit | `fdccb3ea04c48b5db968138d89da2fa0b5432d78` |
| Model | `meta-llama/Llama-3.2-1B-Instruct` |
| Local dense snapshot | `/Users/diego/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6` |
| Quantization | None; dense Llama 3.2 1B Instruct |
| Dataset | `madrylab/gsm8k-platinum` |
| Dataset configuration | `main` |
| Dataset split | `test` |
| Prompt contract | `configs/gsm8k-platinum-cot-llama.yaml` |
| Prompting | Eight fixed few-shot examples, Llama chat template, greedy generation for evaluation |
| Evaluation scorer | `Evalution==0.0.7`, primary metric `acc,num` |
| Recirculation iteration count | 1 |
| Source normalization | Enabled |
| Ramp | Disabled (`ramp_tokens=0`) |
| Layer indexing | Zero-based; path notation is `source→destination` |

## Dataset partition

Intervals below are half-open in code and inclusive in the human-readable row column.

| Role | Code interval | Dataset rows | Count | Used to choose configuration? |
|---|---:|---:|---:|---|
| Locked evaluation | `[144, 272)` | 144–271 | 128 | No |
| Path search | `[272, 304)` | 272–303 | 32 | Yes, path only |
| Alpha search | `[304, 336)` | 304–335 | 32 | Yes, alpha only |

The three intervals are pairwise disjoint. Evaluation rows 144–271 were never read by either corrected search.
Path selection and alpha selection also use separate rows.

## Search metric and limitation

Candidates are ranked by mean teacher-forced negative log likelihood over the complete tokenized final numeric
answer after the fixed phrase `The final answer is `. Lower is better. The path split contains 37 scored answer
tokens across 32 questions; the alpha split contains 36 across 32 questions. Every final answer is one or two
tokens, and all of its tokens are scored. This is a small screening proxy, not a generated GSM8K accuracy result.
The locked evaluation below is the decision metric.

The answer-token boundary was checked on all 64 search rows by comparing separate phrase/answer tokenization with
joint tokenization; all 64 matched exactly.

## Path search

Configuration:

- Backend: MLX.
- Alpha: `0.10`; implicit convex beta: `0.90`.
- Candidate paths: all deeper-source/shallow-destination pairs emitted by the script with maximum distance 12.
- Number of candidates: 102.
- Shared recirculated prompt prefix: 1,078 tokens.
- Runtime: 3,207.015 seconds.
- Machine-readable artifact: `results/mlx_path_screen_same_token_rows272_303_alpha010.json`.

Exact command:

```bash
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 \
NUMEXPR_NUM_THREADS=16 \
python3 scripts/screen_mlx_recirculation.py \
  --model /Users/diego/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --row-start 272 \
  --rows 32 \
  --forbid-range 0:272 \
  --forbid-range 304:336 \
  --alpha 0.10 \
  --max-distance 12 \
  --output results/mlx_path_screen_same_token_rows272_303_alpha010.json
```

Full ranking:

| Rank | Path | Alpha | Gold-answer NLL | Scored tokens |
|---:|---:|---:|---:|---:|
| 1 | 8→2 | 0.1 | 4.38968818252151 | 37 |
| 2 | 7→6 | 0.1 | 4.420029047373179 | 37 |
| 3 | 7→4 | 0.1 | 4.422155741098765 | 37 |
| 4 | 7→2 | 0.1 | 4.432897722398913 | 37 |
| 5 | 8→4 | 0.1 | 4.43461815086571 | 37 |
| 6 | 7→1 | 0.1 | 4.4349416784338045 | 37 |
| 7 | 6→4 | 0.1 | 4.435334386052312 | 37 |
| 8 | 4→2 | 0.1 | 4.437209876807961 | 37 |
| 9 | 14→3 | 0.1 | 4.4457098471151815 | 37 |
| 10 | 5→2 | 0.1 | 4.45007679913495 | 37 |
| 11 | 3→2 | 0.1 | 4.450633899585621 | 37 |
| 12 | 11→5 | 0.1 | 4.486535149651605 | 37 |
| 13 | 12→3 | 0.1 | 4.487083435058594 | 37 |
| 14 | 12→5 | 0.1 | 4.487871221593909 | 37 |
| 15 | 9→2 | 0.1 | 4.488775768795529 | 37 |
| 16 | 13→4 | 0.1 | 4.4955546920363965 | 37 |
| 17 | 9→6 | 0.1 | 4.499102205843539 | 37 |
| 18 | 10→3 | 0.1 | 4.499889631529112 | 37 |
| 19 | 7→3 | 0.1 | 4.499931180799329 | 37 |
| 20 | 14→5 | 0.1 | 4.50204730678249 | 37 |
| 21 | 4→1 | 0.1 | 4.503159445685309 | 37 |
| 22 | 6→1 | 0.1 | 4.5035931870744035 | 37 |
| 23 | 11→3 | 0.1 | 4.506286930393529 | 37 |
| 24 | 7→5 | 0.1 | 4.506748972712336 | 37 |
| 25 | 9→3 | 0.1 | 4.507441340266047 | 37 |
| 26 | 6→3 | 0.1 | 4.5075616063298405 | 37 |
| 27 | 6→2 | 0.1 | 4.508058238673854 | 37 |
| 28 | 14→4 | 0.1 | 4.512044648866396 | 37 |
| 29 | 3→1 | 0.1 | 4.5142969183019686 | 37 |
| 30 | 10→5 | 0.1 | 4.5161698315594645 | 37 |
| 31 | 8→6 | 0.1 | 4.516427890674488 | 37 |
| 32 | 13→3 | 0.1 | 4.517089946849926 | 37 |
| 33 | 12→2 | 0.1 | 4.520926965249552 | 37 |
| 34 | 12→6 | 0.1 | 4.522196640839448 | 37 |
| 35 | 11→2 | 0.1 | 4.524639284288561 | 37 |
| 36 | 9→4 | 0.1 | 4.525973294232343 | 37 |
| 37 | 11→4 | 0.1 | 4.527590571223079 | 37 |
| 38 | 11→1 | 0.1 | 4.528470941492029 | 37 |
| 39 | 10→2 | 0.1 | 4.529262955124314 | 37 |
| 40 | 13→5 | 0.1 | 4.530222609236434 | 37 |
| 41 | 13→6 | 0.1 | 4.534237732758394 | 37 |
| 42 | 14→6 | 0.1 | 4.535231358296162 | 37 |
| 43 | 12→4 | 0.1 | 4.536320969865129 | 37 |
| 44 | 15→5 | 0.1 | 4.536736874967008 | 37 |
| 45 | 11→6 | 0.1 | 4.542562226991396 | 37 |
| 46 | 9→1 | 0.1 | 4.544707220953864 | 37 |
| 47 | 9→5 | 0.1 | 4.547574996948242 | 37 |
| 48 | 15→3 | 0.1 | 4.5535103050438135 | 37 |
| 49 | 5→4 | 0.1 | 4.560405731201172 | 37 |
| 50 | 10→6 | 0.1 | 4.564934086155247 | 37 |
| 51 | 13→2 | 0.1 | 4.565249984328811 | 37 |
| 52 | 12→1 | 0.1 | 4.5661598927265885 | 37 |
| 53 | 5→1 | 0.1 | 4.572842984586148 | 37 |
| 54 | 8→1 | 0.1 | 4.575114121308198 | 37 |
| 55 | 8→3 | 0.1 | 4.576117077389279 | 37 |
| 56 | 8→5 | 0.1 | 4.576621906177418 | 37 |
| 57 | 4→3 | 0.1 | 4.58252984124261 | 37 |
| 58 | 6→5 | 0.1 | 4.593311412914379 | 37 |
| 59 | 15→14 | 0.1 | 4.594628514470281 | 37 |
| 60 | 10→4 | 0.1 | 4.596342550741659 | 37 |
| 61 | 2→1 | 0.1 | 4.60057562750739 | 37 |
| 62 | 14→2 | 0.1 | 4.604590029329867 | 37 |
| 63 | 14→13 | 0.1 | 4.605214866431984 | 37 |
| 64 | 13→9 | 0.1 | 4.606476036277977 | 37 |
| 65 | 15→13 | 0.1 | 4.607373314934808 | 37 |
| 66 | 13→11 | 0.1 | 4.610812780019399 | 37 |
| 67 | 10→1 | 0.1 | 4.614841925131308 | 37 |
| 68 | 14→8 | 0.1 | 4.616376773731129 | 37 |
| 69 | 12→9 | 0.1 | 4.617408082291886 | 37 |
| 70 | 13→1 | 0.1 | 4.625000206199852 | 37 |
| 71 | 14→9 | 0.1 | 4.625993367787954 | 37 |
| 72 | 14→11 | 0.1 | 4.628409205256282 | 37 |
| 73 | 5→3 | 0.1 | 4.631445394979941 | 37 |
| 74 | 13→12 | 0.1 | 4.634300902083114 | 37 |
| 75 | 13→8 | 0.1 | 4.638158076518291 | 37 |
| 76 | 9→8 | 0.1 | 4.638385257205448 | 37 |
| 77 | 11→9 | 0.1 | 4.640365652135901 | 37 |
| 78 | 14→12 | 0.1 | 4.642643902752851 | 37 |
| 79 | 12→11 | 0.1 | 4.644893388490419 | 37 |
| 80 | 14→7 | 0.1 | 4.646436588184254 | 37 |
| 81 | 13→10 | 0.1 | 4.65273480801969 | 37 |
| 82 | 10→8 | 0.1 | 4.663977597210859 | 37 |
| 83 | 14→10 | 0.1 | 4.668589772404851 | 37 |
| 84 | 8→7 | 0.1 | 4.688856073327966 | 37 |
| 85 | 10→9 | 0.1 | 4.691964896949562 | 37 |
| 86 | 9→7 | 0.1 | 4.698226258561418 | 37 |
| 87 | 12→8 | 0.1 | 4.700368314175992 | 37 |
| 88 | 15→6 | 0.1 | 4.703805253312394 | 37 |
| 89 | 15→12 | 0.1 | 4.704041970742716 | 37 |
| 90 | 10→7 | 0.1 | 4.704545923181482 | 37 |
| 91 | 11→10 | 0.1 | 4.712610554050755 | 37 |
| 92 | 13→7 | 0.1 | 4.720846536997202 | 37 |
| 93 | 15→7 | 0.1 | 4.726475483662373 | 37 |
| 94 | 15→8 | 0.1 | 4.731550732174435 | 37 |
| 95 | 11→8 | 0.1 | 4.73513835185283 | 37 |
| 96 | 15→9 | 0.1 | 4.73905779864337 | 37 |
| 97 | 12→10 | 0.1 | 4.750697471000053 | 37 |
| 98 | 12→7 | 0.1 | 4.757440979416306 | 37 |
| 99 | 11→7 | 0.1 | 4.764627250465187 | 37 |
| 100 | 15→11 | 0.1 | 4.854325165619722 | 37 |
| 101 | 15→10 | 0.1 | 4.866805308573955 | 37 |
| 102 | 15→4 | 0.1 | 4.9166299974596175 | 37 |

## Alpha search

Only the top five paths from the disjoint path search were tested.

Configuration:

- Backend: MLX.
- Paths: `8→2`, `7→6`, `7→4`, `7→2`, `8→4`.
- Alphas: `0.025`, `0.05`, `0.075`, `0.10`, `0.125`, `0.15`, `0.20`.
- Implicit beta for each candidate: `1 - alpha`.
- Number of candidates: 35.
- Shared recirculated prompt prefix: 1,078 tokens.
- Runtime: 1,182.615 seconds.
- Machine-readable artifact: `results/mlx_alpha_screen_same_token_rows304_335_top5.json`.

Exact command:

```bash
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 \
NUMEXPR_NUM_THREADS=16 \
python3 scripts/screen_mlx_recirculation.py \
  --model /Users/diego/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --row-start 304 \
  --rows 32 \
  --forbid-range 0:304 \
  --alpha 0.025 \
  --alpha 0.05 \
  --alpha 0.075 \
  --alpha 0.10 \
  --alpha 0.125 \
  --alpha 0.15 \
  --alpha 0.20 \
  --path 8:2 \
  --path 7:6 \
  --path 7:4 \
  --path 7:2 \
  --path 8:4 \
  --output results/mlx_alpha_screen_same_token_rows304_335_top5.json
```

Full ranking:

| Rank | Path | Alpha | Gold-answer NLL | Scored tokens |
|---:|---:|---:|---:|---:|
| 1 | 8→2 | 0.2 | 4.1477419005499945 | 36 |
| 2 | 7→2 | 0.2 | 4.247136857774523 | 36 |
| 3 | 8→2 | 0.15 | 4.279179837968615 | 36 |
| 4 | 7→6 | 0.2 | 4.284699069129096 | 36 |
| 5 | 7→2 | 0.15 | 4.34487862057156 | 36 |
| 6 | 7→6 | 0.15 | 4.357976648542616 | 36 |
| 7 | 8→2 | 0.125 | 4.363093217213948 | 36 |
| 8 | 7→6 | 0.125 | 4.402273125118679 | 36 |
| 9 | 8→2 | 0.1 | 4.411689652336968 | 36 |
| 10 | 7→2 | 0.125 | 4.42764155069987 | 36 |
| 11 | 7→6 | 0.1 | 4.447360833485921 | 36 |
| 12 | 7→4 | 0.2 | 4.449575848049587 | 36 |
| 13 | 7→4 | 0.15 | 4.473520437876384 | 36 |
| 14 | 7→2 | 0.1 | 4.483326858944363 | 36 |
| 15 | 7→4 | 0.125 | 4.502187305026585 | 36 |
| 16 | 7→6 | 0.075 | 4.510033342573378 | 36 |
| 17 | 8→2 | 0.075 | 4.529401779174805 | 36 |
| 18 | 7→4 | 0.1 | 4.543781492445204 | 36 |
| 19 | 7→2 | 0.075 | 4.549839231703016 | 36 |
| 20 | 7→4 | 0.075 | 4.5737313694424095 | 36 |
| 21 | 7→6 | 0.05 | 4.581622494591607 | 36 |
| 22 | 7→2 | 0.05 | 4.588801436954075 | 36 |
| 23 | 8→2 | 0.05 | 4.592340098487006 | 36 |
| 24 | 8→4 | 0.15 | 4.617257277170817 | 36 |
| 25 | 7→4 | 0.05 | 4.622191535101996 | 36 |
| 26 | 8→4 | 0.2 | 4.630086898803711 | 36 |
| 27 | 8→4 | 0.125 | 4.637576738993327 | 36 |
| 28 | 8→4 | 0.1 | 4.645205974578857 | 36 |
| 29 | 8→4 | 0.075 | 4.663922097947863 | 36 |
| 30 | 7→6 | 0.025 | 4.671443462371826 | 36 |
| 31 | 8→4 | 0.05 | 4.671669377221002 | 36 |
| 32 | 7→4 | 0.025 | 4.6787429915534124 | 36 |
| 33 | 8→2 | 0.025 | 4.689200560251872 | 36 |
| 34 | 7→2 | 0.025 | 4.699209372202556 | 36 |
| 35 | 8→4 | 0.025 | 4.722190962897407 | 36 |

Selected configuration: **source layer 8 → destination layer 2, alpha 0.20, beta 0.80**.

## Locked 128-row generated evaluation

Status: complete.

The evaluator runs the dense non-recirculated baseline and corrected recirculation arm on each identical prompt,
using greedy generation and a 256-token generation limit. Primary scoring imports Evalution 0.0.7's
format-insensitive GSM8K-Platinum numeric target, answer extractor, and numeric equality functions. Legacy strict
and flexible extraction is retained only as a diagnostic. The output artifact is
`results/gsm8k_platinum_same_token_8_2_alpha020_rows144_271.json`.

Exact command:

```bash
PYTHONUNBUFFERED=1 \
OMP_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
BLIS_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 \
NUMEXPR_NUM_THREADS=16 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 scripts/eval_gsm8k_platinum.py \
  --model /Users/diego/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 \
  --device mps \
  --row-start 144 \
  --rows 128 \
  --forbid-range 272:304 \
  --forbid-range 304:336 \
  --max-new-tokens 256 \
  --source-layer 8 \
  --destination-layer 2 \
  --alpha 0.20 \
  --ramp-tokens 0 \
  --report-every 4 \
  --output results/gsm8k_platinum_same_token_8_2_alpha020_rows144_271.json
```

| Arm | Correct/128 | Accuracy | Delta vs baseline |
|---|---:|---:|---:|
| Dense baseline | 59 | 46.09% | — |
| Recirculation 8→2, alpha 0.20 | 67 | 52.34% | **+6.25 percentage points** |

Relative accuracy increased by **13.56%** and errors decreased by **11.59%**. Evaluation took 3,902.959 seconds
(65.05 minutes).

| Paired transition | Rows |
|---|---:|
| Wrong → correct | 16 |
| Correct → wrong | 8 |
| Net additional correct | **8** |
| Generated answer changed | 65 |

The exact two-sided paired McNemar p-value is `0.1516`. The paired bootstrap percentile 95% interval for the accuracy
change is `[-0.78, +14.06]` percentage points. The primary result is therefore a positive point estimate but is
**noise-consistent**, not a conclusive accuracy gain at 128 rows.
