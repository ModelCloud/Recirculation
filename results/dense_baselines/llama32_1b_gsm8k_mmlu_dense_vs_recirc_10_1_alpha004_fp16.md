# Llama 3.2 1B dense baseline versus recirculation

This report records the matched dense baseline for the completed Llama 3.2 1B
Instruct recirculation evaluation. The intended model-math difference is only
that the dense arm disables path `10→1`, `α=0.04`, `β=0.96` recirculation.

## Results

| Suite | Rows | Dense correct | Dense accuracy | Recirculation correct | Recirculation accuracy | Recirculation delta |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K Platinum | 1,209 | 588 | 48.6352% | 597 | 49.3797% | +9 / +0.7444 pp |
| MMLU-STEM | 3,153 | 1,263 | 40.0571% | 1,268 | 40.2157% | +5 / +0.1586 pp |
| MMLU-Humanities | 4,705 | 2,027 | 43.0818% | 2,034 | 43.2306% | +7 / +0.1488 pp |

The suites have different scoring contracts and their counts must not be combined
into one benchmark accuracy. On the 4,705 aligned Humanities rows, recirculation
caused 42 wrong→correct flips and 35 correct→wrong flips, for a paired net of +7.
All prompt/target identities aligned; 135 predicted choices changed.

Dense total runtime was 245.08 seconds for 9,067 rows (36.996 rows/s). Dense
GSM8K generation took 87.874 seconds versus 701.00 seconds in the earlier
recirculation run, an observed 7.98× dense-generation speed advantage. MMLU uses
likelihood scoring, so this generation speed comparison does not apply to MMLU.

## Exact command used

The following is the literal command executed from the repository root to create
the matched dense artifact. The MMLU suite batches are explicit because the saved
recirculation artifact resolved both of them to 128.

```bash
PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -X gil=0 scripts/evaluate.py run \
  --model /local-models/Llama-3.2-1B-Instruct \
  --device cuda \
  --benchmark gsm8k_platinum \
  --benchmark mmlu_stem \
  --benchmark mmlu_humanities \
  --benchmark-arg gsm8k_platinum.variant=cot_llama \
  --benchmark-arg gsm8k_platinum.apply_chat_template=true \
  --benchmark-arg mmlu_stem.batch_size=128 \
  --benchmark-arg mmlu_humanities.batch_size=128 \
  --report-every-seconds 60 \
  --output results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_fp16_matched_batch128.json
```

The earlier recirculation commands are preserved verbatim in
`results/evaluations/llama32_1b_gsm8k_mmlu_recirc_10_1_alpha004_fp16_completed.md`.
Those commands left MMLU batch size at its then-current default; the saved
provenance confirms that it resolved to 128. The explicit dense arguments above
pin the same semantic configuration despite subsequent runner changes.

## Resolved runtime configuration

| Setting | Value |
|---|---|
| Repository commit used for dense execution | `f16c2de8cd9246e11d5e9d0e89913d4e88e17b43` |
| Python | 3.14.7 free-threading build; GIL disabled |
| Torch / Transformers / FlashAttention | 2.13.0 / 5.15.1 / 2.8.3.post1 |
| Evalution / Tokenicer | 0.0.12 / 0.0.14 |
| Device | NVIDIA PG506-230, 98,304 MiB; driver 610.43.02 |
| Precision | FP16 |
| Attention | paged FlashAttention 2 |
| Continuous batching | enabled |
| Engine generation batch | 32 |
| MMLU frontend suite batch | 128 |
| MMLU unique-prompt CUDA cohort | 512 |
| MMLU A/B/C/D choice requests | 2,048 |
| MMLU padded-token cohort budget | 524,288 |
| CUDA graph | disabled |
| Paged cache | 256 blocks × 256 tokens; 8 blocks/request |
| Scheduler token budget | 65,536 |
| Allocator | `expandable_segments:True,garbage_collection_threshold:0.8` |
| GSM8K | `cot_llama`, chat template enabled, 8-shot, natural greedy generation, 256-token limit |
| MMLU | five-shot A/B/C/D log-likelihood |

The local dataset snapshots used were GSM8K Platinum cache revision
`e762492455a1cf7967de89f05b6bef72fc713b66` and MMLU cache revision
`c30699e8356da336a370243923dbaf21066bb9fe`.

## Artifacts and integrity

- Dense full output (local):
  `results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_fp16_matched_batch128.json`
- Dense output SHA-256:
  `fb54955ff6d89f9d8cb9fc3b8c7f5f20fc065d077d153e358da0d08746c47e6c`
- Dense status (local):
  `results/dense_baselines/llama32_1b_gsm8k_mmlu_dense_fp16_matched_batch128.status.json`
- Dense status SHA-256:
  `93d00fb8c2076f7d5d7dafa425e80cda710b791848db1d22ff8f2de3db44b78c`
- Recirculation report:
  `results/evaluations/llama32_1b_gsm8k_mmlu_recirc_10_1_alpha004_fp16_completed.md`
- Recirculation Humanities full output (local):
  `results/evaluations/llama32_1b_mmlu_humanities_recirc_10_1_alpha004_fp16_length_bucketed.json`

The full JSON files contain prompts and generated text and are intentionally not
added to Git history; this Markdown report and the compact dense status JSON are
the durable repository records.

## Repeatability caveat

Continuous-batched greedy GSM8K was not bit-for-bit stable while the engine seed
was left unset, as it was in the recirculation run. A dense preflight with the
same generation settings scored 591/1,209; the final batch-matched run scored
588/1,209. The two runs differed on 50 generated strings and five row scores.
Both MMLU suites reproduced exactly. Therefore the one-shot GSM8K improvement is
reported faithfully but should not be interpreted as deterministic without a
future paired, explicitly seeded rerun.
