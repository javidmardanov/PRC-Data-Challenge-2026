# Three-agent RMSE research loop

This adapts Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
experiment/measure/retain loop and the [RLM](https://github.com/alexzhang13/rlm)
idea of keeping context in files, inspecting it with code, and delegating bounded
queries. It is an agent workflow using Codex tools, not a trained RLM or the
upstream RLM runtime. Three GPT-5.6-sol workers report to one reviewing parent;
the parent recursively narrows follow-up questions using measured errors.

## Shared experiment contract

- Workspace: C:/Users/javid/Desktop/PRC-Data-Challenge-2026. Python:
  C:/Users/javid/anaconda3/python.exe. PowerShell. Read ponytail SKILL.md.
- Read only relevant source and report fields. Keep bulky context, predictions,
  and trial logs on disk. Return compact metrics, paths, and next hypotheses.
- Train development on 2025 excluding months 7, 11, 12. July and November are
  repeatedly used tuning data, not unbiased test estimates. Never inspect or
  score December for new selection. Never use ranking labels/official scores to
  fit parameters. Final fits may use all 2025 after parameters are frozen.
- Keep frozen v1 models/configs/predictions intact. All trials get new names.
  Feature inputs must exclude departure BLOCK/TARGET/IDs used as predictors.
  Observed arrival taxi and ranking-available timestamps are allowed.
- Score complete, ID-aligned, routed predictions, and report July/November
  separately. Raw known-only model outputs on missing rows are placeholders.
- Minimize MSE/RMSE with squared-error training. Use analytic derivatives or
  constrained optimization for continuous blend weights, bounded parameter
  search for trees. No claimed monotonic improvement: retain the incumbent
  when a trial loses. Report all attempted trials, including failures.
- Within July and November, separate calibration dates from check dates before
  fitting new weights; report cross-period transfer where practical. This
  reduces selection optimism but does not create an untouched test set.
- CPU-only worker thread_count <= 3; the leased GPU trainer may use 8 CPU threads.
  OPENBLAS_NUM_THREADS=1. Own processes may run
  AboveNormal. Never terminate unrelated processes. One GPU job at a time;
  parent grants ownership. No external API credentials or extra model services.
- No worker commits, pushes, submissions, or external messages. Parent integrates,
  reviews, validates, publishes source and submits the selected candidate.
- One round: at most 4 informative trials, then compact report to parent.
  Continue via bounded follow-up rounds. No recursive agent spawning: exactly
  the three requested workers are sufficient on this hardware.

## Context at the start of this research round

Read README.md, docs/solution.md as needed. Frozen complete v1: local
266.052234, July 297.130401, November 224.100558; official v1 285.0706
(last recorded leader 246.7053, rank 14/81). Winning is not guaranteed.

Base features artifacts/features.parquet and rows.parquet; external features
data/processed/timesfm_hourly_dev.parquet (final counterpart exists),
weather_hourly.parquet, queue_features.parquet, timing_bounds.parquet.
scripts/train.py handles residual CatBoost, --known-only, --sample, --final,
--tfm, --weather, --extra, --interactions, --boost-priority. GPU is RTX3060 12GB.
scripts/ensemble.py contains ID alignment, simplex optimization, missing and
Rome routing and bounds. Inspect actual schemas/functions before reusing them.

V1 order: main weighted blend (tfm_queue_d8, no_tfm_full_d8, lgbm_tfm), missing
specialist, partial Rome identity expert for schedule delta <24000, nonnegative
clip, timing bounds. artifacts/v1_ensemble.json is frozen; docs copy uses relative
paths. Saved *_predictions.parquet are base validation; final counterparts and
submission v1 exist. Preserve exactly 344841 ranking IDs in template order.

Missing v2: scripts/missing_v2.py, docs/missing_v2.json. Raw
artifacts/missing_v2_mixture_validation.parquet has 5294 missing IDs. Selected
two-region blend AFTER COMPLETE V1 improves 266.052 -> 263.222, both months.
Rome alpha ~0.422412, other missing alpha 1; exact fields in report. Preserve
Rome schedule>=24000 tails from complete v1. Final refits were subsequently
implemented and verified; use the frozen reproduction steps below.

Known error opportunities: retained CatBoost borders miss AOBT tails <304.5
and >2218.5 and EOBT>3662.5. Test min(AOBTdelta,300), max(AOBTdelta,2200),
max(EOBTdelta,3600), then tune depth/rate/L2/border_count. Proposed depth6,
8000 iterations, rate .06, L2 20. More depth alone may not help. Existing
artifacts/tfm_queue_d10_v2.json/predictions may be complete; inspect first.
Known v1 RMSE226.8485. LIRF accounts for22.35% known SSE; EOBT>3600 (5224rows)
12.36%. Faulty AOBT near takeoff and LOBT-AOBT~-1to-2hours are small high-error
regimes. LEBL schedule identity is a possible separate expert.

TimesFM3 installed from .vendor/timesfm, official 330M checkpoint cached. See
scripts/timesfm_features.py and docs/timesfm_notes.md. Real inference already
done; no fine-tuning claim. January2025 has no TimesFM context, January2026
has complete context: distribution gap. EHAMJan26 has more missing NM, snow
and high delays; LTFM tails shift. July2026 TimesFM uses June2025 historical
context (no June2026 labels), with actual covariates. Do not introduce validation
labels into forecast context. Weights' noncommercial license is documented;
prize clearance unresolved, never claim permission has been obtained.

## Worker outputs

Each writes docs/research_<lane>.json with trials, selected params, scores by
month, training exclusions, paths, precise apply order, and next hypothesis.
Each selected recipe needs matching final/ranking inference and one meaningful
check. Parent independently reconstructs scores, checks leakage and invariants,
then grants follow-up based on the strongest remaining uncertainty.

## Frozen v2 reproduction

Start from the completed v1 artifacts documented in `docs/solution.md`. The
frozen validation recipe and metrics are in `docs/v2_ensemble.json`; reproduce
its cap-12000 missing expert and integrated validation predictions in this order:

```powershell
python scripts/check_v2.py --snapshot-v1
python scripts/missing_v2.py
python -u scripts/train.py --name tfm_queue_d10_v2 --iterations 4500 --depth 10 --rate .05 --l2 30 --border-count 128 --threads 8 --device GPU --residual --known-only --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --boost-priority
python -u scripts/train.py --name catboost_tail_d6_s600 --iterations 4500 --depth 6 --rate .06 --l2 20 --border-count 128 --threads 8 --device GPU --residual --known-only --tail-copies --sample 600000 --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --boost-priority
python -u scripts/train.py --name catboost_tail_d8_s600 --iterations 4500 --depth 8 --rate .06 --l2 20 --border-count 128 --threads 8 --device GPU --residual --known-only --tail-copies --sample 600000 --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --boost-priority
python scripts/moe_cap_trial.py
python scripts/moe_cap_decision.py
python scripts/timesfm_blend_research.py
```

After the v2 parameters are frozen, build ranking artifacts. The first command
fits the identity classifier used by the second command; the second refits the
selected cap-12000 normal expert and writes its raw missing-row predictions.

```powershell
python scripts/moe_finalize.py
python scripts/moe_finalize_cap12000.py
python -u scripts/train.py --name tfm_queue_d10_v2_final --iterations 4499 --depth 10 --rate .05 --l2 30 --border-count 128 --threads 8 --device GPU --residual --tfm data/processed/timesfm_hourly_final.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --known-only --final --boost-priority
python -u scripts/train.py --name catboost_tail_d6_final --iterations 4496 --depth 6 --rate .06 --l2 20 --border-count 128 --threads 8 --device GPU --residual --tfm data/processed/timesfm_hourly_final.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --known-only --tail-copies --final --boost-priority
```

Apply the frozen seasonal weights, missing expert, and gated CatBoost delta in
the exact saved model order. July ranking rows use the July weights; other
ranking months use the frozen winter weights.

```powershell
python scripts/timesfm_blend_research.py --apply-final --config docs/v2_ensemble.json --predictions artifacts/tfm_queue_final_predictions.parquet artifacts/no_tfm_final_predictions.parquet artifacts/lgbm_tfm_ranking.parquet artifacts/tfm_queue_d10_v2_final_predictions.parquet --missing artifacts/missing_ranking.parquet --rome artifacts/lirf_identity_expert_ranking.parquet --missing-v2 artifacts/moe_normal_cap12000_raw_missing_ranking.parquet --tail-predictions artifacts/catboost_tail_d6_final_predictions.parquet --output submissions/elegant-alligator_v2.parquet
python -O scripts/check_v2.py submissions/elegant-alligator_v2.parquet
```

The final apply step requires 344,841 unique template IDs and finite predictions.
It preserves the stored v1 prediction for missing Rome rows whose schedule delta
is at least 24,000 seconds.
