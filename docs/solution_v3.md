# Frozen v3 reproduction

Start with the completed [v2 artifacts](research_program.md#frozen-v2-reproduction).
V3 reaches 259.976902 seconds on July/November tuning: July 291.504838 and
November 217.166000. The preceding v2 scores 262.526361. These months have been
repeatedly used for tuning; they are not untouched test data. No new selection
uses December departure labels or official ranking labels.

The [frozen recipe](v3_ensemble.json) adds three corrections to complete v2:

1. A full-development CatBoost tail model, blended with coefficient 0.5441252013.
2. A missing-timestamp normal expert with arrival stand/runway context, retaining
   the existing identity classifier and Rome/other routing coefficients.
3. A TimesFM-supported arrival-context model on known, non-Rome departures. Its
   coefficient is 1, applied to the seasonally weighted and routed model delta.

Both known-row deltas are measured against the original v2 seasonal base.
Apply the CatBoost delta and bounds first, then the missing correction, then the
arrival-model delta and its bounds. Missing Rome schedule tails stay unchanged.
January ranking uses the frozen November seasonal weights; July uses July's.

## Development and selection

Arrival features use only supplied arrival observations. Airport-month stand
and runway taxi summaries, and the preceding arrival at the same stand, can be
constructed for ranking data. Departure targets and block timestamps are not
loaded by the feature builder. Its mutation check and independent code review
verify this separation.

Run these commands from the repository root, with `OPENBLAS_NUM_THREADS=1`.
Serialize GPU fits. CPU-only expert fits use three threads.

```powershell
python scripts/check_v3.py --snapshot-v2
python scripts/arrival_context.py --check
python scripts/arrival_context.py
python scripts/train.py --name catboost_tail_d8_full_v3 --iterations 6000 --depth 8 --rate 0.07505790178893094 --l2 4.090915852190167 --seed 42 --threads 8 --device GPU --residual --known-only --tail-copies --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --boost-priority
python scripts/catboost_score_v3.py
python scripts/moe_missing_arrival_v3.py
python scripts/moe_missing_arrival_assess.py
python scripts/combine_v3_validation.py
python scripts/train.py --name tfm_arrival_d8_v3 --iterations 6000 --depth 8 --rate 0.0750579 --l2 4.090916 --seed 42 --threads 8 --device GPU --residual --known-only --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_arrival_context_v3.parquet --interactions --boost-priority
python scripts/timesfm_arrival_v3.py
Copy-Item artifacts/timesfm_arrival_d8_v3_gated_validation.parquet artifacts/v3_validation.parquet
```

The retained development fits both contain 5,995 trees. Continuous correction
weights minimize squared error on even dates; odd-date and cross-month checks
must improve in both months. A two-variable bounded optimization was also
tested, but its November check deteriorated, so the original pair was retained.
Shorter TimesFM context and new known-Rome experts were tested and rejected.
Their reports remain in `docs/research_*v3.json`.

## Final fits

After freezing parameters, refit using 2025 labels from all months. The missing
expert retains the established known-row sample and all eligible missing rows.
The two global models use all known-AOBT training departures.

```powershell
python scripts/moe_missing_arrival_finalize_v3.py
python scripts/train.py --name catboost_tail_d8_full_v3_final --iterations 5995 --depth 8 --rate 0.07505790178893094 --l2 4.090915852190167 --seed 42 --threads 8 --device GPU --residual --known-only --tail-copies --tfm data/processed/timesfm_hourly_final.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --final --boost-priority
python scripts/train.py --name tfm_arrival_d8_v3_final --iterations 5995 --depth 8 --rate 0.0750579 --l2 4.090916 --seed 42 --threads 8 --device GPU --residual --known-only --tfm data/processed/timesfm_hourly_final.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_arrival_context_v3.parquet --interactions --final --boost-priority
```

GPU refits may differ slightly despite fixed seeds. The retained prediction
files and hashes reproduce the measured frozen result.

## Assemble and verify

```powershell
python scripts/catboost_apply_v3.py --baseline submissions/elegant-alligator_v2.parquet --predictions artifacts/tfm_queue_final_predictions.parquet artifacts/no_tfm_final_predictions.parquet artifacts/lgbm_tfm_ranking.parquet artifacts/tfm_queue_d10_v2_final_predictions.parquet --tail-predictions artifacts/catboost_tail_d8_full_v3_final_predictions.parquet --tail-config docs/v3_ensemble.json --missing-arrival artifacts/moe_missing_arrival_v3_raw_missing_ranking.parquet --output artifacts/v3_cat_missing_ranking.parquet
python scripts/timesfm_arrival_apply_v3.py --baseline artifacts/v3_cat_missing_ranking.parquet --arrival-predictions artifacts/tfm_arrival_d8_v3_final_predictions.parquet --config docs/v3_ensemble.json --output submissions/elegant-alligator_v3.parquet
python -O scripts/check_v3.py submissions/elegant-alligator_v3.parquet --config docs/v3_ensemble.json --missing-arrival artifacts/moe_missing_arrival_v3_raw_missing_ranking.parquet --arrival-predictions artifacts/tfm_arrival_d8_v3_final_predictions.parquet --arrival-config docs/v3_ensemble.json
```

The independent checker reconstructs the arithmetic separately, verifies all
344,841 template IDs, preserves 25 protected Rome tails, and verifies the stored
v1/v2 hashes. These commands do not upload. Official results and submission/model
hashes are recorded in [results_v3.json](results_v3.json).
