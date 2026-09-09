# Frozen v1 reproduction guide

Run all commands from the repository root. The canonical blend is
[v1_ensemble.json](v1_ensemble.json); exact measured results are in
[results.json](results.json). Source is GPLv3. These commands create local
artifacts and a submission file; they do not upload anything.

## Method and evaluation

Development models train on January–June and August–October 2025. July and
November are the tuning months for early stopping, parameter selection, expert
selection, and blending. December was excluded from those decisions and scored
once after freezing v1. RMSE is measured in seconds:

| Set | Departures | RMSE |
| --- | ---: | ---: |
| July + November tuning | 353,045 | 266.0522342012221 |
| July tuning | — | 297.1304005768509 |
| November tuning | — | 224.10055777655305 |
| December frozen audit | 165,677 | 217.5040384241259 |

No selection used the December audit score. These are local measurements, not
official ranking scores. The tuning split tests missing months: some training
observations follow July. `train.py --forward` is an optional January–June-to-July
diagnostic, separate from v1. Final ranking models use all official 2025 labels;
the audit uses retained development models.

The residual baseline is takeoff minus the supplied actual off-block proxy,
with a 900-second fallback and a schedule-based fallback for missing Rome
records. Trees learn corrections from flight/aircraft metadata, timestamp
differences, calendar fields, traffic counts, runway gaps, and arrival taxi
statistics. Movement and flight IDs, departure block timestamps, and departure
targets are excluded from model features. This is reconstruction from supplied
movement records, not a live pre-departure forecast. Centered traffic windows
and arrival observations are available in ranking inputs. Timestamp units are
explicitly normalized before window calculations.

Actual Google TimesFM-3 forecasts supply nine deciles, a median, a decile average,
and an interval width for airport-hour taxi-out means. Context is 512 hours and
the horizon is a whole month. Development histories globally exclude July,
November, and December departure labels, including histories for training rows.
Final features are generated separately. July 2026 retrieves June 2025 history;
no unavailable 2026 labels are invented. The checkpoint is
`google/timesfm-3.0-pytorch` revision
`43046b85ec22d584a13f8098c2ed39c889e129c2`; see [TimesFM notes](timesfm_notes.md).
Weather uses public-domain IEM METAR observations joined at or before each
airport-hour; see [weather provenance](weather_notes.md).

The frozen model order and weights are:

| Development artifact | Final artifact | Weight | Settings |
| --- | --- | ---: | --- |
| `tfm_queue_d8` | `tfm_queue_final` | 0.7985604681600207 | CatBoost: 6000 trees, depth 8, rate 0.0750579, L2 4.090916; base + TimesFM + weather + queue + interactions |
| `no_tfm_full_d8` | `no_tfm_final` | 0.17672281746128118 | CatBoost: 5000 trees, depth 8, rate 0.06, L2 10; base features |
| `lgbm_tfm` | `lgbm_tfm_ranking` | 0.024716714378698216 | LightGBM: 188 trees, 63 leaves, rate 0.05, L2 20; base + TimesFM; 500,000 sampled rows |

Both main CatBoost models fit only records with the off-block proxy. Their raw
missing-row predictions must pass through the specialist. LightGBM fits a
uniform sample with all record types and four CPU threads. Tree parameters were
explored with seeded Optuna TPE; that search is unnecessary to reproduce frozen
settings. SLSQP jointly minimizes tuning MSE over simplex model weights and a
bounded missing-specialist coefficient, using analytic gradients and three
starts. These methods handle discrete trees and continuous blends directly.

After the global weighted sum, the missing specialist replaces missing-proxy
rows (alpha = 1 to numerical precision). A partial LIRF expert then blends only
records with takeoff minus schedule below 24,000 seconds. Its least-squares beta
is 0.16359683681421205 for known-proxy rows and 0.9048126597030896 for missing
rows. Finally, apply validation-selected nonnegative clipping and timing bounds
`(MVT − LOBT) ± 3606` seconds where LOBT is present. Bounds use ranking-visible
fields and contain no target column; development verification is recorded in
[queue_features.json](queue_features.json). There is no arbitrary upper cap on
legitimate long records.

## 1. Environment, inputs, and features

The executed environment used Python 3.12, PyTorch 2.7.0+cu118, CatBoost 1.2.10,
and LightGBM 4.7.0. Install a CUDA-enabled PyTorch build compatible with your
machine; requirements alone do not establish GPU availability. TimesFM source
is pinned to a commit in `requirements.txt`. Supply official challenge
credentials locally as `credentials.json` and keep them out of Git.

```powershell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/fetch_data.py --bucket prc-2026-datasets --download --manifest data/manifest.json
python scripts/train.py --prepare
python scripts/check_solution.py
python scripts/weather_features.py
python scripts/queue_features.py
python scripts/timesfm_features.py --mode dev
python scripts/timesfm_features.py --mode final
python scripts/ensemble.py --self-check
```

Inputs are 12 training Parquets, `ranking.parquet`, and `submitting.parquet`
under `data/raw/prc-2026-datasets/`. The downloader records SHA-256 hashes.
Preparation writes `artifacts/features.parquet` and `artifacts/rows.parquet`.
The other generators write `weather_hourly.parquet`, `queue_features.parquet`,
`timing_bounds.parquet`, `timesfm_hourly_dev.parquet`, and
`timesfm_hourly_final.parquet` under `data/processed/`, with provenance reports.
Weather and queue generation take no command-line options.

Run GPU commands sequentially to avoid memory contention. `--boost-priority`
can be appended to `train.py` on Windows to raise only its own process priority.
Forecasts are cached by airport/month; use TimesFM's `--refresh` if inputs or
forecasting settings change. Keep downloaded weather snapshots and manifests
for reproducible source data. GPU fitting and inference can have small
floating-point differences even with fixed seeds; saved prediction caches give
the most exact blend reproduction.

## 2. Specialists and their baseline dependency

The initial residual model is needed by the Rome experiment, although it is
absent from the final blend. The missing-specialist script retains its development
model and also fits its selected configuration on all 2025 labels for ranking.
It does not score December.

```powershell
python -u scripts/train.py --name residual_d8 --residual --iterations 2200 --depth 8 --rate 0.06 --l2 10
python -u scripts/missing_specialist.py
python scripts/ensemble.py --validation artifacts/residual_d8_predictions.parquet --missing-validation artifacts/missing_validation.parquet --config artifacts/baseline_mix.json
python -u scripts/lirf_identity_experiment.py
```

This creates `missing_validation.parquet`, `missing_ranking.parquet`, original
missing audit weights/config, `baseline_mix.json`, and the partial
`lirf_identity_expert_validation.parquet`. The Rome experiment retains its
development classifier and normal-regression model. Its schedule-weighted
classifier models the probability of a schedule-identity record; the expert
mixes schedule duration with the ordinary-record prediction. Selected settings
and feature lists are saved in `docs/lirf_identity_experiment.json`.

## 3. Main development predictions and frozen blend

```powershell
python -u scripts/train.py --name tfm_queue_d8 --residual --known-only --iterations 6000 --depth 8 --rate 0.0750579 --l2 4.090916 --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions
python -u scripts/train.py --name no_tfm_full_d8 --residual --known-only --iterations 5000 --depth 8 --rate 0.06 --l2 10
python -u scripts/lightgbm_model.py --tfm data/processed/timesfm_hourly_dev.parquet
```

CatBoost saves `NAME.cbm`, `NAME.json`, and `NAME_predictions.parquet` under
`artifacts/`. LightGBM saves `lgbm_tfm.txt`, `lgbm_tfm.json`, and
`lgbm_tfm_validation.parquet`; its 1000-tree cap and 100-round early stopping
selected 188 trees. Recompute tuning weights, expert betas, and clipping with
the following command. A separate comparison config preserves committed v1.

```powershell
python scripts/ensemble.py --validation artifacts/tfm_queue_d8_predictions.parquet artifacts/no_tfm_full_d8_predictions.parquet artifacts/lgbm_tfm_validation.parquet --missing-validation artifacts/missing_validation.parquet --rome-validation artifacts/lirf_identity_expert_validation.parquet --bounds data/processed/timing_bounds.parquet --config artifacts/recomputed_v1_ensemble.json
```

The fitter requires all July/November IDs and aligns by ID, rejecting duplicates,
missing IDs, and nonfinite predictions. Specialist files are verified subsets.
Keep the three main files in this order for every frozen apply. Regenerated
predictions can slightly change fitted weights; `docs/v1_ensemble.json` remains
the recorded v1 recipe.

## 4. December audit with retained development models

These commands reproduce the once-completed frozen audit. Prediction generators
do not refit or score; only the last command explicitly evaluates December.
Do not select new parameters from this score and still call it an untouched
audit. Use development TimesFM features and original development weights.

```powershell
python scripts/train.py --name tfm_queue_audit --load artifacts/tfm_queue_d8.cbm --audit --residual --known-only --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions
python scripts/train.py --name no_tfm_audit --load artifacts/no_tfm_full_d8.cbm --audit --residual --known-only
python scripts/lightgbm_model.py --audit --tfm data/processed/timesfm_hourly_dev.parquet
python scripts/missing_specialist.py --predict-audit-december
python scripts/lirf_identity_finalize.py --audit
python scripts/ensemble.py --apply-config docs/v1_ensemble.json --predictions artifacts/tfm_queue_audit_predictions.parquet artifacts/no_tfm_audit_predictions.parquet artifacts/lgbm_tfm_audit.parquet --missing artifacts/missing_december_audit.parquet --rome artifacts/lirf_identity_expert_audit.parquet --audit --output artifacts/v1_december_audit.parquet
```

## 5. Final refits and local submission

Final fitting uses all official 2025 labels with selected parameters. LightGBM
reads its original development report to recover 188 trees, sample size, seed,
and parameters. Missing-specialist final predictions were created in step 2.
Final weights use distinct names so original audit models remain available.

```powershell
python -u scripts/train.py --name tfm_queue_final --final --residual --known-only --iterations 6000 --depth 8 --rate 0.0750579 --l2 4.090916 --tfm data/processed/timesfm_hourly_final.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions
python -u scripts/train.py --name no_tfm_final --final --residual --known-only --iterations 5000 --depth 8 --rate 0.06 --l2 10
python -u scripts/lightgbm_model.py --final --tfm data/processed/timesfm_hourly_final.parquet
python -u scripts/lirf_identity_finalize.py --final
python scripts/ensemble.py --apply-config docs/v1_ensemble.json --predictions artifacts/tfm_queue_final_predictions.parquet artifacts/no_tfm_final_predictions.parquet artifacts/lgbm_tfm_ranking.parquet --missing artifacts/missing_ranking.parquet --rome artifacts/lirf_identity_expert_ranking.parquet --output submissions/elegant-alligator_v1.parquet
python scripts/check_solution.py submissions/elegant-alligator_v1.parquet
```

The writer uses the original template, preserves columns, movement IDs, and row
order, and writes float64 predictions for all 344,841 departures. The checker
verifies these properties and finite values. Bounds use the path in the frozen
config; `--bounds PATH` can override its location. Generation does not contact
the submission bucket. Official uploads are limited to five per day; consult
the [challenge brief](challenge_brief.md).

## Rights and publication status

Never publish `credentials.json`, secret values, restricted raw data, or model
weights. Model reports and prediction caches are local artifacts excluded from
Git; the published deliverables are produced source and the documented recipe.
Google source and pretrained weights retain separate upstream licenses. The
[TimesFM-3 weight license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)
restricts commercial/production use and redistribution; cash-prize competition
clearance has not been established. This guide does not assert prize eligibility.
Official rules also require public GitHub source and documented external-data
rights. Public-source publication was pending when this guide was written.
