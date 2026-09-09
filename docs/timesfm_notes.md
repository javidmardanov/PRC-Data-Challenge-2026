# TimesFM-3 research features

`python scripts/timesfm_features.py` generates actual Google TimesFM-3 forecasts,
using checkpoint `google/timesfm-3.0-pytorch` revision
`43046b85ec22d584a13f8098c2ed39c889e129c2`. Source code is the official repository
at tag `v3.0.0`; the environment uses its editable install and PyTorch CUDA.

The target is mean departure taxi-out seconds at each airport and UTC hour.
Each forecast uses 512 historical hours and forecasts the entire next month.
Development months are February–December 2025. Departure labels in July,
November, and December 2025 are globally excluded from development inputs,
including input histories of training months. No target from the forecast month
enters its model inputs. A runnable mutation check confirms that replacing all
targets from July 2025 onward cannot affect the July 2025 model inputs, and
changing all globally excluded labels cannot affect August/November/December inputs.
For histories blocked by an excluded month, the most recent contiguous permitted
context is retrieved: August uses June, November uses October, December uses October.
These gaps are treated as explicit context retrieval, not fabricated target values.
January 2025 has no earlier supplied history and receives no TimesFM feature.

Eight known covariates span context plus horizon: the mean departure proxy
`MVT_TIME_UTC_mvt - AOBT_3_flt`, clipped to 0–14400 seconds; arrival taxi mean;
departure and arrival counts; hour-of-day and day-of-week sine/cosine pairs.
Arrival taxi times and movement metadata are supplied in the ranking dataset.
Only training departures provide historical taxi-out labels. Missing mean values
are passed as NaNs to the official forecaster; counts are zero only within months
for which airport movements are supplied. Future departure targets are excluded.

Final features are generated separately with `python scripts/timesfm_features.py --mode final`;
all 2025 historical labels are permitted only in that final refit mode.
January 2026 uses historical labels through December 2025. July 2026 deliberately
uses the last 512 hours of June 2025 as a seasonal retrieval context, followed by
the supplied July 2026 covariates. This is a discontinuous seasonal retrieval
strategy, not a claim that June 2026 labels exist. Context calendar covariates
retain their actual 2025 dates and future covariates retain their 2026 dates.
No synthetic labels fill the February–June 2026 gap.

Outputs `data/processed/timesfm_hourly_dev.parquet` and
`data/processed/timesfm_hourly_final.parquet` join to departures on
`ADEP_mvt` and `MVT_TIME_UTC_mvt.dt.floor('h')` matching output `hour`.
It contains nine deciles `tfm_q1`–`tfm_q9`, `tfm_median`, `tfm_decile_mean`,
and `tfm_width`. The decile mean is a truncated approximation to an expectation;
the model's official point prediction is its median, which need not minimize RMSE.
No departure labels are stored in the feature output. The separate internal
hourly input cache includes training-only historical targets.

Batch size is one, symmetric averaging is disabled, and positivity and quantile
sorting are enabled. There are nine total channels, below the evaluator's
32-channel automatic subsampling threshold. Monthly airport outputs are cached
independently so interrupted runs resume. Metadata records timing and memory.
Final mode reuses development predictions only when historical target arrays and
all covariate arrays are exactly equal; changed histories trigger real inference.
Use a different output filename when changing methodology; `--refresh` regenerates
all requested cached forecasts.

Executed on an NVIDIA GeForce RTX 3060: 110 development forecasts required
46.63 seconds of model-call time, and 40 additional final forecasts required
18.32 seconds. Peak allocated CUDA memory was 1.353 GB. Loading, hourly data
preparation, input comparisons, and file writes take additional wall time.
The development output has 80,160 rows and the final output has 95,040 rows.
Every one of the 344,841 ranking departures joins to finite TimesFM features.
Only feature validity and coverage were checked here; no December target score
was computed during this integration.

These are research features. The pretrained weights use the TimesFM
Non-Commercial License v1.0, with non-commercial/non-production restrictions and
no redistribution of weights or derivatives. Prize-competition eligibility is
not explicitly established by that license. The experiment does not redistribute
the checkpoint or imply clearance to publish a prize submission using it.

Sources: [Google announcement](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/),
[official implementation](https://github.com/google-research/timesfm/tree/v3.0.0),
[model](https://huggingface.co/google/timesfm-3.0-pytorch),
[weights license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE).
