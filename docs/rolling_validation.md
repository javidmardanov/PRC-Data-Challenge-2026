# Forward validation and ranking mix audit

Run from the repository root:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python scripts/check_rolling.py
python scripts/rolling_backtest.py
python scripts/validation_shift.py
```

The backtest refits a matched pair of CatBoost residual models for July, October,
and November 2025. Each fit uses only earlier departure labels. Both models use
the existing TimesFM development forecasts, weather, queues, identical fixed
parameters, and 2,000 trees. The second model adds arrival stand/runway context.
Validation labels never choose the stopping iteration. The existing TimesFM
forecasts use earlier target history; arrival and traffic features use information
supplied for the forecast month, consistent with this post-operations task.

The previous completed fold calibrates one blend coefficient by bounded least
squares; the next whole month evaluates it. July uses the existing preference for
the arrival model (coefficient 1); October calibrates on July; November calibrates
on October. No departure labels from the evaluated month calibrate its coefficient.
Every fold checks all departures, including missing timestamps. The
report also scores the two standalone models, so blending can be judged against
simply retaining the arrival model. Run `python scripts/rolling_backtest.py
--score-only` to recompute the report without fitting. Outputs use separate names
and do not replace frozen V3 models or submission files.

These are diagnostic models, not a refit of V3's full specialist ensemble. Their
absolute scores must not be compared directly with V3's 259.98 tuning score or
271.0449 official score. July and November were previously researched, and the
configuration is informed by that research; chronological refitting does not make
them untouched research holdouts. October adds a different evaluation month.
Only one labeled year is available, so no genuine January forward fold exists.
November is an imperfect winter proxy. December remains reserved from this work.

The separate ranking mix audit weights local squared errors by the public ranking
proportions of season, airport, and missing AOBT. It maps ranking January to local
November and ranking July to local July. This moves V3 from 259.9769 to 264.4347
and V2 from 262.5264 to 266.9077. V3's improvement survives this adjustment, but
the estimate still undershoots the official V3 result. This diagnostic cannot
identify changes within those groups or remove optimism from repeated tuning.

Machine-readable results: `rolling_backtest.json` and `validation_shift.json`.

## Results

| Forward month | Baseline | Arrival model | Earlier-fold calibrated blend |
| --- | ---: | ---: | ---: |
| July | 325.1584 | 315.1311 | 315.1311 |
| October | 231.1416 | 229.5556 | 229.4899 |
| November | 236.3788 | 233.5565 | 233.1057 |
| Pooled | 269.5916 | 264.1122 | 263.9724 |

The blend improves the matched baseline by 5.6191 seconds and the arrival model
alone by 0.1397 seconds. All three folds improve versus the baseline. This supports
arrival context and modest seasonal shrinkage, but does not establish a gain over
the full V3 ensemble. No new submission was produced or uploaded. V3 remains the
champion; the verified official score is 271.0449. The 263.9724 above is a different
evaluation and must not be represented as an official score or a new V3 score.

The date-boundary/ranking-exclusion regression check passes, and the scoring
command verifies training cutoffs, ID alignment, matching labels, and absence of
validation-based early stopping for all six fitted models.
