# September autoresearch reproduction

Start with the V3 feature caches, frozen predictions, and final models generated
by [solution_v3.md](solution_v3.md). The experiment loop follows Karpathy's
[autoresearch](https://github.com/karpathy/autoresearch): propose a bounded
change, fit, measure against the same complete baseline, retain or reject, and
record the result. This uses tabular forecasting experiments, not the upstream
LLM training program. Only official uploads consume the user's three attempts.

The parent owns uploads. Three GPT-5.6-sol workers handle residual models,
chronological baseline reconstruction, and independent ensemble checks. They
exchange compact reports and artifact paths. GPU jobs run sequentially on one
RTX 3060; CPU specialist jobs use three threads.

## Frozen candidate

```powershell
$env:OPENBLAS_NUM_THREADS='1'
python scripts/autoresearch_known_lobt.py --name known_lobt_d10_a2 --iterations 3500 --depth 10 --rate .05 --l2 20 --threads 6 --device GPU
python scripts/autoresearch_known_lobt.py --name known_lobt_d10_a2 --score
python scripts/autoresearch_lobt_regional.py
python scripts/autoresearch_identity.py
```

The known-timestamp expert learns `TAXITIME - (movement time - LOBT)`, then adds
that supplied estimate back. It uses the V3 timestamp, airport, weather,
TimesFM, and arrival-context features plus timestamp-range interactions.
Only rows with both AOBT and LOBT receive its correction. The global coefficient
is 0.630797588656986. Airport coefficients are in
[autoresearch_lobt_regional.json](autoresearch_lobt_regional.json), selected
trial 0, with shrinkage equivalent to 1,000 global calibration rows.

The optional winter specialist adds flight identity and destination categories
to the Rome schedule-versus-normal classifier. Its coefficient is
0.8788581367641393. It applies only to missing-AOBT Rome rows whose schedule
delta is below 24,000 seconds, in local November or ranking January. Summer
performance rejected the all-season version. The normal regressor remains the
V3 arrival-context normal model. V3's rare Rome tail remains unchanged.

Model development excludes July, November, and December labels. Blend fitting
uses days 1–14 of July/November; later dates check performance. These months
were researched repeatedly, so the scores and bootstrap intervals are
exploratory. No coefficient is fitted to official scores.

## Complete chronological checks

For each `MONTH` in `2025-07`, `2025-11`, run the specialist commands before the
full baseline. Replace `MONTH` in both flags and filenames.

```text
python scripts/lightgbm_model.py --rolling-month MONTH
python scripts/missing_specialist.py --rolling-month MONTH
python scripts/lirf_identity_experiment.py --rolling-month MONTH
python scripts/forward_missing_mixture.py --month MONTH
python scripts/forward_v3.py --month MONTH --fit-lightgbm --fit-bases --device GPU --threads 6
python scripts/autoresearch_known_lobt.py --name known_lobt_d10_forward_FOLD --rolling-month MONTH --iterations 3500 --depth 10 --rate .05 --l2 20 --threads 6 --device GPU
python scripts/autoresearch_known_lobt.py --name known_lobt_d10_forward_FOLD --score-forward --baseline artifacts/forward_v3_MONTH_complete.parquet
python scripts/autoresearch_lobt_regional.py --map-base artifacts/forward_v3_MONTH_complete.parquet --map-raw artifacts/known_lobt_d10_forward_FOLD_predictions.parquet --output artifacts/autoresearch_lobt_regional_forward_FOLD.parquet
python scripts/autoresearch_lobt_regional_forward_check.py MONTH
```

The explicit LightGBM command is optional if `--fit-lightgbm` is used. `FOLD`
is `july` or `november`. Complete V3 reconstruction has an independent numerical
check against retained original predictions:

```text
python scripts/check_forward_v3.py
python scripts/autoresearch_identity.py --rolling-month 2025-11
python scripts/autoresearch_identity_winter_check.py
python scripts/autoresearch_identity_winter_forward_complete.py
```

Each refitted model sees strictly earlier departure labels. LOBT additionally
retains the development exclusion of prior July. Existing TimesFM development
caches omit July/November/December target history in both baseline and candidate.
Frozen routing and coefficients were selected through earlier research on the
same months, so chronological training does not create an untouched test.
There is no genuine January forward backtest with only one labeled year.

## Final files and official measurement

After validation, fit the frozen recipes on all 2025 labels, including December,
without evaluating or retuning on December:

```text
python scripts/autoresearch_known_lobt.py --name known_lobt_d10_a2_final --final --iterations 3500 --depth 10 --rate .05 --l2 20 --threads 6 --device GPU
python scripts/autoresearch_identity.py --final
python scripts/assemble_autoresearch_attempt.py --version 5
python scripts/assemble_autoresearch_attempt.py --version 6
```

The assembler always starts from original V3. V5 uses the global LOBT
coefficient; V6 uses airport coefficients plus the disjoint winter correction.
It verifies schema, IDs, finite values, unchanged rows outside the gates, and
local reconstruction, and records hashes. Publish the source before uploading
with `scripts/submit.py FILE --upload`. Retrieve scores and team-best rank with
`python scripts/official_status.py`. Outcomes and attempt accounting belong in
[autoresearch_attempts.md](autoresearch_attempts.md).

GPU training may vary slightly with fixed seeds. Retained raw predictions and
their recorded hashes identify the exact submitted files; model weights and
restricted per-row data are not published.
