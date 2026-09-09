# PRC Data Challenge 2026 — TimesFM-3 solution

Taxi-out-time prediction for EUROCONTROL's PRC 2026 challenge. Frozen v1 combines
Google TimesFM-3 airport forecasts with CatBoost and LightGBM residual models,
a missing-timestamp specialist, and a Rome airport expert.

| Evaluation | RMSE, seconds |
| --- | ---: |
| July + November 2025 tuning | 266.052234 |
| July 2025 | 297.130401 |
| November 2025 | 224.100558 |
| December 2025 frozen audit | 217.504038 |
| Official January + July 2026, v1 | 285.0706 |

The tuning set contains 353,045 departures. December contains 165,677 departures
and was scored once after freezing the recipe, without further selection.
The official scorer accepted all 344,841 v1 predictions. The public leader was
246.7053 seconds at the recorded snapshot; v1 is not a winning result.
Exact results and frozen settings are in [results.json](docs/results.json)
and [v1_ensemble.json](docs/v1_ensemble.json).

Follow the [complete reproduction guide](docs/solution.md) for ordered commands
to download inputs, generate features, fit every model, audit the frozen recipe,
and create the submission. It uses Python 3.12, CUDA-enabled PyTorch, and the
pinned dependencies in [requirements.txt](requirements.txt).

```powershell
python -m pip install -r requirements.txt
python scripts/fetch_data.py --bucket prc-2026-datasets --download --manifest data/manifest.json
```

The downloader reads locally supplied `credentials.json` without printing its
secrets. Raw inputs, credentials, model weights, and prediction caches are
gitignored; never publish credentials or restricted datasets. GPU refits can
differ slightly despite fixed seeds; retained prediction caches reproduce the
frozen blend more exactly.

See the [challenge brief](docs/challenge_brief.md), [data dictionary](docs/data_dictionary.md),
[TimesFM checkpoint notes](docs/timesfm_notes.md), and [external weather provenance](docs/weather_notes.md).
Produced source is GPLv3. Google's source and weights have separate licenses;
TimesFM-3 weights restrict commercial/production use and redistribution.
Prize-competition clearance remains unresolved. The required GPLv3 source is
published at [GitHub](https://github.com/javidmardanov/PRC-Data-Challenge-2026).
`elegant-alligator_v1.parquet` was uploaded on 2026-09-09 at 02:54 UTC and scored
at 02:55 UTC. Further uploads count toward the five-per-day limit.
