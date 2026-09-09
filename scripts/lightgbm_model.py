"""CPU residual model: development experiment, frozen final refit, or explicit audit."""
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
ID, TIME, TARGET = "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"


def selected_column(parquet, name, indices):
    # Only one compressed column is decoded at a time; never load full features.
    return parquet.read(columns=[name], use_threads=False).column(0).take(indices).to_pandas()


def residual_base(x):
    base = x["mvt_minus_AOBT_3_flt"].to_numpy(dtype=np.float64).copy()
    missing = base < -100000
    base[(base < 0) | (base > 172800)] = 900
    use_schedule = missing & x.ADEP_mvt.eq("LIRF").to_numpy()
    base[use_schedule] = x.loc[use_schedule, "mvt_minus_SCHED_TIME_UTC_mvt"].to_numpy()
    base[(base < 0) | (base > 172800)] = 900
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfm", type=Path)
    parser.add_argument("--sample", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="Refit frozen settings on an all-2025 sample and predict ranking")
    mode.add_argument("--audit", action="store_true", help="Predict December with the original development model; scoring belongs to the frozen ensemble")
    args = parser.parse_args()
    if not 1 <= args.sample <= 500000:
        parser.error("--sample must be between 1 and 500000")
    pa.set_cpu_count(4)
    pa.set_io_thread_count(2)
    started = time.perf_counter()
    args.tfm = args.tfm or ROOT / "data/processed" / (
        "timesfm_hourly_final.parquet" if args.final else "timesfm_hourly_dev.parquet")
    frozen = None
    original = None
    if args.final or args.audit:
        frozen = json.loads((OUT / "lgbm_tfm.json").read_text(encoding="utf-8"))
        original = lgb.Booster(model_file=str(OUT / "lgbm_tfm.txt"))
        args.sample, args.seed = frozen["training_rows"], frozen["sample_seed"]
    rows_path = OUT / "rows.parquet"
    rows = pd.read_parquet(rows_path, columns=[ID, TIME, "ADEP_mvt", "source"])
    month = rows[TIME].dt.month
    known = rows.source.eq("training") & rows[TIME].dt.year.eq(2025)
    eligible = np.flatnonzero(known if args.final else known & ~month.isin([7, 11, 12]))
    training = np.empty(0, dtype=np.int64) if args.audit else np.random.default_rng(args.seed).choice(
        eligible, min(args.sample, len(eligible)), replace=False)
    validation = np.flatnonzero(rows.source.eq("ranking") if args.final else known & (
        month.eq(12) if args.audit else month.isin([7, 11])))
    selected = np.sort(np.r_[training, validation])
    train = np.isin(selected, training)
    rows = rows.iloc[selected].reset_index(drop=True)
    rows[TARGET] = np.nan
    labels = train if args.final else np.ones(len(rows), dtype=bool)
    rows.loc[labels, TARGET] = selected_column(pq.ParquetFile(rows_path), TARGET, selected[labels]).to_numpy()
    if not np.isfinite(rows.loc[labels, TARGET]).all() or not rows[ID].is_unique:
        raise ValueError("Selected rows must have finite targets and unique IDs")
    if not args.final and not args.audit:
        assert not rows[TIME].dt.month.eq(12).any()
    prediction_scope = "ranking" if args.final else "December audit" if args.audit else "July/November validation"
    print(f"Selected {train.sum()} training rows and {(~train).sum()} {prediction_scope} rows", flush=True)
    parquet = pq.ParquetFile(OUT / "features.parquet")
    if parquet.metadata.num_rows != pq.ParquetFile(rows_path).metadata.num_rows:
        raise ValueError("Feature and row caches differ in length")
    forbidden = {ID, TARGET, "BLOCK_TIME_UTC_mvt", "FLIGHT_ID_mvt"}
    if forbidden.intersection(parquet.schema_arrow.names):
        raise ValueError("IDs or hidden fields found in feature cache")
    x = pd.DataFrame(index=rows.index)
    names = parquet.schema_arrow.names if original is None else [name for name in original.feature_name() if name in parquet.schema_arrow.names]
    for number, name in enumerate(names, 1):
        column = selected_column(parquet, name, selected)
        x[name] = column.fillna("__NA__").astype("category") if column.dtype == object else column.astype("float32")
        del column
        if number % 20 == 0:
            print(f"Loaded {number} feature columns", flush=True)
    del parquet, selected, eligible, training, validation, month, known
    tfm_config = json.loads(args.tfm.with_suffix(".json").read_text(encoding="utf-8"))
    expected_mode, excluded = ("final", set()) if args.final else ("dev", {7, 11, 12})
    if tfm_config.get("mode") != expected_mode or set(tfm_config.get("globally_excluded_departure_target_months", [])) != excluded:
        raise ValueError("TimesFM feature provenance does not match the selected development/final mode")
    tfm = pd.read_parquet(args.tfm)
    keys = ["ADEP_mvt", "hour"]
    if tfm.duplicated(keys).any() or any(not c.startswith("tfm_") for c in tfm if c not in keys):
        raise ValueError("Invalid TimesFM feature schema or duplicate airport-hour keys")
    hours = pd.MultiIndex.from_arrays([rows.ADEP_mvt, rows[TIME].dt.floor("h")], names=keys)
    tfm = tfm.set_index(keys).reindex(hours).reset_index(drop=True)
    if tfm.loc[~train, "tfm_median"].isna().any():
        raise ValueError("TimesFM features are missing prediction airport-hours")
    for name in tfm:
        x[name] = tfm[name].fillna(-999999).to_numpy(dtype=np.float32)
    del tfm, hours
    if original is not None:
        if set(original.feature_name()).difference(x.columns):
            raise ValueError("Frozen model features are missing")
        x = x.loc[:, original.feature_name()]
    gc.collect()
    target = rows[TARGET].to_numpy(dtype=np.float64)
    base = residual_base(x)
    categories = x.select_dtypes("category").columns.tolist()
    params = {"objective": "regression", "metric": "rmse", "learning_rate": .05,
        "num_leaves": 63, "lambda_l2": 20, "num_threads": 4, "device_type": "cpu",
        "max_bin": 127, "force_col_wise": True, "seed": args.seed, "verbosity": -1,
        "deterministic": True}
    if frozen is not None:
        params = dict(frozen["parameters"], num_threads=4, device_type="cpu")
    if not args.audit:
        data = lgb.Dataset(x.loc[train], label=target[train] - base[train],
            categorical_feature=categories, params=params, free_raw_data=True).construct()
    valid_x = x if args.audit else x.loc[~train]
    del x
    gc.collect()
    print(f"Prepared {len(valid_x.columns)} features in {time.perf_counter()-started:.1f}s", flush=True)
    if args.audit:
        model, iterations = original, frozen["best_iteration"]
    elif args.final:
        def progress(env):
            if (env.iteration + 1) % 50 == 0:
                print(f"Final CPU refit: {env.iteration + 1}/{frozen['best_iteration']} trees", flush=True)
        iterations = frozen["best_iteration"]
        model = lgb.train(params, data, num_boost_round=iterations, callbacks=[progress])
    else:
        valid_data = lgb.Dataset(valid_x, label=target[~train] - base[~train], reference=data,
            categorical_feature=categories, params=params, free_raw_data=True).construct()
        model = lgb.train(params, data, num_boost_round=1000, valid_sets=[valid_data],
            valid_names=["july_november"], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
        iterations = model.best_iteration
    prediction = model.predict(valid_x, num_iteration=iterations, num_threads=4) + base[~train]
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced nonfinite predictions")
    result = rows.loc[~train, [ID, TIME, "ADEP_mvt", TARGET]].copy()
    result["prediction"] = prediction
    suffix = "ranking" if args.final else "audit" if args.audit else "validation"
    result.to_parquet(OUT / f"lgbm_tfm_{suffix}.parquet", index=False)
    if not args.audit:
        model.save_model(str(OUT / ("lgbm_tfm_final.txt" if args.final else "lgbm_tfm.txt")), num_iteration=iterations)
    result["month"] = result[TIME].dt.month
    rmse = lambda frame: float(np.sqrt(np.mean((frame[TARGET] - frame.prediction) ** 2)))
    report = {"lightgbm_version": lgb.__version__, "mode": suffix, "training_rows": int(train.sum()),
        "prediction_rows": len(result), "parameters": params, "sample_seed": args.seed,
        "timesfm_file": str(args.tfm), "best_iteration": iterations,
        "seconds": time.perf_counter() - started,
        "feature_importance": dict(sorted(zip(model.feature_name(), model.feature_importance("gain").tolist()), key=lambda pair: -pair[1]))}
    if not args.final and not args.audit:
        report.update(rmse=rmse(result), month_rmse={str(m): rmse(g) for m, g in result.groupby("month")},
            airport_rmse={str(a): rmse(g) for a, g in result.groupby("ADEP_mvt")})
    report_path = OUT / ("lgbm_tfm_final.json" if args.final else "lgbm_tfm_audit.json" if args.audit else "lgbm_tfm.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "feature_importance"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
