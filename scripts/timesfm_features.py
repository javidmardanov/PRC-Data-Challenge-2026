"""Generate real TimesFM-3 hourly forecasts with whole-month target exclusion."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/prc-2026-datasets"
PROCESSED = ROOT / "data/processed"
CHECKPOINT = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
COVARIATES = ["proxy", "arrival_taxi", "departures", "arrivals",
              "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


def aggregate():
    """Departure targets are read from training files only, never ranking."""
    parts = []
    for path in [*sorted(RAW.glob("training*.parquet")), RAW / "ranking.parquet"]:
        frame = pd.read_parquet(path, columns=["ADEP_mvt", "ADES_mvt", "PHASE_mvt",
            "MVT_TIME_UTC_mvt", "AOBT_3_flt", "TAXITIME_SEC_mvt"])
        frame["hour"] = frame.MVT_TIME_UTC_mvt.dt.floor("h")
        dep = frame[frame.PHASE_mvt.eq("DEP")].copy()
        arr = frame[frame.PHASE_mvt.eq("ARR")].copy()
        dep["proxy"] = (dep.MVT_TIME_UTC_mvt - dep.AOBT_3_flt).dt.total_seconds().clip(0, 14400)
        if path.name == "ranking.parquet":
            assert dep.TAXITIME_SEC_mvt.isna().all(), "Ranking departure labels must be absent"
            dep["TAXITIME_SEC_mvt"] = np.nan
        departures = dep.groupby(["ADEP_mvt", "hour"]).agg(
            target_sum=("TAXITIME_SEC_mvt", "sum"), target_count=("TAXITIME_SEC_mvt", "count"),
            proxy_sum=("proxy", "sum"), proxy_count=("proxy", "count"),
            departures=("PHASE_mvt", "size"))
        arrivals = arr.groupby(["ADES_mvt", "hour"]).agg(
            arrival_taxi_sum=("TAXITIME_SEC_mvt", "sum"),
            arrival_taxi_count=("TAXITIME_SEC_mvt", "count"), arrivals=("PHASE_mvt", "size"))
        arrivals.index.names = ["ADEP_mvt", "hour"]
        parts.append(departures.join(arrivals, how="outer").reset_index())
    # Some source files overlap an hour at month boundaries. Combine sufficient
    # statistics so those hours have movement-weighted means, not mean-of-means.
    hourly = pd.concat(parts, ignore_index=True).groupby(["ADEP_mvt", "hour"]).sum().reset_index()
    for name in ["target", "proxy", "arrival_taxi"]:
        hourly[name] = hourly.pop(name + "_sum") / hourly.pop(name + "_count").replace(0, np.nan)
    assert not hourly.duplicated(["ADEP_mvt", "hour"]).any()
    # Count zero means an observed airport-hour without that movement phase.
    # Calendar gaps remain absent; they must never be interpreted as zero traffic.
    hourly[["departures", "arrivals"]] = hourly[["departures", "arrivals"]].fillna(0)
    return hourly.set_index(["ADEP_mvt", "hour"])


def window(hourly, airport, times):
    frame = hourly.loc[airport].reindex(times).copy()
    present_months = set(hourly.loc[airport].index.strftime("%Y-%m"))
    observed = times.strftime("%Y-%m").isin(present_months)
    frame.loc[observed, ["departures", "arrivals"]] = frame.loc[
        observed, ["departures", "arrivals"]].fillna(0)
    frame["hour_sin"] = np.sin(times.hour * (2 * np.pi / 24))
    frame["hour_cos"] = np.cos(times.hour * (2 * np.pi / 24))
    frame["dow_sin"] = np.sin(times.dayofweek * (2 * np.pi / 7))
    frame["dow_cos"] = np.cos(times.dayofweek * (2 * np.pi / 7))
    return frame


def inputs(hourly, airport, month, context, excluded_months=(7, 11, 12)):
    start = pd.Timestamp(month + "-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    future_times = pd.date_range(start, end, inclusive="left", freq="h")
    # Explicit seasonal retrieval: June 2025 history guides July 2026 using
    # July 2026 supplied covariates. No invented February-June 2026 targets.
    history_end = pd.Timestamp("2025-07-01", tz="UTC") if month == "2026-07" else start
    while (history_end - pd.Timedelta(hours=1)).month in excluded_months:
        history_end = (history_end - pd.Timedelta(hours=1)).normalize().replace(day=1)
    history_times = pd.date_range(end=history_end, periods=context + 1, freq="h")[:-1]
    history = window(hourly, airport, history_times)
    future = window(hourly, airport, future_times)
    assert history_times.max() < start
    assert history_times.max() < pd.Timestamp("2026-01-01", tz="UTC")
    assert not history_times.month.isin(excluded_months).any(), "History crosses a globally held-out month"
    target = history.target.to_numpy(dtype=np.float32)
    covariates = pd.concat([history[COVARIATES], future[COVARIATES]]).to_numpy(dtype=np.float32).T
    assert np.isfinite(target).any(), (airport, month, "No historical targets")
    assert covariates.shape == (len(COVARIATES), len(target) + len(future_times))
    return target, covariates, future_times, history_times


def self_check(hourly):
    """Neither future nor globally excluded labels may change model inputs."""
    airport = hourly.index.get_level_values(0)[0]
    original = inputs(hourly, airport, "2025-07", 512)
    changed = hourly.copy()
    changed.loc[changed.index.get_level_values(1) >= pd.Timestamp("2025-07-01", tz="UTC"), "target"] = -999999
    mutated = inputs(changed, airport, "2025-07", 512)
    np.testing.assert_allclose(original[0], mutated[0], equal_nan=True)
    np.testing.assert_allclose(original[1], mutated[1], equal_nan=True)
    assert inputs(hourly, airport, "2026-07", 512)[3].max() == pd.Timestamp("2025-06-30 23:00", tz="UTC")
    changed = hourly.copy()
    changed.loc[changed.index.get_level_values(1).month.isin([7, 11, 12]), "target"] = -999999
    for month in ["2025-08", "2025-11", "2025-12"]:
        original = inputs(hourly, airport, month, 512)
        mutated = inputs(changed, airport, month, 512)
        np.testing.assert_allclose(original[0], mutated[0], equal_nan=True)
        np.testing.assert_allclose(original[1], mutated[1], equal_nan=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--airports", nargs="+")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.months = args.months or ([f"2025-{m:02}" for m in range(2, 13)] +
        ([] if args.mode == "dev" else ["2026-01", "2026-07"]))
    args.output = args.output or PROCESSED / f"timesfm_hourly_{args.mode}.parquet"
    excluded_months = (7, 11, 12) if args.mode == "dev" else ()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED / "timesfm_hourly_inputs.parquet"
    hourly = pd.read_parquet(cache) if cache.exists() and not args.refresh else aggregate()
    if not cache.exists() or args.refresh:
        hourly.to_parquet(cache)
    self_check(hourly)
    if args.prepare_only:
        print(f"Prepared {cache}; label-exclusion checks passed", flush=True)
        return
    airports = args.airports or sorted(hourly.index.get_level_values(0).unique())
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    import torch
    torch.set_num_threads(4)
    started = time.perf_counter()
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=CHECKPOINT, revision=REVISION,
        per_core_batch_size=1, device=args.device))
    print(f"Model loaded in {time.perf_counter() - started:.2f}s", flush=True)
    parts_dir = args.output.parent / (args.output.stem + "_parts")
    parts_dir.mkdir(exist_ok=True)
    rows = []
    timings = []
    for month in args.months:
        for airport in airports:
            part_path = parts_dir / f"{airport}_{month}_{args.context}.parquet"
            if part_path.exists() and not args.refresh:
                rows.append(pd.read_parquet(part_path))
                continue
            target, covariates, times, history = inputs(hourly, airport, month, args.context, excluded_months)
            dev_part = PROCESSED / "timesfm_hourly_dev_parts" / part_path.name
            if args.mode == "final" and dev_part.exists() and not args.refresh:
                dev_target, dev_covariates, _, _ = inputs(hourly, airport, month, args.context)
                if np.array_equal(target, dev_target, equal_nan=True) and np.array_equal(
                        covariates, dev_covariates, equal_nan=True):
                    result = pd.read_parquet(dev_part)
                    result.to_parquet(part_path, index=False)
                    rows.append(result)
                    print(f"{airport} {month}: reused identical development model inputs", flush=True)
                    continue
            started = time.perf_counter()
            output = list(model.predict_batch([target], len(times),
                past_future_covariates=[covariates], return_quantiles=True,
                use_symmetric_averaging=False, make_positive=True, sort_quantiles=True,
                use_znorm=False, padding_mode="none"))[0]
            elapsed = time.perf_counter() - started
            quantiles = np.asarray(output.quantiles, dtype=np.float32)
            assert quantiles.shape == (len(times), 9) and np.isfinite(quantiles).all()
            result = pd.DataFrame({"ADEP_mvt": airport, "hour": times})
            for q in range(9):
                result[f"tfm_q{q + 1}"] = quantiles[:, q]
            result["tfm_median"] = output.forecast.astype(np.float32)
            # Equal-weight decile average is a truncated quantile mean proxy,
            # not the model's unavailable exact predictive expectation.
            result["tfm_decile_mean"] = quantiles.mean(axis=-1)
            result["tfm_width"] = quantiles[:, -1] - quantiles[:, 0]
            result.to_parquet(part_path, index=False)
            rows.append(result)
            timings.append({"airport": airport, "month": month, "seconds": elapsed,
                "history_end": str(history.max()), "horizon": len(times)})
            print(f"{airport} {month}: {elapsed:.2f}s median={result.tfm_median.mean():.1f}", flush=True)
    features = pd.concat(rows, ignore_index=True)
    assert not features.duplicated(["ADEP_mvt", "hour"]).any()
    features.to_parquet(args.output, index=False)
    metadata = {"checkpoint": CHECKPOINT, "revision": REVISION, "context_hours": args.context,
        "mode": args.mode, "globally_excluded_departure_target_months": list(excluded_months),
        "covariates": COVARIATES, "july_2026_history": "seasonal retrieval from June 2025",
        "target_exclusion": "Forecast month has no target values in model inputs",
        "rows": len(features), "calls": timings,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if args.device.startswith("cuda") else None}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: {len(features)} hourly rows", flush=True)


if __name__ == "__main__":
    main()
