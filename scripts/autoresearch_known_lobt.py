"""Train and assess a LOBT-residual expert against frozen complete V3."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import ensemble

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED


def rmse(y, p, mask):
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def load_features(final=False):
    x = pd.read_parquet(ART / "features.parquet")
    rows = pd.read_parquet(ART / "rows.parquet")
    suffix = "final" if final else "dev"
    paths = [
        ROOT / f"data/processed/timesfm_hourly_{suffix}.parquet",
        ROOT / "data/processed/weather_hourly.parquet",
        ROOT / "data/processed/queue_arrival_context_v3.parquet",
    ]
    for path in paths:
        add = pd.read_parquet(path)
        if ID in add:
            add = rows[[ID]].merge(add, on=ID, how="left", validate="one_to_one").drop(columns=ID)
        else:
            keys = rows[["ADEP_mvt"]].assign(hour=rows[TIME].dt.floor("h"))
            add = keys.merge(add, on=["ADEP_mvt", "hour"], how="left", validate="many_to_one").drop(columns=["ADEP_mvt", "hour"])
        for column in add:
            x[column] = add[column].fillna(-999999).astype("float32")
    x["stand_runway"] = x["airport_stand"].astype(str) + "_" + x["RUNWAY_mvt"].astype(str)
    x["airport_operator"] = x["ADEP_mvt"].astype(str) + "_" + x["AIRCRAFT_OPERATOR_flt"].astype(str)
    x["runway_aircraft"] = x["airport_runway"].astype(str) + "_" + x["AIRCRAFT_TYPE_mvt"].astype(str)
    lobt, aobt, eobt = (x[c] for c in ("mvt_minus_LOBT_flt", "mvt_minus_AOBT_3_flt", "mvt_minus_EOBT_1_flt"))
    x["lobt_residual_floor"] = (lobt - 3606).astype("float32")
    x["lobt_residual_ceiling"] = (lobt + 3606).astype("float32")
    x["lobt_aobt_abs"] = (lobt - aobt).abs().astype("float32")
    x["aobt_delta_below_300"] = aobt.clip(upper=300).astype("float32")
    x["aobt_delta_above_2200"] = aobt.clip(lower=2200).astype("float32")
    x["eobt_delta_above_3600"] = eobt.clip(lower=3600).astype("float32")
    for column in x.select_dtypes(["object", "category"]):
        x[column] = x[column].astype("category")
    return x, rows


def train(args):
    x, rows = load_features(args.final)
    month = rows[TIME].dt.month
    known = rows.source.eq("training") & rows[TARGET].notna()
    usable = x.mvt_minus_AOBT_3_flt.gt(-100000) & x.mvt_minus_LOBT_flt.gt(-100000)
    if args.rolling_month:
        start = pd.Timestamp(args.rolling_month + "-01", tz="UTC")
        train_mask = known & usable & rows[TIME].lt(start) & ~month.isin([7, 11, 12])
        valid = known & usable & rows[TIME].ge(start) & rows[TIME].lt(start + pd.offsets.MonthBegin(1))
    else:
        train_mask = known & usable if args.final else known & usable & ~month.isin([7, 11, 12])
        valid = known & month.isin([7, 11]) & usable
    target = rows[TARGET].to_numpy(float) - x.mvt_minus_LOBT_flt.to_numpy(float)
    cats = x.select_dtypes(["object", "category"]).columns.tolist()
    model = CatBoostRegressor(
        iterations=args.iterations, depth=args.depth, learning_rate=args.rate,
        l2_leaf_reg=args.l2, loss_function="RMSE", task_type=args.device,
        devices="0" if args.device == "GPU" else None, random_seed=args.seed,
        thread_count=args.threads, border_count=args.border_count,
        one_hot_max_size=20, max_ctr_complexity=1, gpu_ram_part=.55,
        allow_writing_files=True, train_dir=str(ART / f"{args.name}_logs"))
    started = time.time()
    # Fixed iterations keep both calibration and later-check labels out of model selection.
    model.fit(Pool(x.loc[train_mask], target[train_mask], cat_features=cats), verbose=100)
    model.save_model(str(ART / f"{args.name}.cbm"))
    selected = rows.source.eq("ranking") if args.final else (valid if args.rolling_month else known & month.isin([7, 11]))
    pred = model.predict(x.loc[selected], thread_count=args.threads) + x.loc[selected, "mvt_minus_LOBT_flt"].to_numpy(float)
    result = rows.loc[selected, [ID, TIME, "ADEP_mvt", TARGET]].copy()
    result[PRED] = pred
    result.to_parquet(ART / f"{args.name}_predictions.parquet", index=False)
    info = {"args": vars(args), "training_rows": int(train_mask.sum()), "usable_validation_rows": int(valid.sum()),
            "trees": model.tree_count_, "seconds": time.time() - started, "excluded_months": [] if args.final else [7, 11, 12],
            "rolling_month": args.rolling_month}
    (ART / f"{args.name}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


def score(args):
    raw = ensemble.read_predictions(ART / f"{args.name}_predictions.parquet")
    base = ensemble.read_predictions(ART / "v3_validation.parquet")
    ids = base[ID].to_numpy()
    raw = ensemble.align(raw, ids, "LOBT candidate")
    rows = pd.read_parquet(ART / "rows.parquet")
    feat = pd.read_parquet(ART / "features.parquet", columns=["mvt_minus_AOBT_3_flt", "mvt_minus_LOBT_flt", "mvt_minus_EOBT_1_flt"])
    rows = rows.assign(**{c: feat[c].to_numpy() for c in feat})
    labels = ensemble.validation_labels(ART / "rows.parquet", ids)
    meta = ensemble.align(rows.loc[rows[ID].isin(ids)], ids, "metadata")
    bounds = pd.read_parquet(ROOT / "data/processed/timing_bounds.parquet").set_index(ID).loc[ids]
    y, incumbent, candidate = labels[TARGET].to_numpy(float), base[PRED].to_numpy(float), raw[PRED].to_numpy(float)
    month = labels[TIME].dt.month.to_numpy()
    early = labels[TIME].dt.day.le(14).to_numpy()
    known = meta.mvt_minus_AOBT_3_flt.to_numpy(float) > -100000
    lobt = meta.mvt_minus_LOBT_flt.to_numpy(float)
    aobt = meta.mvt_minus_AOBT_3_flt.to_numpy(float)
    eobt = meta.mvt_minus_EOBT_1_flt.to_numpy(float)
    usable = known & (lobt > -100000) & np.isfinite(candidate)
    faulty = (aobt < 300) | (aobt > 2200) | (eobt > 3600)
    disagree = np.abs(lobt - aobt) > args.disagreement
    gates = {"faulty_or_disagree": usable & (faulty | disagree), "disagree": usable & disagree, "all_lobt": usable}
    lower = np.maximum(0., bounds.lower_bound.fillna(-np.inf).to_numpy(float))
    upper = bounds.upper_bound.fillna(np.inf).to_numpy(float)
    delta = candidate - incumbent
    trials = []
    outputs = {}
    for name, gate in gates.items():
        fit = early & gate
        denom = float(delta[fit] @ delta[fit])
        alpha = float(np.clip(delta[fit] @ (y[fit] - incumbent[fit]) / denom, 0, 1)) if denom else 0.
        pred = incumbent.copy()
        pred[gate] = np.clip(pred[gate] + alpha * delta[gate], lower[gate], upper[gate])
        metric = lambda mask: {"pooled": rmse(y, pred, mask), "july": rmse(y, pred, mask & (month == 7)), "november": rmse(y, pred, mask & (month == 11))}
        trials.append({"gate": name, "gate_rows": int(gate.sum()), "alpha_early": alpha,
                       "early_calibration": metric(early), "later_check": metric(~early), "all": metric(np.ones(len(y), bool))})
        outputs[name] = pred
    baseline_metric = lambda mask: {"pooled": rmse(y, incumbent, mask), "july": rmse(y, incumbent, mask & (month == 7)), "november": rmse(y, incumbent, mask & (month == 11))}
    baseline = {"early_calibration": baseline_metric(early), "later_check": baseline_metric(~early), "all": baseline_metric(np.ones(len(y), bool))}
    eligible = [t for t in trials if t["later_check"]["july"] < baseline["later_check"]["july"] and t["later_check"]["november"] < baseline["later_check"]["november"]]
    selected = min(eligible, key=lambda t: t["later_check"]["july"] + t["later_check"]["november"]) if eligible else None
    output = None
    if selected:
        output = ART / f"{args.name}_gated_validation.parquet"
        pd.DataFrame({ID: ids, PRED: outputs[selected["gate"]]}).to_parquet(output, index=False)
    report = {"lane": "known_lobt_residual", "december_read": False, "official_scores_used": False,
              "training_excluded_months": [7, 11, 12], "candidate": f"artifacts/{args.name}_predictions.parquet",
              "calibration": "UTC days 1-14 in July and November 2025", "check": "UTC days 15-end in each month",
              "baseline": baseline, "trials": trials, "selected": selected,
              "output": None if output is None else str(output.relative_to(ROOT)).replace("\\", "/"),
              "output_sha256": None if output is None else hashlib.sha256(output.read_bytes()).hexdigest(),
              "apply": "On the selected gate, add frozen alpha times (LOBT raw expert minus complete V3), then clip to timing bounds."}
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def score_forward(args):
    """Apply the frozen development alpha to one chronological fold."""
    base = ensemble.read_predictions(args.baseline)
    ids = base[ID].to_numpy()
    raw = ensemble.read_predictions(ART / f"{args.name}_predictions.parquet")
    positions = pd.Index(ids).get_indexer(raw[ID])
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise ValueError("Forward LOBT IDs must be a unique subset of baseline IDs")
    rows = pd.read_parquet(ART / "rows.parquet")
    labels = ensemble.align(rows.loc[rows[ID].isin(ids), [ID, TIME, TARGET]], ids, "forward labels")
    if labels[TARGET].isna().any():
        raise ValueError("Forward baseline contains rows without labels")
    feat = pd.read_parquet(ART / "features.parquet", columns=["mvt_minus_AOBT_3_flt", "mvt_minus_LOBT_flt"])
    rows = rows.assign(**{c: feat[c].to_numpy() for c in feat})
    meta = ensemble.align(rows.loc[rows[ID].isin(ids)], ids, "forward metadata")
    bounds = pd.read_parquet(ROOT / "data/processed/timing_bounds.parquet").set_index(ID).loc[ids]
    gate = meta.mvt_minus_AOBT_3_flt.gt(-100000).to_numpy() & meta.mvt_minus_LOBT_flt.gt(-100000).to_numpy()
    if not np.array_equal(np.flatnonzero(gate), np.sort(positions)):
        raise ValueError("Forward raw expert rows must exactly equal the known AOBT+LOBT gate")
    y, incumbent = labels[TARGET].to_numpy(float), base[PRED].to_numpy(float)
    candidate = incumbent.copy()
    candidate[positions] = raw[PRED].to_numpy(float)
    prediction = incumbent.copy()
    lower = np.maximum(0., bounds.lower_bound.fillna(-np.inf).to_numpy(float))
    prediction[gate] = np.clip(incumbent[gate] + args.alpha * (candidate[gate] - incumbent[gate]),
                               lower[gate], bounds.upper_bound.fillna(np.inf).to_numpy()[gate])
    output = ART / f"{args.name}_fixed_alpha_validation.parquet"
    pd.DataFrame({ID: ids, PRED: prediction}).to_parquet(output, index=False)
    report = {"baseline": str(args.baseline), "candidate": f"artifacts/{args.name}_predictions.parquet",
              "alpha_frozen_from_development": args.alpha, "rows": len(ids), "gate_rows": int(gate.sum()),
              "baseline_rmse": rmse(y, incumbent, np.ones(len(y), bool)),
              "candidate_rmse": rmse(y, prediction, np.ones(len(y), bool)),
              "output": str(output.relative_to(ROOT)).replace("\\", "/"),
              "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    print(json.dumps(report, indent=2))


def apply_ranking(args):
    base = pd.read_parquet(ROOT / "submissions/elegant-alligator_v3.parquet")
    if base.columns.tolist() != [ID, TARGET] or len(base) != 344841:
        raise ValueError("Frozen V3 must preserve the submission template schema")
    base = base.rename(columns={TARGET: PRED})
    ensemble.require_ids(base, "frozen V3 submission")
    if not np.isfinite(base[PRED].to_numpy(float)).all():
        raise ValueError("Frozen V3 predictions must be finite")
    raw = ensemble.align(ensemble.read_predictions(ART / f"{args.name}_predictions.parquet"), base[ID], "final LOBT candidate")
    rows = pd.read_parquet(ART / "rows.parquet")
    feat = pd.read_parquet(ART / "features.parquet", columns=["mvt_minus_AOBT_3_flt", "mvt_minus_LOBT_flt"])
    rows = rows.assign(**{c: feat[c].to_numpy() for c in feat})
    meta = ensemble.align(rows.loc[rows[ID].isin(base[ID])], base[ID], "ranking metadata")
    bounds = pd.read_parquet(ROOT / "data/processed/timing_bounds.parquet").set_index(ID).loc[base[ID]]
    gate = meta.mvt_minus_AOBT_3_flt.gt(-100000).to_numpy() & meta.mvt_minus_LOBT_flt.gt(-100000).to_numpy()
    incumbent, candidate = base[PRED].to_numpy(float), raw[PRED].to_numpy(float)
    result = incumbent.copy()
    lower = np.maximum(0., bounds.lower_bound.fillna(-np.inf).to_numpy(float))
    result[gate] = np.clip(incumbent[gate] + args.alpha * (candidate[gate] - incumbent[gate]),
                           lower[gate], bounds.upper_bound.fillna(np.inf).to_numpy()[gate])
    if not np.array_equal(result[~gate], incumbent[~gate]):
        raise AssertionError("Rows outside the LOBT gate changed")
    out = pd.DataFrame({ID: base[ID], TARGET: result})
    if len(out) != 344841 or not out[ID].is_unique or not np.isfinite(result).all():
        raise ValueError("Ranking output invariant failed")
    out.to_parquet(args.output, index=False)
    print(json.dumps({"output": args.output, "rows": len(out), "gate_rows": int(gate.sum()),
                      "alpha": args.alpha, "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="known_lobt_d8_a1")
    p.add_argument("--score", action="store_true")
    p.add_argument("--score-forward", action="store_true")
    p.add_argument("--final", action="store_true")
    p.add_argument("--apply-ranking", action="store_true")
    p.add_argument("--rolling-month")
    p.add_argument("--iterations", type=int, default=5000)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--rate", type=float, default=.06)
    p.add_argument("--l2", type=float, default=12.)
    p.add_argument("--border-count", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="GPU")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--disagreement", type=float, default=900.)
    p.add_argument("--alpha", type=float, default=.630797588656986)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--output", default=str(ROOT / "submissions/elegant-alligator_lobt.parquet"))
    p.add_argument("--report", default=str(ROOT / "docs/autoresearch_known.json"))
    args = p.parse_args()
    if args.score_forward and args.baseline is None:
        p.error("--score-forward requires --baseline")
    apply_ranking(args) if args.apply_ranking else score_forward(args) if args.score_forward else score(args) if args.score else train(args)
