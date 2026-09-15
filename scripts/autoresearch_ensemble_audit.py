"""Three conservative, CPU-only residual recalibrations of frozen V3."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, TIME, Y, P = "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "prediction"


def rmse(y, p, m):
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def state(aobt, eobt):
    missing = aobt <= -100000
    faulty = ~missing & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    return np.where(missing, "missing", np.where(faulty, "faulty", "normal"))


def main():
    pred = pd.read_parquet(ROOT / "artifacts/v3_validation.parquet")
    rows = pd.read_parquet(ROOT / "artifacts/rows.parquet", columns=[ID, TIME, "ADEP_mvt", Y])
    feat = pd.read_parquet(ROOT / "artifacts/features.parquet", columns=["mvt_minus_AOBT_3_flt", "mvt_minus_EOBT_1_flt"])
    rows[["aobt", "eobt"]] = feat.to_numpy()
    d = rows.merge(pred, on=ID, how="inner", validate="one_to_one")
    d[TIME] = pd.to_datetime(d[TIME], utc=True)
    d = d[d[TIME].dt.month.isin([7, 11]) & d[Y].notna()].copy()
    assert len(d) == len(pred)
    d["month"] = d[TIME].dt.month
    d["state"] = state(d.aobt.to_numpy(), d.eobt.to_numpy())
    d["resid"] = d[Y] - d[P]
    cal = d[TIME].dt.day.le(14).to_numpy()
    hold = ~cal
    y, base = d[Y].to_numpy(float), d[P].to_numpy(float)
    trials = []
    specs = [("v2", "v2_validation.parquet", "v2_ranking.parquet"),
             ("cat_missing", "v3_cat_missing_validation.parquet", "v3_cat_missing_ranking.parquet"),
             ("missing_shrunk", "moe_missing_arrival_v3_shrunk_complete_validation.parquet", None)]
    candidates = {}
    for name, filename, ranking in specs:
        alt = pd.read_parquet(ROOT / "artifacts" / filename).set_index(ID).loc[d[ID], P].to_numpy(float)
        delta = alt - base
        denom = float(delta[cal] @ delta[cal])
        alpha = float(np.clip(delta[cal] @ (y[cal] - base[cal]) / denom, 0, 1)) if denom else 0.
        candidate = base + alpha * delta
        scores = {str(m): {"baseline": rmse(y, base, hold & d.month.eq(m).to_numpy()),
                           "candidate": rmse(y, candidate, hold & d.month.eq(m).to_numpy())}
                  for m in (7, 11)}
        trials.append({"name": name, "cache": filename, "ranking_cache": ranking, "alpha": alpha,
                       "weights": {"v3": 1-alpha, name: alpha}, "scores_days15_plus": scores,
                       "correction_abs_mean": float(np.mean(np.abs(alpha * delta))),
                       "correction_abs_max": float(np.max(np.abs(alpha * delta)))})
        candidates[name] = (candidate, alpha, ranking)

    err2 = d.resid.to_numpy() ** 2
    top = np.argsort(err2)[::-1]
    n1 = max(1, len(d) // 100)
    group = d.assign(sse=err2).groupby(["ADEP_mvt", "month", "state"], observed=True).agg(
        rows=(ID, "size"), sse=("sse", "sum"), rmse=("resid", lambda x: float(np.sqrt(np.mean(x*x)))))
    group = group.sort_values("sse", ascending=False).head(20).reset_index().to_dict("records")
    robust = [t for t in trials if all(t["scores_days15_plus"][str(m)]["candidate"] <
                                      t["scores_days15_plus"][str(m)]["baseline"] for m in (7, 11))]
    chosen = min(robust, key=lambda t: sum(t["scores_days15_plus"][str(m)]["candidate"] for m in (7, 11))) if robust else None
    report = {"december_read": False, "calibration": "UTC days 1-14", "evaluation": "UTC days 15+",
              "baseline_holdout": {str(m): rmse(y, base, hold & d.month.eq(m).to_numpy()) for m in (7, 11)},
              "trials": trials, "selected": chosen["name"] if chosen else None,
              "top_1pct_sse_share": float(err2[top[:n1]].sum() / err2.sum()),
              "top_error_groups": group}

    if chosen:
        name = chosen["name"]
        _, alpha, ranking_file = candidates[name]
        if ranking_file is None:
            raise RuntimeError("Robust candidate lacks a matching ranking cache")
        sub = pd.read_parquet(ROOT / "submissions/elegant-alligator_v3.parquet")
        rank = pd.read_parquet(ROOT / "artifacts" / ranking_file).set_index(ID).loc[sub[ID], P].to_numpy(float)
        out = sub.copy()
        out[Y] = (1-alpha) * out[Y].to_numpy(float) + alpha * rank
        path = ROOT / "submissions/autoresearch_ensemble_v3.parquet"
        out.to_parquet(path, index=False)
        report["ranking_output"] = str(path.relative_to(ROOT)).replace("\\", "/")
    (ROOT / "docs/autoresearch_ensemble_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
