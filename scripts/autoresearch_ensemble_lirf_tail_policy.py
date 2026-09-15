"""Three predeclared training-only LIRF missing-tail policies over frozen V3."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_signals import DATA
from missing_specialist import read_missing

ROOT = Path(__file__).resolve().parents[1]
ID, TIME, Y, P = "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "prediction"
SCHED = "dt_SCHED_TIME_UTC_mvt"
SPECS = [(54000, 86400, 0), (54000, 86400, 1), (50000, 86400, 3)]


def params(d, low, high, shrink):
    rome = d.ADEP_mvt.eq("LIRF")
    ordinary = float(d.loc[rome & d[Y].lt(4000), Y].median())
    level = 86400 + ordinary
    sample = d.loc[rome & d[SCHED].between(low, high)]
    delta = level - sample[SCHED].to_numpy(float)
    residual = sample[Y].to_numpy(float) - sample[SCHED].to_numpy(float)
    raw = float(np.clip(delta @ residual / max(delta @ delta, 1), 0, 1))
    weight = raw * len(sample) / (len(sample) + shrink)
    return {"support": len(sample), "ordinary": ordinary, "day_level": level,
            "raw_weight": raw, "weight": weight, "lower": low, "upper": high,
            "shrink_support": shrink}


def apply(base, frame, par):
    out = base.copy()
    gate = frame.ADEP_mvt.eq("LIRF").to_numpy() & frame[SCHED].between(par["lower"], par["upper"]).to_numpy()
    s = frame[SCHED].to_numpy(float)
    out[gate] = s[gate] + par["weight"] * (par["day_level"] - s[gate])
    return out, gate


def rmse(y, p, m):
    return float(np.sqrt(np.mean((y[m]-p[m])**2))) if m.any() else None


def main():
    paths = sorted(DATA.glob("training*.parquet"))
    development = pd.concat([read_missing(p) for p in paths if "2025-12-01_2026" not in p.name], ignore_index=True)
    train = development.loc[~development[TIME].dt.month.isin([7, 11])].copy()
    valid = development.loc[development[TIME].dt.month.isin([7, 11])].copy()
    assert not train[TIME].dt.month.isin([7, 11, 12]).any()
    frozen = pd.read_parquet(ROOT / "artifacts/v3_validation.parquet").set_index(ID)
    valid = valid.loc[valid[ID].isin(frozen.index)].copy()
    base = frozen.loc[valid[ID], P].to_numpy(float)
    y = valid[Y].to_numpy(float)
    month = valid[TIME].dt.month.to_numpy()
    late = valid[TIME].dt.day.gt(14).to_numpy()
    all_rows = pd.read_parquet(ROOT / "artifacts/rows.parquet", columns=[ID, TIME, Y])
    all_rows = all_rows.merge(frozen.reset_index(), on=ID, validate="one_to_one")
    all_y, all_base = all_rows[Y].to_numpy(float), all_rows[P].to_numpy(float)
    all_month = all_rows[TIME].dt.month.to_numpy(); all_late = all_rows[TIME].dt.day.gt(14).to_numpy()
    all_pos = pd.Index(all_rows[ID]).get_indexer(valid[ID]); assert (all_pos >= 0).all()
    trials = []
    candidates = []
    for low, high, shrink in SPECS:
        par = params(train, low, high, shrink)
        candidate, gate = apply(base, valid, par)
        all_candidate = all_base.copy(); all_candidate[all_pos] = candidate
        day_sse = pd.DataFrame({"day": valid[TIME].dt.floor("D"), "delta": (y-candidate)**2-(y-base)**2,
                                "late": late, "gate": gate}).query("late").groupby("day").agg(delta=("delta","sum"), gate=("gate","sum"))
        rng = np.random.default_rng(20260915)
        boots = np.array([day_sse.delta.to_numpy()[rng.integers(0, len(day_sse), len(day_sse))].sum() for _ in range(5000)])
        trial = {"spec": par, "validation_gate_rows": int(gate.sum()), "late_gate_rows": int((gate & late).sum()),
            "late": {str(m): {"baseline": rmse(y, base, late & (month==m)),
                               "candidate": rmse(y, candidate, late & (month==m)),
                               "gate_baseline": rmse(y, base, late & gate & (month==m)),
                               "gate_candidate": rmse(y, candidate, late & gate & (month==m))} for m in (7,11)},
            "full_v3_late": {str(m): {"baseline": rmse(all_y, all_base, all_late & (all_month==m)),
                                      "candidate": rmse(all_y, all_candidate, all_late & (all_month==m))} for m in (7,11)},
            "late_sse_change": float(np.sum((y[late]-candidate[late])**2)-np.sum((y[late]-base[late])**2)),
            "bootstrap_days_probability_improvement": float(np.mean(boots < 0)),
            "late_day_breakdown": {str(k.date()): {"sse_change": float(v.delta), "gate_rows": int(v.gate)} for k,v in day_sse.iterrows() if v.gate}}
        trials.append(trial); candidates.append((candidate, gate, par))
    eligible = [i for i,t in enumerate(trials) if t["late_sse_change"] < 0 and all(
        t["late"][str(m)]["candidate"] <= t["late"][str(m)]["baseline"] for m in (7,11))]
    selected = min(eligible, key=lambda i: trials[i]["late_sse_change"]) if eligible else None
    report = {"december_scored": False, "development_excluded_months": [7,11,12],
              "predeclared_specs": [list(x) for x in SPECS], "trials": trials,
              "selected_index": selected, "ranking_output": None}
    if selected is not None:
        december = read_missing(next(DATA.glob("training_2025-12-01*.parquet")))
        all_train = pd.concat([development, december], ignore_index=True)
        final_par = params(all_train, *SPECS[selected])
        ranking = read_missing(DATA / "ranking.parquet")
        sub = pd.read_parquet(ROOT / "submissions/elegant-alligator_v3.parquet")
        pos = pd.Index(sub[ID]).get_indexer(ranking[ID]); assert (pos >= 0).all()
        values, gate = apply(sub[Y].to_numpy(float)[pos], ranking, final_par)
        out = sub.copy(); out.loc[pos, Y] = values
        path = ROOT / "submissions/autoresearch_ensemble_lirf_tail_policy.parquet"
        out.to_parquet(path, index=False)
        report.update(final_parameters=final_par, ranking_gate_rows=int(gate.sum()), ranking_output=str(path.relative_to(ROOT)).replace("\\","/"))
    path = ROOT / "docs/autoresearch_ensemble_lirf_tail_policy.json"
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
