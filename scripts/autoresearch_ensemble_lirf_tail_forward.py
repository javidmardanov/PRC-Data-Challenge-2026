"""Strict forward check of the frozen LIRF 54k--86.4k tail policy."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_signals import DATA
from missing_specialist import read_missing

ROOT = Path(__file__).resolve().parents[1]
ID, TIME, Y, P = "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "prediction"
SCHED = "dt_SCHED_TIME_UTC_mvt"
LOW, HIGH = 54000, 86400


def fit_tail(d, shrink):
    rome = d.ADEP_mvt.eq("LIRF")
    ordinary = float(d.loc[rome & d[Y].lt(4000), Y].median())
    level = 86400 + ordinary
    sample = d.loc[rome & d[SCHED].between(LOW, HIGH)]
    x = level - sample[SCHED].to_numpy(float)
    r = sample[Y].to_numpy(float) - sample[SCHED].to_numpy(float)
    raw = float(np.clip(x @ r / max(x @ x, 1), 0, 1))
    return {"support": len(sample), "ordinary": ordinary, "day_level": level,
            "raw_weight": raw, "weight": raw * len(sample)/(len(sample)+shrink), "shrink_support": shrink}


def main():
    reports = {}
    for month in (7, 11):
        cutoff = pd.Timestamp(f"2025-{month:02}-01", tz="UTC")
        raw = pd.concat([read_missing(p) for p in sorted(DATA.glob("training*.parquet"))
                         if pd.Timestamp(p.name.split("training_")[1][:10], tz="UTC") < cutoff], ignore_index=True)
        raw = raw[raw[TIME] < cutoff]
        fold = pd.read_parquet(ROOT / f"artifacts/forward_v3_2025-{month:02}_missing.parquet")
        source = read_missing(next(DATA.glob(f"training_2025-{month:02}-01*.parquet")))[[ID, SCHED]]
        d = fold.merge(source, on=ID, validate="one_to_one")
        gate = d.ADEP_mvt.eq("LIRF").to_numpy() & d[SCHED].between(LOW, HIGH).to_numpy()
        y, schedule, old = d[Y].to_numpy(float), d[SCHED].to_numpy(float), d[P].to_numpy(float)
        prior, selected = fit_tail(raw, 3), fit_tail(raw, 0)
        def tail(par): return schedule + par["weight"]*(par["day_level"]-schedule)
        direct = old.copy(); direct[gate] = tail(selected)[gate]
        def sse(p): return float(np.sum((y[gate]-p[gate])**2))
        reports[str(month)] = {"training_rows": len(raw), "prior": prior, "selected": selected,
            "gate_rows": int(gate.sum()), "gate_targets": y[gate].tolist(), "gate_schedule": schedule[gate].tolist(),
            "existing_fold_raw_sse": sse(old), "prior_tail_raw_sse": sse(np.where(gate,tail(prior),old)),
            "selected_tail_raw_sse": sse(direct),
            "direct_replacement_sse_delta": sse(direct)-sse(old)}
    # Independent current-validation provenance check: protected V3 is the saved
    # specialist output bit-for-bit, so no regional route coefficient applies.
    v3 = pd.read_parquet(ROOT / "artifacts/v3_validation.parquet").set_index(ID)
    raw = pd.read_parquet(ROOT / "artifacts/missing_validation.parquet").set_index(ID)
    rows = read_missing(DATA / "training_2025-07-01_2025-08-01.parquet")
    rows = pd.concat([rows, read_missing(DATA / "training_2025-11-01_2025-12-01.parquet")])
    ids = rows.loc[rows.ADEP_mvt.eq("LIRF") & rows[SCHED].between(LOW,HIGH), ID]
    ids = ids[ids.isin(v3.index) & ids.isin(raw.index)]
    diff = v3.loc[ids,P].to_numpy()-raw.loc[ids,P].to_numpy()
    report = {"december_read": False, "fixed_gate": [LOW,HIGH],
              "interpretation": "Protected complete V3 equals missing-specialist output directly; the later protected fallback bypasses attenuation by Rome/missing routing coefficients.",
              "source_check": {"rows": len(ids), "max_abs_v3_minus_specialist": float(np.max(np.abs(diff))),
                               "bitwise_equal": bool(np.array_equal(diff, np.zeros(len(diff))))},
              "folds": reports}
    path = ROOT / "docs/autoresearch_ensemble_lirf_tail_forward.json"
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
