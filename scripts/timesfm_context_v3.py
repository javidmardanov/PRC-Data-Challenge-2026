"""Screen a TimesFM context-length residual against the frozen v2 pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED


def score(y, p, mask):
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


ids, base = ensemble.matrix([ART / "v2_validation.parquet"])
base = base[:, 0]
_, pair = ensemble.matrix([ART / "tfm_search_3_predictions.parquet",
                           ART / "timesfm_c168_search3_v3_predictions.parquet"], ids)
labels = ensemble.validation_labels(ART / "rows.parquet", ids)
meta = pd.read_parquet(ART / "rows.parquet", columns=[ID, TIME])
features = pd.read_parquet(ART / "features.parquet", columns=["mvt_minus_AOBT_3_flt"])
meta["known"] = features.iloc[:, 0].to_numpy() > -100000
meta = ensemble.align(meta.loc[meta[ID].isin(ids)], ids, "validation metadata")
rome = pd.read_parquet(ART / "lirf_identity_expert_validation.parquet", columns=[ID])
bounds = pd.read_parquet(ROOT / "data/processed/timing_bounds.parquet", columns=ensemble.BOUND_COLUMNS)
bounds = bounds.set_index(ID).loc[ids]
lower = bounds.lower_bound.fillna(-np.inf).to_numpy(float)
upper = bounds.upper_bound.fillna(np.inf).to_numpy(float)
y = labels[TARGET].to_numpy(float)
month = labels[TIME].dt.month.to_numpy()
even = labels[TIME].dt.day.mod(2).eq(0).to_numpy()
known = meta.known.to_numpy()
nonrome = ~pd.Index(ids).isin(rome[ID])
delta = pair[:, 1] - pair[:, 0]
trials = []
for name, gate in (("known_all", known), ("known_nonrome", known & nonrome)):
    fit = even & gate
    alpha = float(np.clip(delta[fit] @ (y[fit] - base[fit]) / (delta[fit] @ delta[fit]), 0, 1))
    prediction = base.copy()
    prediction[gate] = np.clip(base[gate] + alpha * delta[gate], lower[gate], upper[gate])
    scores = {f"m{m}_{part}": score(y, prediction, (month == m) & partmask)
              for m in (7, 11) for part, partmask in (("even", even), ("odd", ~even), ("all", np.ones(len(y), bool)))}
    trials.append({"name": name, "gate_rows": int(gate.sum()), "alpha": alpha, "scores": scores})
base_scores = {f"m{m}_{part}": score(y, base, (month == m) & partmask)
               for m in (7, 11) for part, partmask in (("even", even), ("odd", ~even), ("all", np.ones(len(y), bool)))}
eligible = [t for t in trials if all(t["scores"][f"m{m}_odd"] <= base_scores[f"m{m}_odd"] for m in (7, 11))]
selected = min(eligible, key=lambda t: t["scores"]["m7_odd"] + t["scores"]["m11_odd"]) if eligible else {"name": "none", "alpha": 0.}
report = {"december_read": False, "training_excluded_months": [7, 11, 12],
          "contexts": {"incumbent": 512, "candidate": 168},
          "native_candidate": "data/processed/timesfm_hourly_dev_c168_v3.parquet",
          "native_metadata": {"path": "data/processed/timesfm_hourly_dev_c168_v3.json", "checkpoint": "google/timesfm-3.0-pytorch", "revision": "43046b85ec22d584a13f8098c2ed39c889e129c2", "rows": 80160, "peak_gpu_gb": 1.347249664, "sha256": "1768474f944cfc5e5cd02fd417269103dfaee2bbc472943f74aa4912102b851a"},
          "matched_models": ["artifacts/tfm_search_3_predictions.parquet", "artifacts/timesfm_c168_search3_v3_predictions.parquet"],
          "matched_params": {"sample": 600000, "seed": 42, "known_only": True, "iterations": 1400, "depth": 8, "learning_rate": .07505790178893094, "l2": 4.090915852190167},
          "base": "artifacts/v2_validation.parquet", "base_scores": base_scores,
          "trials": trials, "selected": selected,
          "selection_rule": "retain only if both July and November odd-date complete RMSE do not worsen",
          "january_gap": "Jan-2025 has no TimesFM history while Jan-2026 does; no January-specific parameter was fit.",
          "next_question": "Only test context336 if context168 improves both routed odd-date months."}
(ROOT / "docs/research_timesfm_v3.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
