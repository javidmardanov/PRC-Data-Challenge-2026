"""Score arrival-context D8 as a bounded correction to the complete v3 base."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble
import timesfm_blend_research as blend

ROOT, ART = blend.ROOT, blend.ART
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED


def metrics(y, p, month, mask):
    rmse = lambda m: float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))
    return {"pooled": rmse(mask), "july": rmse(mask & (month == 7)), "november": rmse(mask & (month == 11))}


ids, x = ensemble.matrix(blend.BASES)
labels = ensemble.validation_labels(ART / "rows.parquet", ids)
metadata = blend.metadata_for(ids)
target = labels[TARGET].to_numpy(float)
month = labels[TIME].dt.month.to_numpy()
even = labels[TIME].dt.day.mod(2).eq(0).to_numpy()
known = metadata.mvt_minus_AOBT_3_flt.to_numpy(float) > -100000
nonrome = metadata.ADEP_mvt.ne("LIRF").to_numpy()
route = blend.make_route(ids, metadata, ART / "missing_validation.parquet",
    ART / "lirf_identity_expert_validation.parquet", ROOT / "data/processed/timing_bounds.parquet",
    ART / "moe_normal_cap12000_raw_missing_validation.parquet")
v1 = blend.routed(blend.PRIOR, x, route)
route = (*route[:-1], v1[route[7][route[10]]])
config = json.loads((ROOT / "docs/v2_ensemble.json").read_text())
w7 = np.asarray(config["frozen_blend"]["july_weights"])
winter = np.asarray(config["frozen_blend"]["november_weights"])
seasonal = lambda values: np.where(month == 7, blend.routed(w7, values, route), blend.routed(winter, values, route))
raw = ensemble.align(ensemble.read_predictions(ART / "tfm_arrival_d8_v3_predictions.parquet"), ids, "arrival candidate")[PRED].to_numpy(float)
base_routed = seasonal(x)
baseline = ensemble.align(ensemble.read_predictions(ART / "v3_cat_missing_validation.parquet"), ids, "complete v3 base")[PRED].to_numpy(float)
trials = []
predictions = {}
for name, gate in (("known_all", known), ("known_nonrome", known & nonrome)):
    hybrid = x.copy()
    hybrid[gate, 0] = raw[gate]
    delta = seasonal(hybrid) - base_routed
    fit = even & gate
    denominator = float(delta[fit] @ delta[fit])
    alpha = float(np.clip(delta[fit] @ (target[fit] - baseline[fit]) / denominator, 0, 1)) if denominator else 0.
    candidate = baseline.copy()
    candidate[gate] = np.clip(candidate[gate] + alpha * delta[gate], route[5][gate], route[6][gate])
    transfer = {}
    for source, destination in ((7, 11), (11, 7)):
        source_fit = fit & (month == source)
        denom = float(delta[source_fit] @ delta[source_fit])
        a = float(np.clip(delta[source_fit] @ (target[source_fit] - baseline[source_fit]) / denom, 0, 1)) if denom else 0.
        p = baseline.copy()
        p[gate] = np.clip(p[gate] + a * delta[gate], route[5][gate], route[6][gate])
        check = (~even) & (month == destination)
        transfer[f"m{source}_to_m{destination}"] = {"alpha": a, "destination_odd": float(np.sqrt(np.mean((target[check] - p[check]) ** 2)))}
    trial = {"name": name, "gate_rows": int(gate.sum()), "alpha_even": alpha,
             "all": metrics(target, candidate, month, np.ones(len(ids), bool)),
             "odd_check": metrics(target, candidate, month, ~even), "cross_month": transfer}
    trials.append(trial)
    predictions[name] = candidate
base_scores = {"all": metrics(target, baseline, month, np.ones(len(ids), bool)), "odd_check": metrics(target, baseline, month, ~even)}
eligible = [t for t in trials if t["odd_check"]["july"] <= base_scores["odd_check"]["july"] and t["odd_check"]["november"] <= base_scores["odd_check"]["november"]]
selected = min(eligible, key=lambda t: t["odd_check"]["july"] + t["odd_check"]["november"]) if eligible else {"name": "none", "alpha_even": 0.}
output = ART / "timesfm_arrival_d8_v3_gated_validation.parquet"
chosen = baseline if selected["name"] == "none" else predictions[selected["name"]]
pd.DataFrame({ID: ids, PRED: chosen}).to_parquet(output, index=False)
digest = hashlib.sha256(output.read_bytes()).hexdigest()
report = {"lane": "timesfm_arrival_context_v3", "december_read": False, "training_excluded_months": [7, 11, 12],
          "candidate": "artifacts/tfm_arrival_d8_v3_predictions.parquet", "model": "artifacts/tfm_arrival_d8_v3.cbm",
          "comparator": "artifacts/tfm_queue_d8_predictions.parquet", "complete_base": "artifacts/v3_cat_missing_validation.parquet",
          "raw_known_rmse": {"arrival": 224.87275013873614, "comparator": 227.95751752263178}, "baseline": base_scores,
          "trials": trials, "selected": selected, "output": str(output.relative_to(ROOT)).replace("\\", "/"),
          "output_sha256": digest, "output_rows": len(ids), "outside_selected_gate_array_equal_base": bool(np.array_equal(chosen[~(known & nonrome)], baseline[~(known & nonrome)])),
          "selection_rule": "fit alpha on even dates; retain only if complete odd-date RMSE improves in both July and November",
          "apply": "Replace seasonal ensemble column0 with arrival model only on selected known gate, subtract original routed seasonal base, add frozen alpha to complete v3 base, and clip gate rows to timing bounds."}
(ROOT / "docs/research_timesfm_arrival_v3.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
