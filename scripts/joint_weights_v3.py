"""Joint even-date fit of frozen CatBoost-tail and TimesFM-arrival deltas."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import ensemble
import timesfm_blend_research as blend

ROOT, ART = blend.ROOT, blend.ART
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED


def metric(labels, target, prediction, mask):
    month = labels[TIME].dt.month.to_numpy()
    calc = lambda keep: float(np.sqrt(np.mean((target[keep] - prediction[keep]) ** 2)))
    return {"pooled": calc(mask), "july": calc(mask & (month == 7)),
            "november": calc(mask & (month == 11))}


def main():
    ids, matrix = ensemble.matrix(blend.BASES)
    labels = ensemble.validation_labels(ART / "rows.parquet", ids)
    metadata = blend.metadata_for(ids)
    target = labels[TARGET].to_numpy(float)
    month = labels[TIME].dt.month.to_numpy()
    even = labels[TIME].dt.day.mod(2).eq(0).to_numpy()
    route = blend.make_route(ids, metadata, ART / "missing_validation.parquet",
        ART / "lirf_identity_expert_validation.parquet", ROOT / "data/processed/timing_bounds.parquet",
        ART / "moe_normal_cap12000_raw_missing_validation.parquet")
    v1 = blend.routed(blend.PRIOR, matrix, route)
    route = (*route[:-1], v1[route[7][route[10]]])
    config = json.loads((ROOT / "docs/v2_ensemble.json").read_text())
    w7 = np.asarray(config["frozen_blend"]["july_weights"])
    winter = np.asarray(config["frozen_blend"]["november_weights"])

    def seasonal(values):
        return np.where(month == 7, blend.routed(w7, values, route), blend.routed(winter, values, route))

    routed_base = seasonal(matrix)
    known = metadata.mvt_minus_AOBT_3_flt.gt(-100000).to_numpy()
    cat_gate = known & (metadata.mvt_minus_AOBT_3_flt.lt(300).to_numpy()
        | metadata.mvt_minus_AOBT_3_flt.gt(2200).to_numpy()
        | metadata.mvt_minus_EOBT_1_flt.gt(3600).to_numpy())
    cat_raw = ensemble.align(ensemble.read_predictions(ART / "catboost_tail_d8_full_v3_predictions.parquet"),
                             ids, "CatBoost tail")[PRED].to_numpy(float)
    cat_matrix = matrix.copy()
    cat_matrix[cat_gate, 0] = cat_raw[cat_gate]
    cat_delta = seasonal(cat_matrix) - routed_base

    arrival_gate = known & metadata.ADEP_mvt.ne("LIRF").to_numpy()
    arrival_raw = ensemble.align(ensemble.read_predictions(ART / "tfm_arrival_d8_v3_predictions.parquet"),
                                 ids, "TimesFM arrival")[PRED].to_numpy(float)
    arrival_matrix = matrix.copy()
    arrival_matrix[arrival_gate, 0] = arrival_raw[arrival_gate]
    arrival_delta = seasonal(arrival_matrix) - routed_base
    baseline = ensemble.align(ensemble.read_predictions(
        ART / "moe_missing_arrival_v3_fixed_complete_validation.parquet"), ids, "missing-arrival base")[PRED].to_numpy(float)

    def predict(coefficients):
        result = baseline.copy()
        result[cat_gate] = np.clip(result[cat_gate] + coefficients[0] * cat_delta[cat_gate],
                                   route[5][cat_gate], route[6][cat_gate])
        result[arrival_gate] = np.clip(result[arrival_gate] + coefficients[1] * arrival_delta[arrival_gate],
                                       route[5][arrival_gate], route[6][arrival_gate])
        return result

    current_weights = np.array([0.5441252012777676, 1.0])
    current = predict(current_weights)
    frozen = ensemble.align(ensemble.read_predictions(ART / "v3_validation.parquet"), ids, "frozen v3")[PRED].to_numpy(float)
    np.testing.assert_allclose(current, frozen, rtol=0, atol=1e-9)

    scale = max(float(np.std(target[even])), 1.)
    objective = lambda weights: float(np.mean(((target[even] - predict(weights)[even]) / scale) ** 2))
    fitted = minimize(objective, current_weights, method="L-BFGS-B", bounds=[(0, 1), (0, 1)],
                      options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 200})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(fitted.message)
    candidate = predict(fitted.x)
    odd = ~even
    current_odd = metric(labels, target, current, odd)
    candidate_odd = metric(labels, target, candidate, odd)
    promote = candidate_odd["july"] < current_odd["july"] and candidate_odd["november"] < current_odd["november"]
    selected = fitted.x if promote else current_weights
    report = {"december_read": False, "fit_dates": "even UTC day in July/November 2025",
        "optimizer": {"method": "L-BFGS-B", "gradient": "SciPy finite differences over two bounded scalars",
                      "bounds": [[0, 1], [0, 1]], "success": bool(fitted.success)},
        "coefficient_order": ["catboost_tail", "timesfm_arrival"],
        "current_weights": current_weights.tolist(), "joint_even_fit_weights": fitted.x.tolist(),
        "reconstruction_max_abs": float(np.max(np.abs(current-frozen))),
        "current_all": metric(labels, target, current, np.ones(len(ids), bool)),
        "joint_all": metric(labels, target, candidate, np.ones(len(ids), bool)),
        "current_odd": current_odd, "joint_odd": candidate_odd,
        "promotion_rule": "promote only if both odd-date months improve",
        "promoted": bool(promote), "selected_weights": selected.tolist()}
    (ROOT / "docs/research_joint_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
