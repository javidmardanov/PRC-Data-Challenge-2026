"""Score a full-development CatBoost model against the frozen v2 pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble
import timesfm_blend_research as blend

ROOT, ART = blend.ROOT, blend.ART
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED
CANDIDATE = ART / "catboost_tail_d8_full_v3_predictions.parquet"


def scores(labels, target, prediction, mask=None):
    mask = np.ones(len(target), bool) if mask is None else mask
    month = labels[TIME].dt.month.to_numpy()
    rmse = lambda m: float(np.sqrt(np.mean((target[m] - prediction[m]) ** 2)))
    return {"pooled": rmse(mask), "july": rmse(mask & (month == 7)),
            "november": rmse(mask & (month == 11))}


def main():
    ids, x = ensemble.matrix(blend.BASES)
    labels = ensemble.validation_labels(ART / "rows.parquet", ids)
    metadata = blend.metadata_for(ids)
    target = labels[TARGET].to_numpy(float)
    month = labels[TIME].dt.month.to_numpy()
    even = labels[TIME].dt.day.mod(2).eq(0).to_numpy()
    route = blend.make_route(ids, metadata, ART / "missing_validation.parquet",
        ART / "lirf_identity_expert_validation.parquet", ROOT / "data/processed/timing_bounds.parquet",
        ART / "moe_normal_cap12000_raw_missing_validation.parquet")
    v1 = blend.routed(blend.PRIOR, x, route)
    route = (*route[:-1], v1[route[7][route[10]]])
    config = json.loads((ROOT / "docs/v2_ensemble.json").read_text())
    w7, w11 = (np.asarray(config["frozen_blend"][key]) for key in ("july_weights", "november_weights"))

    def seasonal(matrix):
        return np.where(month == 7, blend.routed(w7, matrix, route), blend.routed(w11, matrix, route))

    base = seasonal(x)
    old_delta, old_gate = blend.tail_delta(x, route, metadata, ROOT / config["tail_delta"]["path"])
    reconstructed = base.copy()
    reconstructed[old_gate] = np.clip(reconstructed[old_gate] + config["tail_delta"]["alpha"] * old_delta[old_gate],
                                      route[5][old_gate], route[6][old_gate])
    frozen = ensemble.align(ensemble.read_predictions(ART / "v2_validation.parquet"), ids, "frozen v2")[PRED].to_numpy(float)
    if not np.allclose(reconstructed, frozen, rtol=0, atol=1e-9):
        raise AssertionError(f"v2 reconstruction mismatch: {np.max(np.abs(reconstructed-frozen))}")

    raw = ensemble.align(ensemble.read_predictions(CANDIDATE), ids, str(CANDIDATE))[PRED].to_numpy(float)
    aobt = metadata.mvt_minus_AOBT_3_flt.to_numpy(float)
    eobt = metadata.mvt_minus_EOBT_1_flt.to_numpy(float)
    gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    replaced = x.copy()
    replaced[:, 0] = raw
    replacement_base = seasonal(replaced)
    replacement = replacement_base.copy()
    replacement[old_gate] = np.clip(replacement[old_gate] + config["tail_delta"]["alpha"] * old_delta[old_gate],
                                    route[5][old_gate], route[6][old_gate])
    replacement[route[7][route[10]]] = route[11]

    hybrid = x.copy()
    hybrid[gate, 0] = raw[gate]
    delta = seasonal(hybrid) - base
    denominator = float(delta[even] @ delta[even])
    alpha = float(np.clip(delta[even] @ (target[even] - frozen[even]) / denominator, 0, 1)) if denominator else 0.
    gated = frozen.copy()
    gated[gate] = np.clip(gated[gate] + alpha * delta[gate], route[5][gate], route[6][gate])
    odd = ~even
    transfer = {}
    for source, destination in ((7, 11), (11, 7)):
        fit = even & (month == source)
        denom = float(delta[fit] @ delta[fit])
        fitted = float(np.clip(delta[fit] @ (target[fit] - frozen[fit]) / denom, 0, 1)) if denom else 0.
        candidate = frozen.copy()
        candidate[gate] = np.clip(candidate[gate] + fitted * delta[gate], route[5][gate], route[6][gate])
        check = odd & (month == destination)
        check_rmse = lambda values: float(np.sqrt(np.mean((target[check] - values[check]) ** 2)))
        transfer[f"july_to_november" if source == 7 else "november_to_july"] = {
            "alpha_fit_on_source_even": fitted,
            "destination_odd_candidate_rmse": check_rmse(candidate),
            "destination_odd_baseline_rmse": check_rmse(frozen)}
    sensitivity = []
    for factor in (.25, .5, .75):
        fixed = factor * alpha
        candidate = frozen.copy()
        candidate[gate] = np.clip(candidate[gate] + fixed * delta[gate], route[5][gate], route[6][gate])
        sensitivity.append({"factor_of_frozen_alpha": factor, "alpha": fixed,
                            "odd_check": scores(labels, target, candidate, odd),
                            "all": scores(labels, target, candidate)})
    training = json.loads((ART / "catboost_tail_d8_full_v3.json").read_text())
    report = {"lane": "catboost_tail_v3", "december_read": False,
        "training_excluded_months": [7, 11, 12],
        "candidate": str(CANDIDATE.relative_to(ROOT)).replace("\\", "/"),
        "model": "artifacts/catboost_tail_d8_full_v3.cbm", "training_args": training["args"],
        "gate_rows": int(gate.sum()), "v2_reconstruction_max_abs": float(np.max(np.abs(reconstructed-frozen))),
        "baseline": scores(labels, target, frozen), "full_base_replacement": scores(labels, target, replacement),
        "gated_alpha_even": alpha, "gated_all": scores(labels, target, gated),
        "gated_odd_check": scores(labels, target, gated, odd), "baseline_odd_check": scores(labels, target, frozen, odd),
        "cross_month_transfer": transfer, "frozen_alpha_shrink_sensitivity": sensitivity}
    output = ART / "catboost_tail_d8_full_v3_gated_validation.parquet"
    pd.DataFrame({ID: ids, PRED: gated}).to_parquet(output, index=False)
    report["output"] = str(output.relative_to(ROOT)).replace("\\", "/")
    report["selected"] = "gated correction; full replacement rejected because July regressed"
    report["apply"] = "On the saved tail gate, add alpha times the seasonally weighted, fully routed candidate-column-0 minus incumbent-column-0 delta to frozen v2, then clip those gate rows to timing bounds."
    report["reproduce"] = "python -u scripts/train.py --name catboost_tail_d8_full_v3 --iterations 6000 --depth 8 --rate 0.07505790178893094 --l2 4.090915852190167 --seed 42 --border-count 128 --threads 8 --device GPU --residual --tfm data/processed/timesfm_hourly_dev.parquet --weather data/processed/weather_hourly.parquet --extra data/processed/queue_features.parquet --interactions --known-only --tail-copies --boost-priority"
    report["proposed_final"] = "Same command with --name catboost_tail_d8_full_v3_final, --tfm data/processed/timesfm_hourly_final.parquet, --iterations 5995 and --final; pending parent review."
    report["next_hypothesis"] = "Retain the gated model unless a combined TimesFM context candidate preserves its odd-date gains in both months."
    (ROOT / "docs/research_catboost_v3.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
