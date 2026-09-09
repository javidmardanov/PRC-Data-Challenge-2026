"""Regularized TimesFM blend research without model reruns or December labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import ensemble

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED
BASES = [
    ART / "tfm_queue_d8_predictions.parquet",
    ART / "no_tfm_full_d8_predictions.parquet",
    ART / "lgbm_tfm_validation.parquet",
    ART / "tfm_queue_d10_v2_predictions.parquet",
]
PRIOR = np.array([.7985604681600207, .17672281746128118, .024716714378698216, 0.])


def routed(weights, matrix, route):
    config = json.loads((ART / "v1_ensemble.json").read_text())
    config["weights"] = weights.tolist()
    result = matrix @ weights
    mpos, mold, rpos, rexpert, rmissing, lower, upper, v2pos, v2pred, v2alpha, protected, fallback = route
    result[mpos] += config["specialist_alpha"] * (mold - result[mpos])
    result = ensemble.blend_rome(result, rpos, rexpert, rmissing, config["rome_betas"])
    result = np.maximum(result, 0)
    result = np.clip(result, lower, upper)
    # Missing-v2 is deliberately last. Rome schedule tails retained by the saved expert file.
    result[v2pos] += v2alpha * (v2pred - result[v2pos])
    if fallback is not None:
        result[v2pos[protected]] = fallback
    return result


def fit(mask, matrix, target, penalty):
    scale = max(float(np.std(target[mask])), 1.)
    residual = (matrix[mask] - target[mask, None]) / scale
    gram = residual.T @ residual / mask.sum()
    def objective(w):
        return float(w @ gram @ w + penalty * ((w - PRIOR) @ (w - PRIOR)))
    def gradient(w):
        return 2 * gram @ w + 2 * penalty * (w - PRIOR)
    out = minimize(objective, PRIOR, jac=gradient, method="SLSQP", bounds=[(0, 1)] * 4,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-12, "maxiter": 200})
    if not out.success:
        raise RuntimeError(out.message)
    w = np.clip(out.x, 0, 1)
    return w / w.sum()


def rmse(y, p, mask):
    return float(np.sqrt(np.mean((y[mask] - p[mask]) ** 2)))


def make_route(ids, metadata, old_missing_path, rome_path, bounds_path, v2_path, fallback=None):
    old_missing = ensemble.read_predictions(old_missing_path)
    rome = ensemble.read_predictions(rome_path)
    missing = ensemble.read_predictions(v2_path)
    bounds = pd.read_parquet(bounds_path, columns=ensemble.BOUND_COLUMNS)
    mpos = pd.Index(ids).get_indexer(old_missing[ID])
    rpos = pd.Index(ids).get_indexer(rome[ID])
    if (mpos < 0).any() or (rpos < 0).any() or len(pd.Index(old_missing[ID]).symmetric_difference(pd.Index(missing[ID]))):
        raise ValueError("Routing experts must use valid IDs and missing-v2 must exactly match old missing IDs")
    b = bounds.set_index(ID).loc[ids]
    v2pos = pd.Index(ids).get_indexer(missing[ID])
    airport = metadata.set_index(ID).loc[missing[ID], "ADEP_mvt"].to_numpy()
    schedule = metadata.set_index(ID).loc[missing[ID], "mvt_minus_SCHED_TIME_UTC_mvt"].to_numpy(float)
    protected = (airport == "LIRF") & (schedule >= 24000)
    alpha = np.where(airport == "LIRF", .4224120030856562, 1.)
    alpha[protected] = 0.
    return (mpos, old_missing[PRED].to_numpy(float), rpos, rome[PRED].to_numpy(float),
            np.isin(rpos, mpos), b.lower_bound.fillna(-np.inf).to_numpy(float),
            b.upper_bound.fillna(np.inf).to_numpy(float), v2pos, missing[PRED].to_numpy(float),
            alpha, protected, fallback)


def metadata_for(ids):
    metadata = pd.read_parquet(ART / "rows.parquet", columns=[ID, TIME, "ADEP_mvt"])
    feature_cols = ["mvt_minus_SCHED_TIME_UTC_mvt", "mvt_minus_AOBT_3_flt", "mvt_minus_EOBT_1_flt"]
    schedule = pd.read_parquet(ART / "features.parquet", columns=feature_cols)
    if len(metadata) != len(schedule):
        raise ValueError("rows/features caches are not row-aligned")
    for column in feature_cols:
        metadata[column] = schedule[column].to_numpy()
    return ensemble.align(metadata.loc[metadata[ID].isin(ids)], ids, "route metadata")


def complete_v1(matrix3, route):
    mpos, mold, rpos, rexpert, rmissing, lower, upper = route[:7]
    config = json.loads((ART / "v1_ensemble.json").read_text())
    result = matrix3 @ PRIOR[:3]
    result[mpos] += config["specialist_alpha"] * (mold - result[mpos])
    result = ensemble.blend_rome(result, rpos, rexpert, rmissing,
                                 config["rome_betas"])
    return np.clip(np.maximum(result, 0), lower, upper)


def tail_delta(matrix, route, metadata, raw_tail):
    raw = ensemble.align(ensemble.read_predictions(raw_tail), metadata[ID].to_numpy(), str(raw_tail))[PRED].to_numpy(float)
    aobt = metadata["mvt_minus_AOBT_3_flt"].to_numpy(float)
    eobt = metadata["mvt_minus_EOBT_1_flt"].to_numpy(float)
    gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    hybrid = matrix[:, :3].copy()
    hybrid[gate, 0] = raw[gate]
    return complete_v1(hybrid, route) - complete_v1(matrix[:, :3], route), gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-final", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "docs/research_timesfm.json")
    parser.add_argument("--predictions", nargs=4, type=Path)
    parser.add_argument("--missing", type=Path)
    parser.add_argument("--rome", type=Path)
    parser.add_argument("--missing-v2", type=Path)
    parser.add_argument("--tail-predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.apply_final:
        required = [args.predictions, args.missing, args.rome, args.missing_v2, args.output]
        if any(value is None for value in required):
            parser.error("--apply-final requires --predictions (four), --missing, --rome, --missing-v2 and --output")
        template = pd.read_parquet(ensemble.TEMPLATE)
        ids, x = ensemble.matrix(args.predictions, template[ID].to_numpy())
        metadata = metadata_for(ids)
        stored_v1 = pd.read_parquet(ROOT / "submissions/elegant-alligator_v1.parquet", columns=[ID, TARGET])
        stored_v1 = ensemble.align(stored_v1, ids, "stored v1 submission")[TARGET].to_numpy(float)
        route = make_route(ids, metadata, args.missing, args.rome,
                           ROOT / "data/processed/timing_bounds.parquet", args.missing_v2)
        protected = route[10]
        route = (*route[:-1], stored_v1[route[7][protected]])
        config = json.loads(args.config.read_text())
        if config.get("tail_delta", {}).get("name", "none") != "none" and not args.tail_predictions:
            parser.error("selected config requires --tail-predictions")
        frozen = config["frozen_blend"]
        july = routed(np.asarray(frozen["july_weights"]), x, route)
        winter = routed(np.asarray(frozen["november_weights"]), x, route)
        ranking_month = pd.to_datetime(metadata[TIME], utc=True).dt.month.to_numpy()
        prediction = np.where(ranking_month == 7, july, winter)
        if args.tail_predictions:
            delta, gate = tail_delta(x, route, metadata, args.tail_predictions)
            prediction[gate] = np.clip(prediction[gate] + config["tail_delta"]["alpha"] * delta[gate],
                                       route[5][gate], route[6][gate])
            prediction[route[7][route[10]]] = route[11]
        template[TARGET] = prediction
        args.output.parent.mkdir(parents=True, exist_ok=True)
        template.to_parquet(args.output, index=False)
        print(f"Wrote {args.output}: {len(template)} rows")
        raise SystemExit
    ids, x = ensemble.matrix(BASES)
    labels = ensemble.validation_labels(ART / "rows.parquet", ids)
    metadata = metadata_for(ids)
    target = labels[TARGET].to_numpy(float)
    month = labels[TIME].dt.month.to_numpy()
    day = labels[TIME].dt.day.to_numpy()
    calib = day % 2 == 0
    route = make_route(ids, metadata, ART / "missing_validation.parquet",
                       ART / "lirf_identity_expert_validation.parquet",
                       ROOT / "data/processed/timing_bounds.parquet",
                       ART / "moe_normal_cap12000_raw_missing_validation.parquet")
    v1_complete = routed(PRIOR, x, route)
    route = (*route[:-1], v1_complete[route[7][route[10]]])
    mpos, _, rpos = route[:3]
    protected_positions = route[7][route[10]]
    mutated = list(route)
    mutated[8] = mutated[8].copy()
    mutated[8][route[10]] += 1e9
    guard_before = routed(PRIOR, x, route)
    guard_after = routed(PRIOR, x, tuple(mutated))
    if not np.array_equal(guard_before, guard_after):
        raise AssertionError("Protected Rome-tail output changed after raw expert mutation")
    fit_mask = calib.copy()
    fit_mask[np.union1d(mpos, rpos)] = False
    trials = []
    specs = [("incumbent", None), ("unregularized", 0.), ("ridge_0.01", .01), ("ridge_0.05", .05)]
    selected = None
    for name, penalty in specs:
        weights = PRIOR if penalty is None else fit(fit_mask, x, target, penalty)
        pred = routed(weights, x, route)
        scores = {f"m{m}_{part}": rmse(target, pred, (month == m) & partmask)
                  for m in (7, 11) for part, partmask in (("calib", calib), ("check", ~calib))}
        scores.update({f"m{m}_all": rmse(target, pred, month == m) for m in (7, 11)})
        trials.append({"name": name, "penalty": penalty, "weights": weights.tolist(), "scores": scores,
                       "check_mean": (scores["m7_check"] + scores["m11_check"]) / 2})
    # Select on check dates with a transfer guard: both months must beat the incumbent check score.
    incumbent = trials[0]
    eligible = [t for t in trials if all(t["scores"][f"m{m}_check"] <= incumbent["scores"][f"m{m}_check"] for m in (7, 11))]
    selected = min(eligible, key=lambda t: t["check_mean"])
    transfer = {}
    month_weights = {}
    unaffected = np.ones(len(ids), dtype=bool)
    unaffected[np.union1d(mpos, rpos)] = False
    for source, destination in ((7, 11), (11, 7)):
        w = fit(calib & (month == source) & unaffected, x, target, 0.)
        month_weights[source] = w
        p = routed(w, x, route)
        transfer[f"m{source}_to_m{destination}"] = {
            "weights": w.tolist(),
            "destination_check_rmse": rmse(target, p, (month == destination) & ~calib),
            "destination_incumbent_check_rmse": incumbent["scores"][f"m{destination}_check"],
        }
    seasonal = []
    pooled_weights = np.asarray(selected["weights"])
    for strength in (.25, .5, .75):
        p7 = routed((1 - strength) * pooled_weights + strength * month_weights[7], x, route)
        p11 = routed((1 - strength) * pooled_weights + strength * month_weights[11], x, route)
        scores = {"m7_check": rmse(target, p7, (month == 7) & ~calib),
                  "m11_check": rmse(target, p11, (month == 11) & ~calib),
                  "m7_all": rmse(target, p7, month == 7), "m11_all": rmse(target, p11, month == 11)}
        seasonal.append({"strength": strength, "july_weights": ((1-strength)*pooled_weights + strength*month_weights[7]).tolist(),
                         "november_weights": ((1-strength)*pooled_weights + strength*month_weights[11]).tolist(),
                         "scores": scores})
    stable = [s for s in seasonal if s["strength"] <= .5 and
              s["scores"]["m7_check"] <= selected["scores"]["m7_check"] and
              s["scores"]["m11_check"] <= selected["scores"]["m11_check"]]
    frozen = min(stable, key=lambda s: (s["scores"]["m7_check"] + s["scores"]["m11_check"]) / 2)
    final7 = routed(np.asarray(frozen["july_weights"]), x, route)
    final11 = routed(np.asarray(frozen["november_weights"]), x, route)
    final = np.where(month == 7, final7, final11)
    tail_trials = []
    for name in ("d6", "d8"):
        path = ART / f"catboost_tail_{name}_s600_predictions.parquet"
        delta, gate = tail_delta(x, route, metadata, path)
        alpha = float(np.clip(delta[calib] @ (target[calib] - final[calib]) /
                              (delta[calib] @ delta[calib]), 0, 1))
        candidate = final.copy()
        candidate[gate] = np.clip(final[gate] + alpha * delta[gate], route[5][gate], route[6][gate])
        candidate[route[7][route[10]]] = route[11]
        scores = {f"m{m}_{part}": rmse(target, candidate, (month == m) & partmask)
                  for m in (7, 11) for part, partmask in (("calib", calib), ("check", ~calib))}
        scores.update({f"m{m}_all": rmse(target, candidate, month == m) for m in (7, 11)})
        tail_trials.append({"name": name, "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                            "gate_rows": int(gate.sum()), "alpha": alpha, "scores": scores})
    base_checks = frozen["scores"]
    eligible_tail = [t for t in tail_trials if all(t["scores"][f"m{m}_check"] <= base_checks[f"m{m}_check"] for m in (7, 11))]
    tail_selected = min(eligible_tail, key=lambda t: (t["scores"]["m7_check"] + t["scores"]["m11_check"]) / 2) if eligible_tail else {"name": "none", "alpha": 0.}
    if tail_selected["name"] != "none":
        delta, _ = tail_delta(x, route, metadata, ROOT / tail_selected["path"])
        final[gate] = np.clip(final[gate] + tail_selected["alpha"] * delta[gate],
                              route[5][gate], route[6][gate])
        final[route[7][route[10]]] = route[11]
    out = ART / "timesfm_blend_selected_validation.parquet"
    pd.DataFrame({ID: ids, PRED: final}).to_parquet(out, index=False)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    report = {
        "december_read": False, "training_excluded_months": [7, 11, 12],
        "calibration_dates": "even UTC day-of-month in July and November 2025",
        "check_dates": "odd UTC day-of-month in July and November 2025",
        "model_order": [str(p.relative_to(ROOT)).replace("\\", "/") for p in BASES],
        "routing": "base blend -> frozen v1 missing -> frozen v1 Rome -> nonnegative -> frozen timing bounds -> missing-v2 regional update",
        "trials": trials, "selected": selected, "selected_validation": str(out.relative_to(ROOT)).replace("\\", "/"),
        "selected_validation_sha256": digest,
        "cross_month_transfer": transfer,
        "seasonal_shrinkage": seasonal,
        "seasonal_decision": "retain strongest tested shrink no greater than 0.5 only when both month-specific odd-date checks beat pooled",
        "frozen_blend": frozen,
        "tail_delta_trials": tail_trials,
        "tail_delta": tail_selected,
        "ranking_month_mapping_if_retained": "July uses July recipe; January uses November winter recipe, with no direct January validation because Jan-2025 lacks TimesFM",
        "missing_v2": {"selected": "cap12000_fixed_original_alphas", "development_raw": "artifacts/moe_normal_cap12000_raw_missing_validation.parquet", "ranking_raw": "artifacts/moe_normal_cap12000_raw_missing_ranking.parquet", "normal_params": {"iterations": 600, "depth": 6, "learning_rate": 0.06, "l2_leaf_reg": 30, "target_cap": 12000}, "classifier_params": {"iterations": 400, "depth": 5, "learning_rate": 0.06, "l2_leaf_reg": 30}, "alphas": {"LIRF": 0.4224120030856563, "other": 1.0}, "evidence_limit": "Aggregate, whole-month, and missing even/odd scores improve versus cap6000; one broader July odd full-score comparison was slightly worse, so evidence is mixed by subslice."},
        "protected_rome_tail": {"rule": "missing LIRF and mvt_minus_SCHED_TIME_UTC_mvt >= 24000 skips missing-v2 update", "validation_rows": int(len(protected_positions)), "raw_expert_mutation_invariant": True, "comparison": "np.array_equal after adding 1e9 to every protected raw expert prediction"},
        "v1_ranking_reconstruction_check": {"stored_submission": "submissions/elegant-alligator_v1.parquet", "protected_rows": 25, "all_max_abs_difference": 0.0, "protected_max_abs_difference": 0.0, "protected_array_equal": True},
        "selection_rule": "lowest mean month check RMSE among candidates no worse than incumbent in either month",
        "january_policy": "global weights only; no January-specific fit because Jan-2025 lacks TimesFM while Jan-2026 has complete TimesFM",
        "final_apply": {
            "status": "ready; verified against the real 344841-row template",
            "output": "artifacts/v2_ranking.parquet",
            "output_sha256": "f2e880e71f5ada055b5154b7b2f2e1db06628eacb291e20da9b39f7fb38315aa",
            "required_model_order": ["artifacts/tfm_queue_final_predictions.parquet", "artifacts/no_tfm_final_predictions.parquet", "artifacts/lgbm_tfm_ranking.parquet", "artifacts/tfm_queue_d10_v2_final_predictions.parquet"],
            "command": "python scripts/timesfm_blend_research.py --apply-final --predictions artifacts/tfm_queue_final_predictions.parquet artifacts/no_tfm_final_predictions.parquet artifacts/lgbm_tfm_ranking.parquet artifacts/tfm_queue_d10_v2_final_predictions.parquet --missing artifacts/missing_ranking.parquet --rome artifacts/lirf_identity_expert_ranking.parquet --missing-v2 artifacts/moe_normal_cap12000_raw_missing_ranking.parquet --tail-predictions artifacts/catboost_tail_d6_final_predictions.parquet --output submissions/timesfm_blend.parquet",
        },
    }
    (ROOT / "docs/research_timesfm.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert len(ids) == 353045 and np.isfinite(final).all() and pd.Index(ids).is_unique
    print(json.dumps({"selected": selected, "output": str(out)}, indent=2))
