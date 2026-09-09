"""Fit validation-only ensemble weights and specialist routing, then freeze them.

Fit: --validation model_a.parquet model_b.parquet --config artifacts/mix.json
     [--ranking final_a.parquet final_b.parquet --output submission.parquet]
Apply: --apply-config artifacts/mix.json --predictions final_a.parquet final_b.parquet
Audit: --apply-config artifacts/mix.json --predictions audit_a.parquet audit_b.parquet --audit
"""
# SPDX-License-Identifier: GPL-3.0-only
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
ID = "MVT_ID_mvt"
TIME = "MVT_TIME_UTC_mvt"
TARGET = "TAXITIME_SEC_mvt"
PRED = "prediction"
BOUND_COLUMNS = [ID, "lower_bound", "upper_bound"]
TEMPLATE = ROOT / "data/raw/prc-2026-datasets/submitting.parquet"


def require_ids(frame, name):
    if frame.empty or frame[ID].isna().any() or not frame[ID].is_unique:
        raise ValueError(f"{name}: IDs must be nonempty, non-null and unique")


def align(frame, ids, name):
    """Align explicitly by ID, never by the prediction file's row order."""
    require_ids(frame, name)
    ids = pd.Index(ids, name=ID)
    found = pd.Index(frame[ID])
    if not ids.is_unique or ids.hasnans:
        raise ValueError("Expected IDs must be unique and non-null")
    if len(ids.difference(found)) or len(found.difference(ids)):
        raise ValueError(f"{name}: prediction IDs differ from the expected IDs")
    return frame.set_index(ID).loc[ids].reset_index()


def read_predictions(path):
    frame = pd.read_parquet(path, columns=[ID, PRED])
    require_ids(frame, str(path))
    if not np.isfinite(frame[PRED].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{path}: predictions must be finite numbers")
    return frame


def matrix(paths, ids=None):
    frames = [read_predictions(path) for path in paths]
    if ids is None:
        ids = frames[0][ID].to_numpy()
    values = [align(frame, ids, str(path))[PRED].to_numpy(dtype=np.float64)
              for path, frame in zip(paths, frames)]
    return np.asarray(ids), np.column_stack(values)


def simplex_weights(predictions, target):
    if predictions.shape[1] == 1:
        return np.ones(1)
    # The sum-to-one constraint makes residual covariance a small quadratic
    # objective. Its analytic gradient avoids finite-difference model reruns.
    residuals = (predictions - target[:, None]) / max(float(np.std(target)), 1.0)
    gram = residuals.T @ residuals / len(target)
    count = predictions.shape[1]
    result = minimize(lambda w: float(w @ gram @ w), np.full(count, 1 / count),
        jac=lambda w: 2 * gram @ w, method="SLSQP", bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones(count)},
        options={"ftol": 1e-12, "maxiter": 1000})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Simplex optimization failed: {result.message}")
    weights = np.clip(result.x, 0, 1)
    return weights / weights.sum()


def specialist_positions(path, ids):
    frame = read_predictions(path)
    positions = pd.Index(ids).get_indexer(frame[ID])
    if (positions < 0).any():
        raise ValueError(f"{path}: specialist contains IDs absent from the ensemble")
    return positions, frame[PRED].to_numpy(dtype=np.float64)


def joint_weights(predictions, target, positions, specialist):
    """Jointly minimize routed MSE with simplex weights and a bounded alpha."""
    known = np.ones(len(target), dtype=bool)
    known[positions] = False
    count = predictions.shape[1]
    initial = simplex_weights(predictions[known], target[known]) if known.any() else np.full(count, 1 / count)
    scale = max(float(np.std(target)), 1.0)
    residuals = (predictions - target[:, None]) / scale
    missing = residuals[positions]
    extra = (specialist - target[positions]) / scale
    # These small sufficient statistics evaluate the exact routed objective:
    # known residual = D w; missing residual = (1-alpha) D w + alpha extra.
    g_known = residuals[known].T @ residuals[known] / len(target)
    g_missing = missing.T @ missing / len(target)
    cross = missing.T @ extra / len(target)
    square = float(extra @ extra / len(target))

    def objective(parameters):
        w, alpha = parameters[:-1], parameters[-1]
        return float(w @ g_known @ w + (1-alpha)**2 * (w @ g_missing @ w)
                     + 2*alpha*(1-alpha) * (w @ cross) + alpha**2 * square)

    def gradient(parameters):
        w, alpha = parameters[:-1], parameters[-1]
        dw = 2*g_known @ w + 2*(1-alpha)**2 * (g_missing @ w) + 2*alpha*(1-alpha)*cross
        da = -2*(1-alpha)*(w @ g_missing @ w) + 2*(1-2*alpha)*(w @ cross) + 2*alpha*square
        return np.r_[dw, da]

    candidates = []
    for alpha in (0., .5, 1.):
        result = minimize(objective, np.r_[initial, alpha], jac=gradient,
            method="SLSQP", bounds=[(0, 1)] * (count + 1),
            constraints={"type": "eq", "fun": lambda p: p[:-1].sum() - 1,
                         "jac": lambda p: np.r_[np.ones(count), 0.]},
            options={"ftol": 1e-12, "maxiter": 1000})
        if result.success and np.isfinite(result.x).all():
            w = np.clip(result.x[:-1], 0, 1)
            candidates.append(np.r_[w / w.sum(), np.clip(result.x[-1], 0, 1)])
    if not candidates:
        raise RuntimeError("Joint ensemble optimization failed at all three alpha starts")
    best = min(candidates, key=objective)
    return best[:-1], float(best[-1])


def blend_alpha(base, specialist, target):
    delta = specialist - base
    denominator = float(delta @ delta)
    return float(np.clip(delta @ (target - base) / denominator, 0, 1)) if denominator else 0.0


def apply_specialist(base, ids, path, alpha):
    result = base.copy()
    positions, specialist = specialist_positions(path, ids)
    result[positions] += alpha * (specialist - result[positions])
    return result


def blend_rome(base, positions, expert, is_missing, betas):
    result = base.copy()
    beta = np.where(is_missing, betas["missing"], betas["known"])
    result[positions] += beta * (expert - result[positions])
    return result


def project_bounds(predictions, ids, bounds):
    """Bounds may contain extra IDs; every requested ID needs exactly one row."""
    require_ids(bounds, "timing bounds")
    positions = pd.Index(bounds[ID]).get_indexer(ids)
    if (positions < 0).any():
        raise ValueError("Timing bounds are missing prediction IDs")
    lower = bounds.lower_bound.to_numpy(dtype=np.float64)
    upper = bounds.upper_bound.to_numpy(dtype=np.float64)
    if np.isinf(lower).any() or np.isinf(upper).any():
        raise ValueError("Timing bounds must be finite or NaN for unbounded")
    lower = np.where(np.isnan(lower), -np.inf, lower)
    upper = np.where(np.isnan(upper), np.inf, upper)
    if (lower > upper).any():
        raise ValueError("Timing lower bounds must not exceed upper bounds")
    return np.clip(predictions, lower[positions], upper[positions])


def score(target, predictions):
    return float(np.sqrt(np.mean((target - predictions) ** 2)))


def metrics(labels, predictions):
    target = labels[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Scored targets must be finite")
    month = labels[TIME].dt.strftime("%Y-%m").to_numpy()
    return {"pooled_rmse": score(target, predictions), "month_rmse": {
        value: score(target[month == value], predictions[month == value])
        for value in sorted(set(month))}}


def validation_labels(path, ids):
    # Predicate filtering keeps December labels out of ensemble fitting.
    filters = [[(TIME, ">=", pd.Timestamp(f"2025-{m:02}-01", tz="UTC")),
                (TIME, "<", pd.Timestamp(f"2025-{m + 1:02}-01", tz="UTC"))]
               for m in (7, 11)]
    labels = pd.read_parquet(path, columns=[ID, TIME, TARGET], filters=filters)
    labels = labels.loc[labels[TARGET].notna()]
    labels = align(labels, ids, "July/November validation labels")
    if not np.isfinite(labels[TARGET].to_numpy(dtype=np.float64)).all():
        raise ValueError("Validation targets must be finite")
    return labels


def save_config(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    weights = np.asarray(config["weights"], dtype=np.float64)
    alpha = float(config["specialist_alpha"])
    if config.get("version") != 1 or not len(weights) or not np.isfinite(weights).all():
        raise ValueError("Invalid ensemble configuration")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1) or not 0 <= alpha <= 1:
        raise ValueError("Configuration weights must form a simplex and alpha must be in [0, 1]")
    if config.get("bounds_enabled", False) and not config.get("bounds_path"):
        raise ValueError("Selected timing bounds require a saved bounds_path")
    if "rome_betas" in config:
        betas = config["rome_betas"]
        if not isinstance(betas, dict) or set(betas) != {"known", "missing"} or any(
                not np.isfinite(value) or not 0 <= value <= 1 for value in betas.values()):
            raise ValueError("Rome betas must contain finite known/missing values in [0, 1]")
    return config, weights


def frozen_predictions(config, paths, specialist=None, ids=None, bounds=None, rome=None):
    weights = np.asarray(config["weights"], dtype=np.float64)
    if len(paths) != len(weights):
        raise ValueError("Provide one prediction file per frozen weight, in the original model order")
    if bool(specialist) != bool(config["specialist_enabled"]):
        raise ValueError("Specialist file presence must match the frozen configuration")
    betas = config.get("rome_betas", {"known": 0., "missing": 0.})
    if any(value > 0 for value in betas.values()) and not rome:
        raise ValueError("The frozen nonzero Rome betas require a matching --rome expert file")
    if rome and "rome_betas" not in config:
        raise ValueError("The frozen configuration did not fit a Rome expert")
    ids, predictions = matrix(paths, ids)
    result = predictions @ weights
    if specialist:
        result = apply_specialist(result, ids, specialist, config["specialist_alpha"])
    if rome:
        positions, expert = specialist_positions(rome, ids)
        missing_positions = specialist_positions(specialist, ids)[0] if specialist else np.empty(0, dtype=int)
        result = blend_rome(result, positions, expert, np.isin(positions, missing_positions), betas)
    if config.get("clip_nonnegative", False):
        result = np.maximum(result, 0)
    if config.get("bounds_enabled", False):
        path = Path(bounds) if bounds else ROOT / config["bounds_path"]
        result = project_bounds(result, ids, pd.read_parquet(path, columns=BOUND_COLUMNS))
    if not np.isfinite(result).all():
        raise ValueError("Ensemble predictions must be finite")
    return ids, result


def write_submission(config, paths, specialist, output, bounds=None, rome=None):
    template = pd.read_parquet(TEMPLATE)
    require_ids(template, "submission template")
    ids, predictions = frozen_predictions(config, paths, specialist, template[ID].to_numpy(), bounds=bounds, rome=rome)
    template[TARGET] = predictions.astype(np.float64)
    assert np.array_equal(template[ID].to_numpy(), ids)
    output.parent.mkdir(parents=True, exist_ok=True)
    template.to_parquet(output, index=False)
    print(f"Wrote {output}: {len(template)} finite float64 predictions in template order")


def self_check():
    ids = np.array([30., 10., 20., 40.])
    a = np.array([0., 5., 1., 9.])
    b = np.array([6., 1., 7., 2.])
    shuffled = pd.DataFrame({ID: ids[[2, 0, 3, 1]], PRED: b[[2, 0, 3, 1]]})
    restored = align(shuffled, ids, "synthetic")[PRED].to_numpy()
    np.testing.assert_array_equal(restored, b)
    target = .25 * a + .75 * b
    weights = simplex_weights(np.column_stack([a, restored]), target)
    np.testing.assert_allclose(weights, [.25, .75], atol=1e-7)
    np.testing.assert_allclose(blend_alpha(a, b, a + .4 * (b - a)), .4, atol=1e-12)
    positions, specialist = np.array([1, 3]), np.array([10., -4.])
    target[positions] = .6 * target[positions] + .4 * specialist
    weights, alpha = joint_weights(np.column_stack([a, b]), target, positions, specialist)
    np.testing.assert_allclose(weights, [.25, .75], atol=1e-6)
    np.testing.assert_allclose(alpha, .4, atol=1e-6)
    rome = pd.DataFrame({ID: [20., 30.], PRED: [7., 6.]})
    positions = pd.Index(ids).get_indexer(rome[ID])
    is_missing = rome[ID].isin([30.]).to_numpy()
    expected = np.array([4.5, 5., 2.5, 9.])
    betas = {name: blend_alpha(a[positions[mask]], rome[PRED].to_numpy()[mask], expected[positions[mask]])
             for name, mask in [("known", ~is_missing), ("missing", is_missing)]}
    np.testing.assert_allclose([betas["known"], betas["missing"]], [.25, .75])
    np.testing.assert_allclose(blend_rome(a, positions, rome[PRED].to_numpy(), is_missing, betas), expected)
    bounds = pd.DataFrame({ID: [99., 20., 30., 10., 40.],
        "lower_bound": [0., np.nan, 0., 4., np.nan],
        "upper_bound": [1., 8., 3., np.nan, np.nan]})
    np.testing.assert_array_equal(project_bounds(np.array([-2., 5., 9., 2.]), ids, bounds), [0., 5., 8., 2.])
    bounds.loc[bounds[ID].eq(20), "lower_bound"] = 9
    try:
        project_bounds(a, ids, bounds)
    except ValueError:
        pass
    else:
        raise AssertionError("Inverted timing bounds were silently accepted")
    try:
        align(shuffled.iloc[:-1], ids, "missing synthetic ID")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing IDs were silently accepted")
    print("PASS: ID alignment, synthetic optimizers, partial shuffled Rome betas, and timing bounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", nargs="+", action="extend", type=Path)
    parser.add_argument("--ranking", nargs="+", action="extend", type=Path)
    parser.add_argument("--missing-validation", type=Path)
    parser.add_argument("--missing-ranking", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "artifacts/ensemble.json")
    parser.add_argument("--apply-config", type=Path)
    parser.add_argument("--predictions", nargs="+", action="extend", type=Path)
    parser.add_argument("--missing", type=Path)
    parser.add_argument("--rome-validation", type=Path)
    parser.add_argument("--rome-ranking", type=Path)
    parser.add_argument("--rome", type=Path, help="Matching partial Rome expert for frozen apply/audit")
    parser.add_argument("--rows", type=Path, default=ROOT / "artifacts/rows.parquet")
    parser.add_argument("--bounds", type=Path,
        help="Candidate timing bounds in fit mode; override the selected bounds file in apply mode")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    args.output = args.output or ROOT / "artifacts" / (
        "ensemble_audit_predictions.parquet" if args.audit else "ensemble_submission.parquet")
    if args.apply_config:
        if (not args.predictions or args.validation or args.ranking or args.missing_validation
                or args.missing_ranking or args.rome_validation or args.rome_ranking):
            parser.error("Apply mode requires --predictions; use --missing and --rome for its expert files")
        config, _ = load_config(args.apply_config)
        if args.audit:
            ids, predictions = frozen_predictions(config, args.predictions, args.missing, bounds=args.bounds, rome=args.rome)
            labels = pd.read_parquet(args.rows, columns=[ID, TIME, TARGET], filters=[
                (TIME, ">=", pd.Timestamp("2025-12-01", tz="UTC")),
                (TIME, "<", pd.Timestamp("2026-01-01", tz="UTC"))])
            labels = labels.loc[labels[TARGET].notna()]
            labels = align(labels, ids, "audit labels")
            if not (labels[TIME].dt.year.eq(2025) & labels[TIME].dt.month.eq(12)).all():
                raise ValueError("--audit accepts December 2025 IDs only")
            print(json.dumps({"audit": metrics(labels, predictions)}, indent=2, allow_nan=False))
            output = labels.copy()
            output[PRED] = predictions
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(args.output, index=False)
            print(f"Wrote frozen December audit predictions to {args.output}")
        else:
            write_submission(config, args.predictions, args.missing, args.output, args.bounds, args.rome)
        return
    if not args.validation or args.predictions or args.missing or args.rome or args.audit:
        parser.error("Fitting requires --validation; --predictions, --missing, --rome and --audit require --apply-config")
    if args.ranking and len(args.ranking) != len(args.validation):
        parser.error("--ranking files must correspond one-to-one to --validation files in the same model order")
    if args.missing_ranking and not args.ranking:
        parser.error("--missing-ranking requires --ranking")
    if args.ranking and bool(args.missing_validation) != bool(args.missing_ranking):
        parser.error("Provide both validation and ranking specialist files when building a submission")
    if args.rome_validation and not args.missing_validation:
        parser.error("--rome-validation requires --missing-validation to define known/missing membership")
    if args.rome_ranking and (not args.ranking or not args.rome_validation):
        parser.error("--rome-ranking requires --ranking and --rome-validation")
    ids, predictions = matrix(args.validation)
    labels = validation_labels(args.rows, ids)
    target = labels[TARGET].to_numpy(dtype=np.float64)
    alpha = 0.0
    if args.missing_validation:
        positions, specialist = specialist_positions(args.missing_validation, ids)
        weights, alpha = joint_weights(predictions, target, positions, specialist)
    else:
        weights = simplex_weights(predictions, target)
    combined = predictions @ weights
    config = {"version": 1, "models": [str(path.resolve()) for path in args.validation],
        "weights": weights.tolist(), "specialist_enabled": bool(args.missing_validation),
        "specialist_alpha": alpha, "validation_rows": len(ids),
        "base_validation": metrics(labels, combined),
        "model_validation": [metrics(labels, predictions[:, i]) for i in range(len(weights))]}
    if args.missing_validation:
        config["specialist_validation_rows"] = len(positions)
        combined[positions] += alpha * (specialist - combined[positions])
    if args.rome_validation:
        rome_positions, expert = specialist_positions(args.rome_validation, ids)
        is_missing = np.isin(rome_positions, positions)
        betas = {name: blend_alpha(combined[rome_positions[mask]], expert[mask], target[rome_positions[mask]])
                 for name, mask in [("known", ~is_missing), ("missing", is_missing)]}
        config["rome_betas"] = betas
        config["rome_validation_file"] = str(args.rome_validation.resolve())
        config["rome_validation_rows"] = {"known": int((~is_missing).sum()), "missing": int(is_missing.sum())}
        config["rome_grouping"] = "Membership in the missing-specialist prediction IDs"
        combined = blend_rome(combined, rome_positions, expert, is_missing, betas)
    config["fit_sse"] = float(np.sum((combined - target) ** 2))
    clipped = np.maximum(combined, 0)
    config["clip_nonnegative"] = bool(np.sum((clipped - target) ** 2) < np.sum((combined - target) ** 2))
    if config["clip_nonnegative"]:
        combined = clipped
    config["bounds_enabled"] = False
    if args.bounds:
        bounded = project_bounds(combined, ids, pd.read_parquet(args.bounds, columns=BOUND_COLUMNS))
        if np.sum((bounded - target) ** 2) < np.sum((combined - target) ** 2):
            config["bounds_enabled"] = True
            path = args.bounds.resolve()
            try:
                path = path.relative_to(ROOT)
            except ValueError:
                pass
            config["bounds_path"] = path.as_posix()
            config["bounds_changed_rows"] = int(np.count_nonzero(bounded != combined))
            combined = bounded
    config["validation_sse"] = float(np.sum((combined - target) ** 2))
    config["validation"] = metrics(labels, combined)
    save_config(config, args.config)
    print(json.dumps(config, indent=2, allow_nan=False))
    print(f"Saved frozen weights to {args.config}")
    if args.ranking:
        write_submission(config, args.ranking, args.missing_ranking, args.output, args.bounds, args.rome_ranking)


if __name__ == "__main__":
    main()
