"""Score CatBoost candidates as frozen-v1 base replacements and tail gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import (ID, TARGET, TIME, align, frozen_predictions, load_config, metrics,
                      project_bounds, validation_labels)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research_catboost_scores.json")
    args = parser.parse_args()
    config, _ = load_config(ROOT / "docs/v1_ensemble.json")
    original = [ROOT / path for path in config["models"]]
    ids, base = frozen_predictions(config, original, OUT / "missing_validation.parquet",
                                   rome=OUT / "lirf_identity_expert_validation.parquet")
    labels = validation_labels(OUT / "rows.parquet", ids)
    rows = pd.read_parquet(OUT / "rows.parquet", columns=[ID])
    features = pd.read_parquet(OUT / "features.parquet",
                               columns=["mvt_minus_AOBT_3_flt", "mvt_minus_EOBT_1_flt"])
    gate_frame = rows.join(features)
    gate_frame = gate_frame.loc[gate_frame[ID].isin(ids)]
    gate_frame = align(gate_frame, ids, "tail features")
    aobt = gate_frame.mvt_minus_AOBT_3_flt.to_numpy()
    eobt = gate_frame.mvt_minus_EOBT_1_flt.to_numpy()
    gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    target = labels[TARGET].to_numpy(float)
    even = labels[TIME].dt.day.mod(2).eq(0).to_numpy()
    bounds = pd.read_parquet(ROOT / config["bounds_path"])
    report = {"baseline": metrics(labels, base), "tail_rows": int(gate.sum()), "candidates": []}
    incumbent = pd.read_parquet(original[0], columns=[ID, "prediction"])
    for path in args.candidates:
        candidate = align(pd.read_parquet(path), ids, str(path))
        hybrid = align(incumbent, ids, str(original[0]))
        hybrid.loc[gate, "prediction"] = candidate.loc[gate, "prediction"].to_numpy()
        hybrid_path = OUT / f"catboost_{path.stem}_tail_hybrid.parquet"
        hybrid.to_parquet(hybrid_path, index=False)
        _, full = frozen_predictions(config, [path, *original[1:]], OUT / "missing_validation.parquet",
                                     ids=ids, rome=OUT / "lirf_identity_expert_validation.parquet")
        _, gated = frozen_predictions(config, [hybrid_path, *original[1:]], OUT / "missing_validation.parquet",
                                      ids=ids, rome=OUT / "lirf_identity_expert_validation.parquet")
        delta = gated - base
        fit = gate & even
        denominator = float(delta[fit] @ delta[fit])
        alpha = float(np.clip(delta[fit] @ (target[fit] - base[fit]) / denominator, 0, 1)) if denominator else 0.0
        blended = project_bounds(base + alpha * delta, ids, bounds)
        blended_path = OUT / f"catboost_{path.stem}_gated_complete_validation.parquet"
        pd.DataFrame({ID: ids, "prediction": blended}).to_parquet(blended_path, index=False)
        check = gate & ~even
        report["candidates"].append({"path": str(path), "hybrid_path": str(hybrid_path),
            "full_replacement": metrics(labels, full), "gated_alpha_even_dates": alpha,
            "gated_all": metrics(labels, blended), "gated_complete_path": str(blended_path),
            "gated_odd_complete": metrics(labels.loc[~even].reset_index(drop=True), blended[~even]),
            "baseline_odd_complete": metrics(labels.loc[~even].reset_index(drop=True), base[~even]),
            "gated_even_fit_rmse": float(np.sqrt(np.mean((target[fit]-blended[fit])**2))),
            "gated_odd_check_rmse": float(np.sqrt(np.mean((target[check]-blended[check])**2))),
            "baseline_odd_check_rmse": float(np.sqrt(np.mean((target[check]-base[check])**2)))})
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
