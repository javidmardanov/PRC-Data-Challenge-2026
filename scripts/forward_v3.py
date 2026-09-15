"""Chronological, complete V3 reconstruction for one July/November fold.

The fold month is predicted by models fitted only on earlier labelled departures.
December is rejected.  This program intentionally writes namespaced artifacts and
never replaces the frozen development or ranking artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble
import timesfm_blend_research as blend

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED
BASE_SPECS = (
    ("tfm", 6000, 8, .0750579, 4.090916, True),
    ("plain", 5000, 8, .06, 10., False),
    ("d10", 4499, 10, .05, 30., True),
)


def run_catboost(prefix: str, month: str, device: str, threads: int) -> list[Path]:
    paths = []
    for label, trees, depth, rate, l2, enriched in BASE_SPECS:
        name = f"{prefix}_{label}"
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--name", name,
               "--rolling-month", month, "--iterations", str(trees), "--depth", str(depth),
               "--rate", str(rate), "--l2", str(l2), "--threads", str(threads),
               "--device", device, "--residual", "--known-only"]
        if enriched:
            cmd += ["--tfm", "data/processed/timesfm_hourly_dev.parquet", "--weather",
                    "data/processed/weather_hourly.parquet", "--extra",
                    "data/processed/queue_features.parquet", "--interactions"]
        subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
        paths.append(ART / f"{name}_predictions.parquet")
    for label, extra, copies, trees, depth, rate, l2 in (("v2tail", "queue_features.parquet", True, 4496, 6, .06, 20),
                                 ("tail", "queue_features.parquet", True, 5995, 8, .0750579, 4.090916),
                                 ("arrival", "queue_arrival_context_v3.parquet", False, 5995, 8, .0750579, 4.090916)):
        name = f"{prefix}_{label}"
        cmd = [sys.executable, str(ROOT / "scripts/train.py"), "--name", name,
               "--rolling-month", month, "--iterations", str(trees), "--depth", str(depth),
               "--rate", str(rate), "--l2", str(l2), "--threads", str(threads),
               "--device", device, "--residual", "--known-only", "--tfm",
               "data/processed/timesfm_hourly_dev.parquet", "--weather",
               "data/processed/weather_hourly.parquet", "--extra", f"data/processed/{extra}",
               "--interactions"]
        if copies:
            cmd.append("--tail-copies")
        subprocess.run(cmd, cwd=ROOT, check=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
    return paths


def checked_frame(path: Path, ids, name: str) -> np.ndarray:
    return ensemble.align(ensemble.read_predictions(path), ids, name)[PRED].to_numpy(float)


def reconstruct(month: str, prefix: str, base_paths: list[Path], output: Path) -> dict:
    """Apply frozen V3 routing to chronological raw predictions.

    Fold-specific specialist files use ``PREFIX_missing``, ``PREFIX_rome``,
    ``PREFIX_missing_v2`` and ``PREFIX_missing_arrival``.  They are deliberately
    explicit inputs because the repository's retained specialist programs hard-code
    the original July/November tuning split and are unsafe for a forward fold.
    """
    first = pd.read_parquet(base_paths[0])
    ids = first[ID].to_numpy()
    matrix = np.column_stack([checked_frame(p, ids, str(p)) for p in base_paths])
    labels = ensemble.align(first[[ID, TIME, "ADEP_mvt", TARGET]], ids, "fold labels")
    if not labels[TIME].dt.strftime("%Y-%m").eq(month).all():
        raise ValueError("Every prediction must belong to the requested fold")
    metadata = blend.metadata_for(ids)
    specialist = {key: ART / f"{prefix}_{key}.parquet" for key in
                  ("missing", "rome", "missing_v2", "missing_arrival")}
    absent = [str(p) for p in specialist.values() if not p.exists()]
    if absent:
        raise FileNotFoundError("Missing chronological specialist predictions: " + ", ".join(absent))
    route = blend.make_route(ids, metadata, specialist["missing"], specialist["rome"],
                             ROOT / "data/processed/timing_bounds.parquet", specialist["missing_v2"])
    # In a forward fold, the protected-tail fallback is the reconstructed V1 value.
    v1 = blend.complete_v1(matrix[:, :3], route)
    route = (*route[:-1], v1[route[7][route[10]]])
    cfg2 = json.loads((ROOT / "docs/v2_ensemble.json").read_text())
    cfg3 = json.loads((ROOT / "docs/v3_ensemble.json").read_text())
    weights = np.asarray(cfg2["frozen_blend"]["july_weights" if month.endswith("-07") else "november_weights"])
    seasonal_base = blend.routed(weights, matrix, route)
    base = seasonal_base.copy()

    # V2's retained depth-6 tail correction is part of the parent baseline.
    v2delta, v2gate = blend.tail_delta(matrix, route, metadata, ART / f"{prefix}_v2tail_predictions.parquet")
    base[v2gate] = np.clip(base[v2gate] + cfg2["tail_delta"]["alpha"] * v2delta[v2gate],
                           route[5][v2gate], route[6][v2gate])
    base[route[7][route[10]]] = route[11]

    aobt = metadata.mvt_minus_AOBT_3_flt.to_numpy(float)
    eobt = metadata.mvt_minus_EOBT_1_flt.to_numpy(float)
    tail_gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    tail = checked_frame(ART / f"{prefix}_tail_predictions.parquet", ids, "tail")
    hybrid = matrix.copy(); hybrid[tail_gate, 0] = tail[tail_gate]
    result = base.copy()
    result[tail_gate] = np.clip(result[tail_gate] + cfg3["gated_alpha_even"] *
                                (blend.routed(weights, hybrid, route)[tail_gate] - seasonal_base[tail_gate]),
                                route[5][tail_gate], route[6][tail_gate])

    pos = route[7]; protected = route[10]
    new_missing = ensemble.align(ensemble.read_predictions(specialist["missing_arrival"]),
                                 ids[pos], "missing arrival")[PRED].to_numpy(float)
    apply = ~protected
    result[pos[apply]] += route[9][apply] * (new_missing[apply] - route[8][apply])
    result[pos[protected]] = base[pos[protected]]

    arrival = checked_frame(ART / f"{prefix}_arrival_predictions.parquet", ids, "arrival")
    arrival_gate = (aobt > -100000) & metadata.ADEP_mvt.ne("LIRF").to_numpy()
    hybrid = matrix.copy(); hybrid[arrival_gate, 0] = arrival[arrival_gate]
    delta = blend.routed(weights, hybrid, route) - seasonal_base
    result[arrival_gate] = np.clip(result[arrival_gate] + cfg3["selected"]["alpha_even"] * delta[arrival_gate],
                                   route[5][arrival_gate], route[6][arrival_gate])
    if not pd.Index(ids).is_unique or not np.isfinite(result).all() or len(result) != len(labels):
        raise AssertionError("Complete ID-aligned finite fold invariant failed")
    out = labels.copy(); out[PRED] = result
    output.parent.mkdir(parents=True, exist_ok=True); out.to_parquet(output, index=False)
    y = out[TARGET].to_numpy(float)
    report = {"month": month, "training_rule": f"labels strictly before {month}-01",
              "rows": len(out), "rmse": float(np.sqrt(np.mean((y-result)**2))),
              "tail_rows": int(tail_gate.sum()), "arrival_rows": int(arrival_gate.sum()),
              "protected_rome_tails": int(protected.sum()), "december_evaluated": False,
              "output": str(output)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--month", required=True, choices=("2025-07", "2025-11"))
    p.add_argument("--prefix")
    p.add_argument("--fit-bases", action="store_true")
    p.add_argument("--fit-lightgbm", action="store_true")
    p.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--check", action="store_true", help="syntax/configuration check; fits nothing")
    args = p.parse_args()
    if args.threads > (8 if args.device == "GPU" else 3):
        p.error("CPU fits allow at most 3 threads; leased GPU fits allow at most 8")
    prefix = args.prefix or "forward_v3_" + args.month
    paths = [ART / f"{prefix}_{x}_predictions.parquet" for x in ("tfm", "plain", "lgbm", "d10")]
    if args.check:
        print(json.dumps({"status": "ok", "month": args.month, "prefix": prefix,
                          "required_specialists": [f"{prefix}_{x}.parquet" for x in
                          ("missing", "rome", "missing_v2", "missing_arrival")]})); return
    if args.fit_bases:
        cb = run_catboost(prefix, args.month, args.device, args.threads)
        paths[0], paths[1], paths[3] = cb
    if args.fit_lightgbm:
        subprocess.run([sys.executable, str(ROOT / "scripts/lightgbm_model.py"),
                        "--rolling-month", args.month], cwd=ROOT, check=True,
                       env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
        shutil.copyfile(ART / f"lgbm_tfm_rolling_{args.month}.parquet", paths[2])
    if not paths[2].exists():
        raise FileNotFoundError(f"{paths[2]} (chronological 188-tree LightGBM prediction) is required")
    output = args.output or ART / f"{prefix}_complete.parquet"
    print(json.dumps(reconstruct(args.month, prefix, paths, output), indent=2))


if __name__ == "__main__":
    main()
