"""Apply the frozen v3 CatBoost tail delta to a complete v2 ranking prediction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble
import timesfm_blend_research as blend

ROOT, ART = blend.ROOT, blend.ART
ID, TARGET, PRED, TIME = ensemble.ID, ensemble.TARGET, ensemble.PRED, ensemble.TIME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--predictions", nargs=4, type=Path, required=True)
    parser.add_argument("--tail-predictions", type=Path, required=True)
    parser.add_argument("--missing-arrival", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "docs/v2_ensemble.json")
    parser.add_argument("--tail-config", type=Path, default=ROOT / "docs/research_catboost_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = pd.read_parquet(ensemble.TEMPLATE)
    ids, matrix = ensemble.matrix(args.predictions, template[ID].to_numpy())
    metadata = blend.metadata_for(ids)
    config = json.loads(args.config.read_text())
    tail_config = json.loads(args.tail_config.read_text())
    route = blend.make_route(ids, metadata, ART / "missing_ranking.parquet",
        ART / "lirf_identity_expert_ranking.parquet", ROOT / "data/processed/timing_bounds.parquet",
        ART / "moe_normal_cap12000_raw_missing_ranking.parquet")
    stored_v1 = ensemble.align(pd.read_parquet(ROOT / "submissions/elegant-alligator_v1.parquet", columns=[ID, TARGET]),
                               ids, "stored v1")[TARGET].to_numpy(float)
    route = (*route[:-1], stored_v1[route[7][route[10]]])
    month = pd.to_datetime(metadata[TIME], utc=True).dt.month.to_numpy()
    w7 = np.asarray(config["frozen_blend"]["july_weights"])
    winter = np.asarray(config["frozen_blend"]["november_weights"])

    def seasonal(values):
        return np.where(month == 7, blend.routed(w7, values, route), blend.routed(winter, values, route))

    base = seasonal(matrix)
    raw = ensemble.align(ensemble.read_predictions(args.tail_predictions), ids, str(args.tail_predictions))[PRED].to_numpy(float)
    aobt = metadata.mvt_minus_AOBT_3_flt.to_numpy(float)
    eobt = metadata.mvt_minus_EOBT_1_flt.to_numpy(float)
    gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (eobt > 3600))
    hybrid = matrix.copy()
    hybrid[gate, 0] = raw[gate]
    delta = seasonal(hybrid) - base
    baseline_frame = pd.read_parquet(args.baseline)
    value_column = TARGET if TARGET in baseline_frame else PRED
    baseline = ensemble.align(baseline_frame[[ID, value_column]], ids, "v2 baseline")[value_column].to_numpy(float)
    result = baseline.copy()
    alpha = float(tail_config["gated_alpha_even"])
    result[gate] = np.clip(result[gate] + alpha * delta[gate], route[5][gate], route[6][gate])
    authorized = gate.copy()
    if args.missing_arrival:
        expected_ids = ids[route[7]]
        arrival = ensemble.read_predictions(args.missing_arrival)
        if len(pd.Index(arrival[ID]).symmetric_difference(pd.Index(expected_ids))):
            raise ValueError("Missing-arrival predictions must exactly match the frozen missing-ID set")
        arrival_raw = arrival.set_index(ID).loc[expected_ids, PRED].to_numpy(float)
        apply = ~route[10]
        result[route[7][apply]] += route[9][apply] * (arrival_raw[apply] - route[8][apply])
        result[route[7][route[10]]] = baseline[route[7][route[10]]]
        authorized[route[7][apply]] = True
    if not np.array_equal(result[~authorized], baseline[~authorized]):
        raise AssertionError("Predictions outside the authorized correction masks changed")
    if not np.isfinite(result).all() or len(ids) != len(template) or not pd.Index(ids).is_unique:
        raise AssertionError("Output must contain one finite prediction per unique template ID")
    output = template.copy()
    output[TARGET] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}: {len(output)} rows, {int(gate.sum())} gated, "
          f"missing-arrival={'on' if args.missing_arrival else 'off'}")


if __name__ == "__main__":
    main()
