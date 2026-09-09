"""Apply the frozen arrival-context correction to a complete ranking baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble
import timesfm_blend_research as blend

ROOT, ART = blend.ROOT, blend.ART
ID, TIME, TARGET, PRED = ensemble.ID, ensemble.TIME, ensemble.TARGET, ensemble.PRED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--arrival-predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "docs/research_timesfm_arrival_v3.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.config.read_text())
    if frozen["selected"]["name"] != "known_nonrome":
        raise ValueError("Frozen arrival config must select known_nonrome")
    alpha = float(frozen["selected"]["alpha_even"])
    template = pd.read_parquet(ensemble.TEMPLATE)
    paths = [ART / "tfm_queue_final_predictions.parquet", ART / "no_tfm_final_predictions.parquet",
             ART / "lgbm_tfm_ranking.parquet", ART / "tfm_queue_d10_v2_final_predictions.parquet"]
    ids, matrix = ensemble.matrix(paths, template[ID].to_numpy())
    metadata = blend.metadata_for(ids)
    route = blend.make_route(ids, metadata, ART / "missing_ranking.parquet",
        ART / "lirf_identity_expert_ranking.parquet", ROOT / "data/processed/timing_bounds.parquet",
        ART / "moe_normal_cap12000_raw_missing_ranking.parquet")
    stored_v1 = ensemble.align(pd.read_parquet(ROOT / "submissions/elegant-alligator_v1.parquet", columns=[ID, TARGET]),
                               ids, "stored v1")[TARGET].to_numpy(float)
    route = (*route[:-1], stored_v1[route[7][route[10]]])
    v2 = json.loads((ROOT / "docs/v2_ensemble.json").read_text())
    month = metadata[TIME].dt.month.to_numpy()
    w7 = np.asarray(v2["frozen_blend"]["july_weights"])
    winter = np.asarray(v2["frozen_blend"]["november_weights"])
    seasonal = lambda values: np.where(month == 7, blend.routed(w7, values, route), blend.routed(winter, values, route))
    base_routed = seasonal(matrix)
    raw = ensemble.align(ensemble.read_predictions(args.arrival_predictions), ids, "arrival predictions")[PRED].to_numpy(float)
    known_nonrome = (metadata.mvt_minus_AOBT_3_flt.to_numpy(float) > -100000) & metadata.ADEP_mvt.ne("LIRF").to_numpy()
    hybrid = matrix.copy()
    hybrid[known_nonrome, 0] = raw[known_nonrome]
    delta = seasonal(hybrid) - base_routed
    baseline_frame = pd.read_parquet(args.baseline)
    value = TARGET if TARGET in baseline_frame else PRED
    baseline = ensemble.align(baseline_frame[[ID, value]], ids, "complete baseline")[value].to_numpy(float)
    result = baseline.copy()
    result[known_nonrome] = np.clip(baseline[known_nonrome] + alpha * delta[known_nonrome],
                                    route[5][known_nonrome], route[6][known_nonrome])
    if not np.array_equal(result[~known_nonrome], baseline[~known_nonrome]):
        raise AssertionError("Arrival correction changed predictions outside its frozen gate")
    if len(ids) != 344841 or not np.isfinite(result).all() or not pd.Index(ids).is_unique:
        raise AssertionError("Ranking output invariant failed")
    output = template.copy()
    output[TARGET] = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    print(f"Wrote {args.output}: {len(output)} rows, alpha={alpha}, gate={known_nonrome.sum()}")


if __name__ == "__main__":
    main()
