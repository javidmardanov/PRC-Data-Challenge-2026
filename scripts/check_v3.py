"""Independently reconstruct the v3 tail correction and check frozen artifacts."""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_solution import check_submission
from ensemble import ROOT, ID, TIME, TARGET, PRED, align, read_predictions, project_bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path, nargs='?')
    parser.add_argument('--snapshot-v2', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT/'docs/research_catboost_v3.json')
    parser.add_argument('--missing-arrival', type=Path)
    parser.add_argument('--arrival-predictions', type=Path)
    parser.add_argument('--arrival-config', type=Path, default=ROOT/'docs/research_timesfm_arrival_v3.json')
    args = parser.parse_args()
    if args.snapshot_v2:
        paths = ['submissions/elegant-alligator_v2.parquet', 'artifacts/v2_validation.parquet',
                 'docs/v2_ensemble.json', 'artifacts/tfm_queue_d10_v2_final.cbm',
                 'artifacts/catboost_tail_d6_final.cbm', 'artifacts/moe_normal_cap12000_final.cbm',
                 'artifacts/moe_missing_v2_identity_final.cbm']
        hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        snapshot = ROOT/'artifacts/research_v2_hashes.json'
        if snapshot.exists() and json.loads(snapshot.read_text()) != hashes:
            raise ValueError('Existing frozen v2 snapshot differs; refusing to replace it')
        if not snapshot.exists():
            snapshot.write_text(json.dumps(hashes, indent=2)+'\n')
        print('Frozen v2 hashes recorded or verified')
        return
    if args.submission is None:
        parser.error('provide a submission or --snapshot-v2')
    check_submission(args.submission)
    actual = pd.read_parquet(args.submission)
    ids = actual[ID].to_numpy()
    v1 = json.loads((ROOT/'docs/v1_ensemble.json').read_text())
    v2 = json.loads((ROOT/'docs/v2_ensemble.json').read_text())
    config = json.loads(args.config.read_text())
    paths = ['tfm_queue_final_predictions.parquet', 'no_tfm_final_predictions.parquet',
             'lgbm_tfm_ranking.parquet', 'tfm_queue_d10_v2_final_predictions.parquet']
    matrix = np.column_stack([align(read_predictions(ROOT/'artifacts'/p), ids, p)[PRED] for p in paths])
    meta = pd.read_parquet(ROOT/'artifacts/rows.parquet', columns=[ID, TIME, 'ADEP_mvt'])
    x = pd.read_parquet(ROOT/'artifacts/features.parquet', columns=[
        'mvt_minus_AOBT_3_flt', 'mvt_minus_EOBT_1_flt', 'mvt_minus_SCHED_TIME_UTC_mvt'])
    if len(meta) != len(x):
        raise ValueError('Feature/metadata length mismatch')
    meta = align(meta.join(x).loc[meta[ID].isin(ids)], ids, 'metadata')
    known = meta.mvt_minus_AOBT_3_flt.gt(-100000).to_numpy()
    gate = known & (meta.mvt_minus_AOBT_3_flt.lt(300) | meta.mvt_minus_AOBT_3_flt.gt(2200)
                    | meta.mvt_minus_EOBT_1_flt.gt(3600)).to_numpy()
    weights = np.where(meta[TIME].dt.month.eq(7).to_numpy()[:, None],
                       v2['frozen_blend']['july_weights'], v2['frozen_blend']['november_weights'])
    old = (matrix * weights).sum(axis=1)
    raw = align(read_predictions(ROOT/'artifacts/catboost_tail_d8_full_v3_final_predictions.parquet'), ids, 'v3 tail')[PRED].to_numpy()
    new = old + weights[:, 0] * (raw - matrix[:, 0])
    rome = read_predictions(ROOT/'artifacts/lirf_identity_expert_ranking.parquet')
    rpos = pd.Index(ids).get_indexer(rome[ID])
    if (rpos < 0).any():
        raise ValueError('Rome expert contains unexpected IDs')
    selected = known[rpos]
    rpos, expert = rpos[selected], rome[PRED].to_numpy()[selected]
    # Only known rows can enter the correction gate, so missing routing has no
    # bearing on this independent calculation.
    for values in [old, new]:
        values[rpos] += v1['rome_betas']['known'] * (expert-values[rpos])
    bounds = pd.read_parquet(ROOT/v1['bounds_path'])
    old = project_bounds(np.maximum(old, 0), ids, bounds)
    new = project_bounds(np.maximum(new, 0), ids, bounds)
    baseline = align(pd.read_parquet(ROOT/'submissions/elegant-alligator_v2.parquet'), ids, 'frozen v2')[TARGET].to_numpy()
    expected = baseline.copy()
    expected[gate] += config['gated_alpha_even'] * (new-old)[gate]
    expected[gate] = project_bounds(expected, ids, bounds)[gate]
    permitted = gate.copy()
    missing_rows = 0
    if args.missing_arrival:
        previous = read_predictions(ROOT/'artifacts/moe_normal_cap12000_raw_missing_ranking.parquet')
        replacement = align(read_predictions(args.missing_arrival), previous[ID], 'missing arrival')
        pos = pd.Index(ids).get_indexer(previous[ID])
        if (pos < 0).any() or known[pos].any():
            raise ValueError('Missing correction contains an unexpected or known ID')
        eligible = ~(meta.iloc[pos].ADEP_mvt.eq('LIRF').to_numpy()
                     & meta.iloc[pos].mvt_minus_SCHED_TIME_UTC_mvt.ge(24000).to_numpy())
        scale = np.where(meta.iloc[pos].ADEP_mvt.eq('LIRF'), .4224120030856563, 1.)
        expected[pos[eligible]] += scale[eligible] * (replacement[PRED]-previous[PRED]).to_numpy()[eligible]
        permitted[pos[eligible]] = True
        missing_rows = int(eligible.sum())
    arrival_rows = 0
    if args.arrival_predictions:
        recipe = json.loads(args.arrival_config.read_text())['selected']
        if recipe['name'] != 'known_nonrome':
            raise ValueError('Expected the frozen known non-Rome arrival correction')
        agate = known & meta.ADEP_mvt.ne('LIRF').to_numpy()
        arrival = align(read_predictions(args.arrival_predictions), ids, 'arrival model')[PRED].to_numpy()
        amended = (matrix * weights).sum(axis=1) + weights[:, 0] * (arrival-matrix[:, 0])
        amended = project_bounds(np.maximum(amended, 0), ids, bounds)
        expected[agate] += recipe['alpha_even'] * (amended-old)[agate]
        expected[agate] = project_bounds(expected, ids, bounds)[agate]
        permitted |= agate
        arrival_rows = int(agate.sum())
    result = actual[TARGET].to_numpy()
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-8)
    np.testing.assert_array_equal(result[~permitted], baseline[~permitted])
    protected = ~known & meta.ADEP_mvt.eq('LIRF').to_numpy() & meta.mvt_minus_SCHED_TIME_UTC_mvt.ge(24000).to_numpy()
    if protected.sum() != 25:
        raise ValueError('Expected 25 protected missing Rome rows')
    for snapshot in ['research_v1_hashes.json', 'research_v2_hashes.json']:
        for path, expected_hash in json.loads((ROOT/'artifacts'/snapshot).read_text()).items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f'Frozen artifact changed: {path}')
    report = {'rows': len(ids), 'tail_gate_rows': int(gate.sum()), 'protected_rows': int(protected.sum()),
              'missing_correction_rows': missing_rows,
              'known_arrival_correction_rows': arrival_rows,
              'outside_correction_masks_bitwise_unchanged': True, 'v1_v2_hashes_unchanged': True,
              'independent_max_abs_difference': float(np.max(np.abs(result-expected))),
              'sha256': hashlib.sha256(args.submission.read_bytes()).hexdigest(), 'december_scored': False}
    (ROOT/'docs/review_v3_submission.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
