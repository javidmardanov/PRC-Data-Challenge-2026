"""Independently reconstruct v2 ranking arithmetic and verify the submission."""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_solution import check_submission
from ensemble import (ROOT, ID, TIME, TARGET, PRED, align, blend_rome,
                      frozen_predictions, project_bounds, read_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', nargs='?', type=Path)
    parser.add_argument('--snapshot-v1', action='store_true')
    args = parser.parse_args()
    if args.snapshot_v1:
        names = ['artifacts/v1_ensemble.json', 'artifacts/tfm_queue_d8.cbm',
            'artifacts/no_tfm_full_d8.cbm', 'artifacts/lgbm_tfm.txt',
            'submissions/elegant-alligator_v1.parquet']
        hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}
        (ROOT/'artifacts/research_v1_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
        print('Recorded v1 artifact hashes before v2 work')
        return
    if args.submission is None:
        parser.error('provide a submission or --snapshot-v1')
    check_submission(args.submission)
    actual = pd.read_parquet(args.submission)
    ids = actual[ID].to_numpy()
    v2 = json.loads((ROOT/'docs/v2_ensemble.json').read_text())
    v1 = json.loads((ROOT/'docs/v1_ensemble.json').read_text())
    paths = [ROOT/'artifacts'/name for name in ['tfm_queue_final_predictions.parquet',
        'no_tfm_final_predictions.parquet', 'lgbm_tfm_ranking.parquet',
        'tfm_queue_d10_v2_final_predictions.parquet']]
    missing_path = ROOT/'artifacts/missing_ranking.parquet'
    rome_path = ROOT/'artifacts/lirf_identity_expert_ranking.parquet'
    metadata = pd.read_parquet(ROOT/'artifacts/rows.parquet', columns=[ID, TIME, 'ADEP_mvt'])
    features = pd.read_parquet(ROOT/'artifacts/features.parquet', columns=[
        'mvt_minus_AOBT_3_flt', 'mvt_minus_EOBT_1_flt', 'mvt_minus_SCHED_TIME_UTC_mvt'])
    if len(metadata) != len(features):
        raise ValueError('Feature and metadata row counts differ')
    metadata = align(metadata.join(features).loc[metadata[ID].isin(ids)], ids, 'ranking metadata')
    season = {}
    for name, key in [('summer', 'july_weights'), ('winter', 'november_weights')]:
        config = {**v1, 'weights': v2['frozen_blend'][key]}
        _, season[name] = frozen_predictions(config, paths, missing_path, ids, rome=rome_path)
    result = np.where(metadata[TIME].dt.month.eq(7), season['summer'], season['winter'])
    raw = read_predictions(ROOT/v2['missing_expert_recipe']['ranking_raw'])
    pos = pd.Index(ids).get_indexer(raw[ID])
    old_missing = read_predictions(missing_path)
    if (pos < 0).any() or set(raw[ID]) != set(old_missing[ID]):
        raise ValueError('Missing expert IDs do not match the expected subset')
    rome_missing = metadata.iloc[pos].ADEP_mvt.eq('LIRF').to_numpy()
    protected = rome_missing & metadata.iloc[pos].mvt_minus_SCHED_TIME_UTC_mvt.ge(24000).to_numpy()
    alpha = np.where(rome_missing, v2['missing_expert_recipe']['alphas']['Rome'], 1.)
    result[pos[~protected]] += alpha[~protected]*(raw[PRED].to_numpy()[~protected]-result[pos[~protected]])

    _, original = frozen_predictions(v1, paths[:3], missing_path, ids, rome=rome_path)
    aobt = metadata.mvt_minus_AOBT_3_flt.to_numpy()
    gate = (aobt > -100000) & ((aobt < 300) | (aobt > 2200) | (metadata.mvt_minus_EOBT_1_flt.to_numpy() > 3600))
    base = np.column_stack([align(read_predictions(p), ids, str(p))[PRED] for p in paths[:3]])
    tail_path = ROOT/'artifacts/catboost_tail_d6_final_predictions.parquet'
    tail = align(read_predictions(tail_path), ids, 'tail model')[PRED].to_numpy()
    base[gate, 0] = tail[gate]
    hybrid = base @ np.asarray(v1['weights'])
    old_pos = pd.Index(ids).get_indexer(old_missing[ID])
    hybrid[old_pos] += v1['specialist_alpha']*(old_missing[PRED].to_numpy()-hybrid[old_pos])
    rome = read_predictions(rome_path)
    rpos = pd.Index(ids).get_indexer(rome[ID])
    hybrid = blend_rome(hybrid, rpos, rome[PRED].to_numpy(), np.isin(rpos, old_pos), v1['rome_betas'])
    bounds = pd.read_parquet(ROOT/v1['bounds_path'])
    hybrid = project_bounds(np.maximum(hybrid, 0), ids, bounds)
    result[gate] += v2['tail_delta']['alpha']*(hybrid-original)[gate]
    result[gate] = project_bounds(result, ids, bounds)[gate]
    stored = align(pd.read_parquet(ROOT/'submissions/elegant-alligator_v1.parquet'), ids, 'v1 submission')[TARGET].to_numpy()
    result[pos[protected]] = stored[pos[protected]]
    np.testing.assert_allclose(result, actual[TARGET], rtol=0, atol=1e-8)
    np.testing.assert_array_equal(actual[TARGET].to_numpy()[pos[protected]], stored[pos[protected]])
    if int(protected.sum()) != 25:
        raise ValueError('Expected 25 protected ranking rows')
    for name, expected in json.loads((ROOT/'artifacts/research_v1_hashes.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen v1 artifact changed: {name}')
    report = {'rows': len(ids), 'protected_rows': int(protected.sum()), 'tail_gate_rows': int(gate.sum()),
        'independent_max_abs_difference': float(np.max(np.abs(result-actual[TARGET].to_numpy()))),
        'sha256': hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        'v1_hashes_unchanged': True, 'december_scored': False}
    (ROOT/'docs/review_v2_submission.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
