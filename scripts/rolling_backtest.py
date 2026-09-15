"""Matched forward refits and chronological calibration; never evaluates December."""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from ensemble import ROOT, ID, TIME, TARGET, align, blend_alpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-only', action='store_true')
    args = p.parse_args()
    months = ['2025-07', '2025-10', '2025-11']
    report = {'protocol': 'Fixed 2000 trees; strictly earlier training months; blend calibrated on previous completed fold only. First fold uses existing arrival-model preference (alpha=1).',
              'limitations': ['Only one labeled year: no genuine January forward backtest is possible.',
                             'July/November were previously researched; these are not pristine holdouts.',
                             'Matched models are diagnostic refits, not the complete frozen V3 ensemble.'],
              'folds': []}
    held = []
    previous = None
    for month in months:
        frames = []
        for variant in ['base', 'arrival']:
            name = 'rolling_' + month + '_' + variant
            if not args.score_only:
                command = [sys.executable, str(ROOT/'scripts/train.py'), '--name', name,
                           '--rolling-month', month, '--iterations', '2000', '--depth', '8',
                           '--rate', '0.0750579', '--l2', '4.090916', '--threads', '6',
                           '--residual', '--interactions', '--tfm', 'data/processed/timesfm_hourly_dev.parquet',
                           '--weather', 'data/processed/weather_hourly.parquet', '--extra',
                           'data/processed/queue_features.parquet' if variant == 'base' else
                           'data/processed/queue_arrival_context_v3.parquet']
                subprocess.run(command, cwd=ROOT, check=True)
            frames.append(pd.read_parquet(ROOT/'artifacts'/f'{name}_predictions.parquet'))
        base = frames[0]
        candidate = align(frames[1], base[ID], 'arrival')
        start = pd.Timestamp(month + '-01', tz='UTC')
        if not (base[TIME].ge(start) & base[TIME].lt(start + pd.offsets.MonthBegin(1))).all():
            raise ValueError('Prediction dates do not match the requested fold')
        for variant in ['base', 'arrival']:
            metadata = json.loads((ROOT/'artifacts'/f'rolling_{month}_{variant}.json').read_text())
            split = metadata['split']
            if pd.Timestamp(split['train_max']) >= start or split['validation_used_for_early_stopping']:
                raise ValueError('Validation labels leaked into model training or stopping')
        np.testing.assert_array_equal(base[TARGET], candidate[TARGET])
        y = base[TARGET].to_numpy()
        a, b = base.prediction.to_numpy(), candidate.prediction.to_numpy()
        if not len(y) or not np.isfinite(np.column_stack([y, a, b])).all():
            raise ValueError('Empty check partition or nonfinite values')
        alpha = 1.0 if previous is None else blend_alpha(previous[1], previous[2], previous[0])
        pred = a + alpha*(b-a)
        scores = {name: float(np.sqrt(np.mean((v-y)**2)))
                  for name, v in [('baseline', a), ('arrival', b), ('calibrated_blend', pred)]}
        report['folds'].append({'month': month, 'calibration_month': None if previous is None else previous[3],
                                'check_rows': len(y), 'alpha': alpha, 'check_rmse': scores})
        held.append(np.column_stack([y, a, b, pred]))
        previous = (y, a, b, month)
        print(json.dumps(report['folds'][-1]), flush=True)
    values = np.concatenate(held)
    report['pooled_check_rmse'] = dict(zip(['baseline', 'arrival', 'calibrated_blend'],
        np.sqrt(np.mean((values[:, 1:]-values[:, :1])**2, axis=0)).tolist()))
    report['change_vs_baseline'] = report['pooled_check_rmse']['calibrated_blend'] - report['pooled_check_rmse']['baseline']
    report['improves_every_fold'] = all(f['check_rmse']['calibrated_blend'] < f['check_rmse']['baseline'] for f in report['folds'])
    (ROOT/'docs/rolling_backtest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
