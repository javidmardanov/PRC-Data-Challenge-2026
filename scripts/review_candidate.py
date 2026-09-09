"""Independently compare complete candidate predictions with the frozen v1."""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import (ROOT, ID, TIME, TARGET, PRED, align, frozen_predictions,
                      metrics, read_predictions, validation_labels)


def review(candidate, reference=None):
    frame = read_predictions(candidate)
    labels = validation_labels(ROOT/'artifacts/rows.parquet', frame[ID])
    if reference:
        baseline = align(read_predictions(reference), frame[ID], 'reference')[PRED].to_numpy()
    else:
        config = json.loads((ROOT/'docs/v1_ensemble.json').read_text())
        _, baseline = frozen_predictions(config, config['models'],
            ROOT/'artifacts/missing_validation.parquet', ids=frame[ID],
            rome=ROOT/'artifacts/lirf_identity_expert_validation.parquet')
    prediction = frame[PRED].to_numpy(dtype=float)
    target = labels[TARGET].to_numpy(dtype=float)
    error = (prediction-target)**2
    base_error = (baseline-target)**2
    # Resample whole calendar days to retain shared airport/weather errors.
    # Repeated tuning means these intervals are descriptive, not fresh-test CIs.
    daily = pd.DataFrame({'day': labels[TIME].dt.floor('D'),
        'candidate': error, 'reference': base_error, 'rows': 1}).groupby('day').sum()
    samples = np.random.default_rng(20260909).integers(0, len(daily), (2000, len(daily)))
    totals = daily.to_numpy()[samples].sum(axis=1)
    delta = np.sqrt(totals[:, 0]/totals[:, 2])-np.sqrt(totals[:, 1]/totals[:, 2])
    result = {'candidate': str(candidate), 'reference': str(reference or 'complete frozen v1'),
        'rows': len(frame), 'candidate_metrics': metrics(labels, prediction),
        'reference_metrics': metrics(labels, baseline),
        'rmse_change': float(np.sqrt(error.mean())-np.sqrt(base_error.mean())),
        'day_bootstrap_rmse_change_95pct': np.quantile(delta, [.025, .975]).tolist(),
        'uncertainty_note': 'Exploratory: July/November repeatedly used for tuning; not an untouched test.',
        'changed_rows': int(np.count_nonzero(prediction != baseline)),
        'december_scored': False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = review(args.candidate, args.reference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
