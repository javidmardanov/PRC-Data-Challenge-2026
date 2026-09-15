"""Estimate V2/V3 risk using ranking-visible season/airport/missingness proportions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ensemble import ROOT, ID, TIME, TARGET, align


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--versions', nargs='+', default=['v2', 'v3'])
    parser.add_argument('--output', type=Path, default=ROOT/'docs/validation_shift.json')
    args = parser.parse_args()
    rows = pd.read_parquet(ROOT/'artifacts/rows.parquet')
    proxy = pd.read_parquet(ROOT/'artifacts/features.parquet', columns=['mvt_minus_AOBT_3_flt'])
    rows['missing'] = proxy.iloc[:, 0].lt(-100000).to_numpy()
    rows['season'] = np.where(rows[TIME].dt.month.eq(7), 'summer', 'winter')
    ranking = rows.loc[rows.source.eq('ranking')]
    v3 = pd.read_parquet(ROOT/'artifacts/v3_validation.parquet')
    meta = rows.set_index(ID).loc[v3[ID]].reset_index()
    keys = ['season', 'ADEP_mvt', 'missing']
    target_share = ranking.groupby(keys).size()/len(ranking)
    local_count = meta.groupby(keys).size()
    if len(target_share.index.difference(local_count.index)):
        raise ValueError('Ranking has strata absent from validation')
    weights = meta[keys].join((target_share/local_count).rename('weight'), on=keys).weight.fillna(0).to_numpy()
    np.testing.assert_allclose(weights.sum(), 1)
    report = {'method': 'Reweight local squared errors by ranking season, airport, and AOBT missingness proportions.',
              'limitation': 'November approximates January; cannot detect error changes within strata or remove tuning optimism.',
              'models': {}}
    for version in args.versions:
        prediction = align(pd.read_parquet(ROOT/'artifacts'/f'{version}_validation.parquet'), v3[ID], version).prediction.to_numpy()
        errors = (prediction-meta[TARGET].to_numpy())**2
        report['models'][version] = {'local_rmse': float(np.sqrt(errors.mean())),
                                     'ranking_mix_rmse': float(np.sqrt(weights@errors))}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
