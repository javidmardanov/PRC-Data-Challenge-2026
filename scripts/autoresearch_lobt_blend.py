"""One constrained cached LOBT ensemble trial; fixed early/late date split."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET, align, validation_labels, simplex_weights


def main():
    art = ROOT/'artifacts'
    base = pd.read_parquet(art/'v3_validation.parquet')
    labels = validation_labels(art/'rows.parquet', base[ID])
    rows = pd.read_parquet(art/'rows.parquet', columns=[ID])
    x = pd.read_parquet(art/'features.parquet', columns=['mvt_minus_AOBT_3_flt', 'mvt_minus_LOBT_flt'])
    x[ID] = rows[ID]; x = x.set_index(ID).loc[base[ID]]
    gate = (x.iloc[:, 0].to_numpy() > -100000) & (x.iloc[:, 1].to_numpy() > -100000)
    paths = [art/'v3_validation.parquet', art/'known_lobt_d10_a2_predictions.parquet', art/'known_lobt_d10_a3_predictions.parquet']
    matrix = np.column_stack([align(pd.read_parquet(p), base[ID], str(p)).prediction for p in paths])
    y = labels[TARGET].to_numpy()
    fit = labels[TIME].dt.day.le(14).to_numpy() & gate
    weights = simplex_weights(matrix[fit], y[fit])
    pred = base.prediction.to_numpy().copy(); pred[gate] = matrix[gate]@weights
    bounds = pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[base[ID]]
    pred[gate] = np.clip(pred[gate], bounds.lower_bound.fillna(-np.inf).to_numpy()[gate], bounds.upper_bound.fillna(np.inf).to_numpy()[gate])
    incumbent = align(pd.read_parquet(art/'known_lobt_d10_a2_gated_validation.parquet'), base[ID], 'a2').prediction.to_numpy()
    report = {'weights': weights.tolist(), 'model_order': ['v3', 'known_lobt_d10_a2', 'known_lobt_d10_a3'],
              'full_rmse': float(np.sqrt(np.mean((pred-y)**2))), 'a2_rmse': float(np.sqrt(np.mean((incumbent-y)**2))), 'late': {}}
    for month in [7, 11]:
        mask = labels[TIME].dt.month.eq(month).to_numpy() & labels[TIME].dt.day.ge(15).to_numpy()
        report['late'][str(month)] = {'a2': float(np.sqrt(np.mean((incumbent[mask]-y[mask])**2))),
                                      'blend': float(np.sqrt(np.mean((pred[mask]-y[mask])**2)))}
    report['eligible'] = all(v['blend'] < v['a2'] for v in report['late'].values())
    base[[ID]].assign(prediction=pred).to_parquet(art/'autoresearch_lobt_blend_validation.parquet', index=False)
    (ROOT/'docs/autoresearch_lobt_blend.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
