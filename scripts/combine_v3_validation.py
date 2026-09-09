"""Combine the disjoint validated CatBoost and missing-row corrections."""
import numpy as np

from ensemble import ROOT, ID, PRED, align, read_predictions


def main():
    artifacts = ROOT/'artifacts'
    baseline = read_predictions(artifacts/'v2_validation.parquet')
    cat = align(read_predictions(artifacts/'catboost_tail_d8_full_v3_gated_validation.parquet'), baseline[ID], 'tail correction')
    missing = align(read_predictions(artifacts/'moe_missing_arrival_v3_fixed_complete_validation.parquet'), baseline[ID], 'missing correction')
    cat_delta = cat[PRED].to_numpy()-baseline[PRED].to_numpy()
    missing_delta = missing[PRED].to_numpy()-baseline[PRED].to_numpy()
    if np.any((cat_delta != 0) & (missing_delta != 0)):
        raise ValueError('The tail and missing corrections must affect disjoint rows')
    result = baseline.copy()
    result[PRED] += cat_delta+missing_delta
    result.to_parquet(artifacts/'v3_cat_missing_validation.parquet', index=False)
    print('Wrote the complete v3 base with disjoint corrections')


if __name__ == '__main__':
    main()
