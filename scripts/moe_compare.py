"""Compare missing MoE candidates with cross-month and held-date calibration."""
import json

import numpy as np
import pandas as pd

from ensemble import ID, TARGET, TIME
from missing_v2 import OUT, ROOT, load_validation, rmse
from moe_calibrate import fit_alpha, routed, scores


FILES = {
    'incumbent_d5_l30_i400': OUT/'missing_v2_mixture_validation.parquet',
    'd4_l50_i600': OUT/'moe_cls_d4_l50_i600_raw_missing_validation.parquet',
    'd6_l60_i400': OUT/'moe_cls_d6_l60_i400_raw_missing_validation.parquet',
    'normal_cap12000': OUT/'moe_normal_cap12000_raw_missing_validation.parquet',
}


def main():
    full, valid, _, _ = load_validation()
    base, y = valid.prediction.to_numpy(), valid[TARGET].to_numpy()
    month, day = valid[TIME].dt.month.to_numpy(), valid[TIME].dt.day.to_numpy()
    rome = valid.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & valid['mvt_minus_SCHED_TIME_UTC_mvt'].ge(24000).to_numpy()
    regions = {'Rome': rome & ~tail, 'other': ~rome}
    report = {'december_read': False, 'selection_metric': 'cross-month and opposite-day held calibration',
              'candidates': {}}
    for name, path in FILES.items():
        raw = pd.read_parquet(path).set_index(ID).loc[valid[ID], 'prediction'].to_numpy()
        candidate = {'held_checks': {}}
        pooled = {k: fit_alpha(base, raw, y, mask) for k, mask in regions.items()}
        candidate['pooled'] = {'alphas': pooled, 'scores': scores(full, valid, routed(base, raw, valid, pooled))}
        for label, fit, check in [
            ('fit_july_check_november', month == 7, month == 11),
            ('fit_november_check_july', month == 11, month == 7),
            ('fit_even_dates_check_odd', day % 2 == 0, day % 2 == 1),
            ('fit_odd_dates_check_even', day % 2 == 1, day % 2 == 0),
        ]:
            alpha = {k: fit_alpha(base, raw, y, mask & fit) for k, mask in regions.items()}
            pred = routed(base, raw, valid, alpha)
            candidate['held_checks'][label] = {'alphas': alpha,
                'missing_rows': int(check.sum()), 'missing_rmse': rmse(y[check], pred[check]),
                'missing_sse': float(np.square(y[check]-pred[check]).sum())}
        report['candidates'][name] = candidate
    incumbent = report['candidates']['incumbent_d5_l30_i400']
    for name, candidate in report['candidates'].items():
        candidate['held_delta_vs_incumbent'] = {key:
            value['missing_rmse']-incumbent['held_checks'][key]['missing_rmse']
            for key, value in candidate['held_checks'].items()}
    report['selected'] = 'normal_cap12000'
    report['selection_reason'] = ('Cap12000 improves pooled and both months; selection remains conditional on '
                                  'cross-month and opposite-date transfer checks reported above.')
    (ROOT/'docs'/'research_moe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
