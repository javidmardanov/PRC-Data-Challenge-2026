"""Fixed-alpha decision checks for cap6000, cap12000, and their 50/50 raw blend."""
import json

import numpy as np
import pandas as pd

from ensemble import ID, TARGET, TIME
from missing_v2 import OUT, ROOT, load_validation, rmse
from moe_calibrate import routed, scores


def main():
    full, valid, _, _ = load_validation()
    base, y = valid.prediction.to_numpy(), valid[TARGET].to_numpy()
    old = pd.read_parquet(OUT/'missing_v2_mixture_validation.parquet').set_index(ID).loc[valid[ID], 'prediction'].to_numpy()
    new = pd.read_parquet(OUT/'moe_normal_cap12000_raw_missing_validation.parquet').set_index(ID).loc[valid[ID], 'prediction'].to_numpy()
    alpha = {'Rome': 0.4224120030856563, 'other': 1.0}
    day = valid[TIME].dt.day.to_numpy()
    report = {'alphas': alpha, 'candidates': {}}
    for name, raw in [('cap6000', old), ('cap12000', new), ('raw_average_50_50', .5*(old+new))]:
        pred = routed(base, raw, valid, alpha)
        row = {'scores': scores(full, valid, pred), 'date_parity': {}}
        for parity in [0, 1]:
            mask = day % 2 == parity
            row['date_parity'][str(parity)] = {'rows': int(mask.sum()), 'missing_rmse': rmse(y[mask], pred[mask]),
                'missing_sse': float(np.square(y[mask]-pred[mask]).sum())}
        report['candidates'][name] = row
    incumbent = report['candidates']['cap6000']
    for candidate in report['candidates'].values():
        candidate['delta_vs_cap6000'] = {
            'full': candidate['scores']['combined']-incumbent['scores']['combined'],
            'july': candidate['scores']['7']-incumbent['scores']['7'],
            'november': candidate['scores']['11']-incumbent['scores']['11'],
            'even_missing': candidate['date_parity']['0']['missing_rmse']-incumbent['date_parity']['0']['missing_rmse'],
            'odd_missing': candidate['date_parity']['1']['missing_rmse']-incumbent['date_parity']['1']['missing_rmse']}
    report['selected'] = 'cap12000_fixed_original_alphas'
    report['diagnosis'] = ('At fixed original alphas, cap12000 improves both date parities if both deltas are negative; '
                           'the prior even->odd held loss then comes from unstable refitted Rome alpha.')
    (ROOT/'docs'/'research_moe_cap_decision.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
