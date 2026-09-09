"""Deployment-weight assessment for the known-Rome arrival trial."""
import json
import numpy as np
import pandas as pd
from ensemble import ID, TARGET, TIME, validation_labels
from lirf_identity_experiment import OUT, ROOT, SCHEDULE, load, rmse

x, meta = load()
valid = meta[TIME].dt.month.isin([7, 11]).to_numpy()
vm = meta.loc[valid].reset_index(drop=True)
v2 = pd.read_parquet(OUT/'v2_validation.parquet')
full = validation_labels(OUT/'rows.parquet', v2[ID].to_numpy()); full['prediction'] = v2.prediction
base = full.set_index(ID).loc[vm[ID], 'prediction'].to_numpy()
expert = pd.read_parquet(OUT/'moe_known_arrival_v3_raw_validation.parquet').set_index(ID).loc[vm[ID], 'expert_prediction'].to_numpy()
y = vm[TARGET].to_numpy(float); month = vm[TIME].dt.month.to_numpy(); day = vm[TIME].dt.day.to_numpy()
known = x.loc[valid, 'mvt_minus_AOBT_3_flt'].gt(-100000).to_numpy()
eligible = known & x.loc[valid, SCHEDULE].lt(24000).to_numpy()
d = expert-base; fit = eligible & (day%2==0)
alpha = float(np.clip(np.dot(d[fit], y[fit]-base[fit])/max(np.dot(d[fit],d[fit]),1),0,1))
report = {'frozen_even_alpha': alpha, 'odd_checks': {}}
for m in [7,11]:
    mask = eligible & (day%2==1) & (month==m)
    pred = base[mask]+alpha*d[mask]
    report['odd_checks'][str(m)] = {'rows': int(mask.sum()), 'rmse': rmse(y[mask],pred),
                                    'baseline_rmse': rmse(y[mask],base[mask])}
report['selected'] = 'frozen_v2_no_correction' if any(v['rmse']>=v['baseline_rmse'] for v in report['odd_checks'].values()) else 'arrival_even_alpha'
(ROOT/'docs/research_moe_known_arrival_assessment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
