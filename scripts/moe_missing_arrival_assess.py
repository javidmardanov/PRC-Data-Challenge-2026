"""Cross-month assessment of the fixed missing-arrival correction."""
import json
import numpy as np
import pandas as pd
from ensemble import ID,TARGET,TIME,validation_labels
from missing_v2 import OUT,ROOT,load_validation,rmse

_,missing,_,_=load_validation(); v2=pd.read_parquet(OUT/'v2_validation.parquet')
full=validation_labels(OUT/'rows.parquet',v2[ID].to_numpy());base=v2.prediction.to_numpy()
candidate=pd.read_parquet(OUT/'moe_missing_arrival_v3_fixed_complete_validation.parquet').set_index(ID).loc[v2[ID],'prediction'].to_numpy()
pos=pd.Index(v2[ID]).get_indexer(missing[ID]); b,c=base[pos],candidate[pos];y=missing[TARGET].to_numpy();m=missing[TIME].dt.month.to_numpy();d=c-b
report={'full_months':{},'cross_month':{}}
for month in [7,11]:
    mask=full[TIME].dt.month.eq(month).to_numpy();report['full_months'][str(month)]={'baseline':rmse(full.loc[mask,TARGET],base[mask]),'candidate':rmse(full.loc[mask,TARGET],candidate[mask])}
for fit_month,check_month in [(7,11),(11,7)]:
    fit=m==fit_month;check=m==check_month;beta=float(np.clip(np.dot(d[fit],y[fit]-b[fit])/max(np.dot(d[fit],d[fit]),1),0,1))
    report['cross_month'][f'fit_{fit_month}_check_{check_month}']={'beta':beta,'baseline_rmse':rmse(y[check],b[check]),'rmse':rmse(y[check],b[check]+beta*d[check])}
report['selected']='missing_arrival_fixed_existing_alphas' if all(v['rmse']<v['baseline_rmse'] for v in report['cross_month'].values()) else 'frozen_v2'
(ROOT/'docs/research_moe_missing_arrival_assessment.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
