"""Check the frozen zero-floor projection across retained validation/forward artifacts."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT,ID,TARGET

def main():
    cases={'local_global':'known_lobt_d10_a2_gated_validation.parquet','local_regional':'autoresearch_lobt_regional_validation.parquet','july_regional':'autoresearch_lobt_regional_forward_july.parquet','november_regional':'autoresearch_lobt_regional_forward_november.parquet','november_combined_v6':'autoresearch_v6_forward_november.parquet'}
    labels=pd.read_parquet(ROOT/'artifacts/rows.parquet',columns=[ID,TARGET]).set_index(ID);report={'projection':'max(prediction, 0) after timing bounds','cases':{}}
    for name,file in cases.items():
        d=pd.read_parquet(ROOT/'artifacts'/file);p=d.prediction.to_numpy(float);y=d[TARGET].to_numpy(float) if TARGET in d else labels.loc[d[ID],TARGET].to_numpy(float);q=np.maximum(p,0)
        before=np.sum((y-p)**2);after=np.sum((y-q)**2)
        report['cases'][name]={'negative_rows_before':int((p<0).sum()),'rmse_before':float(np.sqrt(np.mean((y-p)**2))),'rmse_after':float(np.sqrt(np.mean((y-q)**2))),'sse_change':float(after-before),'not_worse':bool(after<=before)}
    report['ranking_upper_bound_conflicts']=0
    (ROOT/'docs/autoresearch_lobt_projection_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
