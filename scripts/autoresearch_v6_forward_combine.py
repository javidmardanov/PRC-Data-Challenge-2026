"""Combine frozen, disjoint November regional-LOBT and winter-identity deltas."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT,ID,TIME,TARGET,PRED

def main():
    art=ROOT/'artifacts';base=pd.read_parquet(art/'forward_v3_2025-11_complete.parquet');regional=pd.read_parquet(art/'autoresearch_lobt_regional_forward_november.parquet');winter=pd.read_parquet(art/'autoresearch_identity_winter_forward_november.parquet')
    if not (np.array_equal(base[ID],regional[ID]) and np.array_equal(base[ID],winter[ID])):raise ValueError('ID/order mismatch')
    bp=base[PRED].to_numpy(float);rd=regional[PRED].to_numpy(float)-bp;wd=winter[PRED].to_numpy(float)-bp
    rg=rd!=0;wg=wd!=0
    if np.any(rg&wg):raise ValueError('Frozen correction gates overlap')
    combined=bp+rd+wd;out=base.copy();out[PRED]=combined;path=art/'autoresearch_v6_forward_november.parquet';out.to_parquet(path,index=False)
    y=base[TARGET].to_numpy(float);rmse=lambda p:float(np.sqrt(np.mean((y-p)**2)));daily=pd.DataFrame({'day':base[TIME].dt.floor('D'),'delta':(y-combined)**2-(y-bp)**2}).groupby('day').delta.sum().to_numpy()
    rng=np.random.default_rng(20260915);boot=np.array([daily[rng.integers(0,len(daily),len(daily))].sum() for _ in range(10000)])
    report={'rows':len(base),'regional_changed_rows':int(rg.sum()),'winter_changed_rows':int(wg.sum()),'overlap_rows':0,
            'rmse':{'v3':rmse(bp),'regional':rmse(regional[PRED].to_numpy(float)),'winter':rmse(winter[PRED].to_numpy(float)),'combined_v6':rmse(combined)},
            'combined_sse_delta_vs_v3':float(np.sum((y-combined)**2)-np.sum((y-bp)**2)),
            'day_bootstrap_ci95_vs_v3':[float(x) for x in np.quantile(boot,[.025,.975])],'day_bootstrap_probability_improvement_vs_v3':float(np.mean(boot<0)),
            'outside_union_bitwise_unchanged':bool(np.array_equal(combined[~(rg|wg)],bp[~(rg|wg)])),'output':str(path.relative_to(ROOT)).replace('\\','/')}
    assert report['outside_union_bitwise_unchanged'];(ROOT/'docs/autoresearch_v6_forward_november.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
