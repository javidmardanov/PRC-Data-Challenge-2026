"""Independent aggregate check of frozen regional A2 on complete forward folds."""
import json,sys
import numpy as np
import pandas as pd
from ensemble import ROOT,ID,TIME,TARGET

GLOBAL=.630797588656986

def main(month='2025-07'):
    art=ROOT/'artifacts';tag='july' if month.endswith('07') else 'november'
    base=pd.read_parquet(art/f'forward_v3_{month}_complete.parquet');regional=pd.read_parquet(art/f'autoresearch_lobt_regional_forward_{tag}.parquet')
    raw=pd.read_parquet(art/f'known_lobt_d10_forward_{tag}_predictions.parquet');pos=pd.Index(base[ID]).get_indexer(raw[ID]);assert (pos>=0).all()
    expected=pd.Index(base.loc[base[ID].isin(raw[ID]),ID]);assert not len(expected.symmetric_difference(pd.Index(raw[ID])))
    globalp=base.prediction.to_numpy(float).copy();globalp[pos]+=GLOBAL*(raw.prediction.to_numpy(float)-globalp[pos])
    bounds=pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[base[ID]];globalp[pos]=np.clip(globalp[pos],bounds.lower_bound.fillna(-np.inf).to_numpy()[pos],bounds.upper_bound.fillna(np.inf).to_numpy()[pos])
    y=base[TARGET].to_numpy(float);rp=regional.prediction.to_numpy(float);bp=base.prediction.to_numpy(float);outside=np.ones(len(base),bool);outside[pos]=False
    rmse=lambda p:float(np.sqrt(np.mean((y-p)**2)));daily=pd.DataFrame({'day':base[TIME].dt.floor('D'),'delta':(y-rp)**2-(y-globalp)**2}).groupby('day').delta.sum().to_numpy()
    rng=np.random.default_rng(20260915);boot=np.array([daily[rng.integers(0,len(daily),len(daily))].sum() for _ in range(10000)])
    report={'month':month,'gate_rows':len(pos),'baseline_rmse':rmse(bp),'global_alpha_rmse':rmse(globalp),'regional_rmse':rmse(rp),
            'regional_vs_global_sse_delta':float(np.sum((y-rp)**2)-np.sum((y-globalp)**2)),
            'day_bootstrap_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'day_bootstrap_probability_improvement':float(np.mean(boot<0)),
            'outside_gate_bitwise_unchanged':bool(np.array_equal(bp[outside],rp[outside]))}
    assert report['outside_gate_bitwise_unchanged'];path=ROOT/f'docs/autoresearch_lobt_regional_forward_{tag}.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '2025-07')
