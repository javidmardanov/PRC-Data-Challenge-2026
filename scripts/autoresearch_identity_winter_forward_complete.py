"""Apply frozen winter identity correction to complete November forward V3."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT,ID,TIME,TARGET,PRED

ALPHA=.8788581367641393

def main():
    art=ROOT/'artifacts';base=pd.read_parquet(art/'forward_v3_2025-11_complete.parquet');raw=pd.read_parquet(art/'autoresearch_flight_identity_2025-11_raw.parquet')
    rows=pd.read_parquet(art/'rows.parquet',columns=[ID,'ADEP_mvt']);feat=pd.read_parquet(art/'features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_SCHED_TIME_UTC_mvt']);rows[['aobt','sched']]=feat.to_numpy();meta=rows.set_index(ID).loc[base[ID]]
    gate=(meta.aobt.to_numpy()<=-100000)&meta.ADEP_mvt.eq('LIRF').to_numpy()&(meta.sched.to_numpy()<24000)
    expected=pd.Index(base.loc[gate,ID]);found=pd.Index(raw[ID]);
    if raw[ID].duplicated().any() or len(expected.difference(found)):raise ValueError('Identity raw must cover every winter gate ID uniquely')
    raw=raw.set_index(ID).loc[expected];pos=np.flatnonzero(gate);bp=base[PRED].to_numpy(float);candidate=bp.copy();candidate[pos]+=ALPHA*(raw[PRED].to_numpy(float)-bp[pos])
    if not np.array_equal(candidate[~gate],bp[~gate]) or not np.isfinite(candidate).all():raise AssertionError('Gate preservation/finite invariant failed')
    out=base.copy();out[PRED]=candidate;path=art/'autoresearch_identity_winter_forward_november.parquet';out.to_parquet(path,index=False)
    y=base[TARGET].to_numpy(float);rmse=lambda p,m:float(np.sqrt(np.mean((y[m]-p[m])**2)));daily=pd.DataFrame({'day':base[TIME].dt.floor('D'),'delta':(y-candidate)**2-(y-bp)**2}).groupby('day').delta.sum().to_numpy()
    rng=np.random.default_rng(20260915);boot=np.array([daily[rng.integers(0,len(daily),len(daily))].sum() for _ in range(10000)])
    report={'alpha':ALPHA,'rows':len(base),'gate_rows':int(gate.sum()),'full_rmse':{'baseline':rmse(bp,np.ones(len(bp),bool)),'candidate':rmse(candidate,np.ones(len(bp),bool))},
            'gate_rmse':{'baseline':rmse(bp,gate),'candidate':rmse(candidate,gate)},'sse_delta':float(np.sum((y-candidate)**2)-np.sum((y-bp)**2)),
            'day_bootstrap_ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'day_bootstrap_probability_improvement':float(np.mean(boot<0)),
            'outside_gate_bitwise_unchanged':True,'output':str(path.relative_to(ROOT)).replace('\\','/')}
    (ROOT/'docs/autoresearch_identity_winter_forward_november.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
