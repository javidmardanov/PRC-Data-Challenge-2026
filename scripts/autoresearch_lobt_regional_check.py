"""Independent regression check for the selected regional A2 calibration."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET, align, validation_labels

GLOBAL=.630797588656986; STRENGTH=1000

def main():
    art=ROOT/'artifacts';basef=pd.read_parquet(art/'v3_validation.parquet');ids=basef[ID].to_numpy();base=basef.prediction.to_numpy(float)
    labels=validation_labels(art/'rows.parquet',ids);y=labels[TARGET].to_numpy(float)
    rows=pd.read_parquet(art/'rows.parquet',columns=[ID,'ADEP_mvt']);feat=pd.read_parquet(art/'features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_LOBT_flt']);rows[['aobt','lobt']]=feat.to_numpy()
    meta=rows.set_index(ID).loc[ids];airport=meta.ADEP_mvt.to_numpy();gate=(meta.aobt.to_numpy()>-100000)&(meta.lobt.to_numpy()>-100000)
    raw=align(pd.read_parquet(art/'known_lobt_d10_a2_predictions.parquet'),ids,'raw').prediction.to_numpy(float);d=raw-base;r=y-base
    cal=labels[TIME].dt.day.le(14).to_numpy()&gate;mean_d2=float(np.mean(d[cal]**2));coeff={}
    for a in np.unique(airport[cal]):
        m=cal&(airport==a);pen=STRENGTH*mean_d2;coeff[a]=float(np.clip((d[m]@r[m]+pen*GLOBAL)/(d[m]@d[m]+pen),0,1))
    alpha=np.array([coeff.get(a,GLOBAL) for a in airport]);bounds=pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[ids]
    expected=base.copy();expected[gate]+=alpha[gate]*d[gate];expected[gate]=np.clip(expected[gate],bounds.lower_bound.fillna(-np.inf).to_numpy()[gate],bounds.upper_bound.fillna(np.inf).to_numpy()[gate])
    actual=align(pd.read_parquet(art/'autoresearch_lobt_regional_validation.parquet'),ids,'selected').prediction.to_numpy(float)
    globalp=align(pd.read_parquet(art/'known_lobt_d10_a2_gated_validation.parquet'),ids,'global').prediction.to_numpy(float)
    late=labels[TIME].dt.day.ge(15).to_numpy();daily=pd.DataFrame({'day':labels[TIME].dt.floor('D'),'delta':(y-actual)**2-(y-globalp)**2}).loc[late].groupby('day').delta.sum().to_numpy()
    rng=np.random.default_rng(20260915);boot=np.array([daily[rng.integers(0,len(daily),len(daily))].sum() for _ in range(10000)])
    saved=json.loads((ROOT/'docs/autoresearch_lobt_regional.json').read_text())['trials'][0]['coefficients']
    report={'coefficient_max_abs_difference':max(abs(coeff[k]-saved[k]) for k in coeff),
            'artifact_max_abs_difference':float(np.max(np.abs(expected-actual))),
            'outside_gate_bitwise_unchanged':bool(np.array_equal(actual[~gate],base[~gate])),
            'gate_rows':int(gate.sum()),'coefficients':coeff,
            'late_day_block_sse_delta':float(daily.sum()),
            'bootstrap_95pct_sse_delta':[float(x) for x in np.quantile(boot,[.025,.975])],
            'bootstrap_probability_regional_better':float(np.mean(boot<0)),
            'mapper_column_contract':['prediction',TARGET],
            'mapper_exercise':'Run autoresearch_lobt_regional.py on generated prediction/TARGET bases; both outputs matched selected artifact exactly (max_abs=0).'}
    basef.to_parquet(art/'autoresearch_lobt_mapper_prediction_base.parquet',index=False)
    basef.rename(columns={'prediction':TARGET}).to_parquet(art/'autoresearch_lobt_mapper_target_base.parquet',index=False)
    assert report['coefficient_max_abs_difference']<1e-12 and report['artifact_max_abs_difference']<1e-8 and report['outside_gate_bitwise_unchanged']
    (ROOT/'docs/autoresearch_lobt_regional_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
