"""Independent arithmetic and invariant check for assembled V5/V6 artifacts."""
import hashlib,json
import numpy as np
import pandas as pd
from ensemble import ROOT,ID,TIME,TARGET,PRED,align

GLOBAL=.630797588656986;WINTER=.8788581367641393

def main():
    art=ROOT/'artifacts';base=pd.read_parquet(ROOT/'submissions/elegant-alligator_v3.parquet');ids=base[ID];bp=base[TARGET].to_numpy(float)
    rows=pd.read_parquet(art/'rows.parquet',columns=[ID,TIME,'ADEP_mvt']);feat=pd.read_parquet(art/'features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_LOBT_flt','mvt_minus_SCHED_TIME_UTC_mvt']);rows[['aobt','lobt','sched']]=feat.to_numpy();meta=rows.set_index(ID).loc[ids]
    gate=(meta.aobt.to_numpy()>-100000)&(meta.lobt.to_numpy()>-100000);bounds=pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[ids];lo=bounds.lower_bound.fillna(-np.inf).to_numpy();hi=bounds.upper_bound.fillna(np.inf).to_numpy()
    raw=align(pd.read_parquet(art/'known_lobt_d10_a2_final_predictions.parquet'),ids,'final A2')[PRED].to_numpy(float)
    expected5=bp.copy();expected5[gate]=np.clip(bp[gate]+GLOBAL*(raw[gate]-bp[gate]),np.maximum(lo[gate],0),hi[gate])
    cfg=json.loads((ROOT/'docs/autoresearch_lobt_regional.json').read_text());coeff=cfg['trials'][cfg['selected_index']]['coefficients'];a=meta.ADEP_mvt.map(coeff).fillna(GLOBAL).to_numpy(float)
    expected6=bp.copy();expected6[gate]=np.clip(bp[gate]+a[gate]*(raw[gate]-bp[gate]),np.maximum(lo[gate],0),hi[gate])
    wg=(meta.aobt.to_numpy()<-100000)&meta.ADEP_mvt.eq('LIRF').to_numpy()&meta[TIME].dt.month.eq(1).to_numpy()&(meta.sched.to_numpy()<24000)
    ident=pd.read_parquet(art/'autoresearch_flight_identity_final_raw.parquet').set_index(ID);pos=np.flatnonzero(wg);expected6[pos]+=WINTER*(ident.loc[ids.iloc[pos],PRED].to_numpy(float)-bp[pos])
    report={'status':'complete','ranking':{},'local_validation':{}}
    for version,expected in [(5,expected5),(6,expected6)]:
        path=ROOT/f'submissions/elegant-alligator_v{version}.parquet';actual=pd.read_parquet(path)
        changed=gate if version==5 else gate|wg
        report['ranking'][str(version)]={'rows':len(actual),'id_order_exact':bool(np.array_equal(actual[ID],ids)),'finite':bool(np.isfinite(actual[TARGET]).all()),'nonnegative':bool((actual[TARGET]>=0).all()),
          'negative_rows':int((actual[TARGET]<0).sum()),'baseline_negative_rows':int((bp<0).sum()),
          'max_abs_arithmetic_difference':float(np.max(np.abs(actual[TARGET].to_numpy(float)-expected))),
          'outside_gate_bitwise_unchanged':bool(np.array_equal(actual.loc[~changed,TARGET].to_numpy(),bp[~changed])),
          'known_gate_within_bounds':bool(np.all((actual.loc[gate,TARGET].to_numpy()>=lo[gate])&(actual.loc[gate,TARGET].to_numpy()<=hi[gate]))),
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # Independently reproduce saved local combined validation artifacts.
    v3=pd.read_parquet(art/'v3_validation.parquet');vids=v3[ID]
    globalv=align(pd.read_parquet(art/'known_lobt_d10_a2_gated_validation.parquet'),vids,'local global')
    regional=align(pd.read_parquet(art/'autoresearch_lobt_regional_validation.parquet'),vids,'local regional')
    winter=align(pd.read_parquet(art/'autoresearch_identity_winter_validation.parquet'),vids,'local winter')
    for version,primary in [(5,globalv),(6,regional)]:
        expected=primary[PRED].to_numpy(float).copy()
        if version==6:expected+=winter[PRED].to_numpy(float)-v3[PRED].to_numpy(float)
        expected=np.maximum(expected,0)
        savedf=align(pd.read_parquet(art/f'autoresearch_v{version}_validation.parquet'),vids,f'local V{version}')
        saved=savedf[PRED].to_numpy(float)
        report['local_validation'][str(version)]={'id_order_exact':bool(np.array_equal(savedf[ID],vids)),'max_abs_reconstruction_difference':float(np.max(np.abs(saved-expected)))}
    assert all(x['max_abs_arithmetic_difference']<1e-9 and x['id_order_exact'] and x['finite'] and x['nonnegative'] and x['outside_gate_bitwise_unchanged'] and x['known_gate_within_bounds'] for x in report['ranking'].values())
    assert all(x['max_abs_reconstruction_difference']<1e-9 for x in report['local_validation'].values())
    (ROOT/'docs/autoresearch_final_artifact_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
