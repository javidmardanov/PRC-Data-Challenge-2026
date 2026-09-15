"""Airport-ridge recalibration of the frozen known-LOBT A2 correction."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET, align, validation_labels

GLOBAL = .630797588656986


def apply(base, raw, airport, gate, alpha, lower, upper):
    a = airport.map(alpha).fillna(GLOBAL).to_numpy(float)
    out = base.copy(); out[gate] += a[gate]*(raw[gate]-base[gate])
    out[gate] = np.clip(out[gate], np.maximum(lower[gate], 0), upper[gate])
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--map-base',type=Path);ap.add_argument('--map-raw',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    if args.map_base or args.map_raw:
        if not (args.map_base and args.map_raw and args.output):ap.error('--map-base, --map-raw and --output are required together')
        config=json.loads((ROOT/'docs/autoresearch_lobt_regional.json').read_text());sel=config['selected_index']
        if sel is None:raise ValueError('No frozen regional candidate')
        coeff=config['trials'][sel]['coefficients'];bf=pd.read_parquet(args.map_base);value='prediction' if 'prediction' in bf else TARGET
        rf=pd.read_parquet(args.map_raw);rows=pd.read_parquet(ROOT/'artifacts/rows.parquet',columns=[ID,'ADEP_mvt'])
        feat=pd.read_parquet(ROOT/'artifacts/features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_LOBT_flt']);rows[['aobt','lobt']]=feat.to_numpy();meta=rows.set_index(ID).loc[bf[ID]]
        gate=(meta.aobt.to_numpy()>-100000)&(meta.lobt.to_numpy()>-100000);bounds=pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[bf[ID]]
        if rf[ID].duplicated().any() or rf[ID].isna().any():raise ValueError('mapping raw IDs must be unique and non-null')
        expected=pd.Index(bf.loc[gate,ID]);found=pd.Index(rf[ID]);raw=bf[value].to_numpy(float).copy()
        if len(found)==len(bf) and not len(pd.Index(bf[ID]).symmetric_difference(found)):
            raw=align(rf,bf[ID],'mapping raw').prediction.to_numpy(float)
        elif not len(expected.symmetric_difference(found)):
            pos=pd.Index(bf[ID]).get_indexer(rf[ID]);raw[pos]=rf.prediction.to_numpy(float)
        else:raise ValueError('mapping raw IDs must equal either all base IDs or the complete applicable gate IDs')
        pred=apply(bf[value].to_numpy(float),raw,meta.ADEP_mvt,gate,coeff,bounds.lower_bound.fillna(-np.inf).to_numpy(),bounds.upper_bound.fillna(np.inf).to_numpy())
        out=bf.copy();out[value]=pred;args.output.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(args.output,index=False);print(f'Wrote {args.output}');return
    art=ROOT/'artifacts';basef=pd.read_parquet(art/'v3_validation.parquet');ids=basef[ID]
    labels=validation_labels(art/'rows.parquet',ids); y=labels[TARGET].to_numpy(float);base=basef.prediction.to_numpy(float)
    rows=pd.read_parquet(art/'rows.parquet',columns=[ID,'ADEP_mvt']);feat=pd.read_parquet(art/'features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_LOBT_flt']);rows[['aobt','lobt']]=feat.to_numpy();meta=rows.set_index(ID).loc[ids]
    raw=align(pd.read_parquet(art/'known_lobt_d10_a2_predictions.parquet'),ids,'A2 raw').prediction.to_numpy(float)
    gate=(meta.aobt.to_numpy()>-100000)&(meta.lobt.to_numpy()>-100000)
    bounds=pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[ids]
    lower=bounds.lower_bound.fillna(-np.inf).to_numpy(float);upper=bounds.upper_bound.fillna(np.inf).to_numpy(float)
    globalp=apply(base,raw,meta.ADEP_mvt,gate,{},lower,upper)
    saved=align(pd.read_parquet(art/'known_lobt_d10_a2_gated_validation.parquet'),ids,'saved global').prediction.to_numpy(float)
    reconstruction=float(np.max(np.abs(globalp-saved)))
    cal=labels[TIME].dt.day.le(14).to_numpy()&gate;late=labels[TIME].dt.day.ge(15).to_numpy();d=raw-base;r=y-base
    mean_d2=float(np.mean(d[cal]**2));trials=[];outputs=[]
    for strength in (1000,5000):
        coeff={}
        for airport,ix in meta.loc[cal].groupby('ADEP_mvt').groups.items():
            pos=pd.Index(ids).get_indexer(ix);assert (pos>=0).all();den=float(d[pos]@d[pos]+strength*mean_d2)
            coeff[airport]=float(np.clip((d[pos]@r[pos]+strength*mean_d2*GLOBAL)/den,0,1))
        pred=apply(base,raw,meta.ADEP_mvt,gate,coeff,lower,upper);outputs.append(pred)
        score=lambda p,m:float(np.sqrt(np.mean((y[m]-p[m])**2)))
        result={'pseudo_rows':strength,'coefficients':coeff,'coefficient_range':[min(coeff.values()),max(coeff.values())],
                'full':{'v3':score(base,np.ones(len(y),bool)),'global_a2':score(globalp,np.ones(len(y),bool)),'regional':score(pred,np.ones(len(y),bool))},'late':{}}
        blendpath=art/'autoresearch_lobt_blend_validation.parquet'
        blend=align(pd.read_parquet(blendpath),ids,'three-column blend').prediction.to_numpy(float) if blendpath.exists() else None
        if blend is not None:result['full']['three_column_blend']=score(blend,np.ones(len(y),bool))
        for m in (7,11):
            mask=late&labels[TIME].dt.month.eq(m).to_numpy();result['late'][str(m)]={'v3':score(base,mask),'global_a2':score(globalp,mask),'regional':score(pred,mask)}
            if blend is not None:result['late'][str(m)]['three_column_blend']=score(blend,mask)
        trials.append(result)
    eligible=[i for i,t in enumerate(trials) if all(t['late'][str(m)]['regional']<t['late'][str(m)]['global_a2'] for m in (7,11))]
    selected=min(eligible,key=lambda i:trials[i]['full']['regional']) if eligible else None
    report={'december_read':False,'global_alpha':GLOBAL,'reconstruction_max_abs':reconstruction,'ridge_definition':'penalty equals pseudo_rows times global calibration mean(delta^2), centered on global alpha','trials':trials,'selected_index':selected}
    if selected is not None:
        out=basef[[ID]].assign(prediction=outputs[selected]);path=art/'autoresearch_lobt_regional_validation.parquet';out.to_parquet(path,index=False);report['validation_output']=str(path.relative_to(ROOT)).replace('\\','/')
    (ROOT/'docs/autoresearch_lobt_regional.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
