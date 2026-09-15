"""Independent winter-only check of the flight-identity missing-Rome correction."""
import json
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET, align, validation_labels

ALPHA=.8788581367641393

def main():
    art=ROOT/'artifacts';basef=pd.read_parquet(art/'v3_validation.parquet');ids=basef[ID];base=basef.prediction.to_numpy(float)
    labels=validation_labels(art/'rows.parquet',ids);y=labels[TARGET].to_numpy(float)
    rows=pd.read_parquet(art/'rows.parquet',columns=[ID,'ADEP_mvt']);feat=pd.read_parquet(art/'features.parquet',columns=['mvt_minus_AOBT_3_flt','mvt_minus_SCHED_TIME_UTC_mvt']);rows[['aobt','sched']]=feat.to_numpy();meta=rows.set_index(ID).loc[ids]
    gate=(meta.aobt.to_numpy()<=-100000)&(meta.ADEP_mvt.to_numpy()=='LIRF')&(meta.sched.to_numpy()<24000)
    allseason=align(pd.read_parquet(art/'autoresearch_identity_validation.parquet'),ids,'all-season identity').prediction.to_numpy(float)
    # Recover the saved candidate raw delta from the independently stored alpha.
    raw_delta=np.zeros(len(base));raw_delta[gate]=(allseason[gate]-base[gate])/ALPHA
    winter=labels[TIME].dt.month.eq(11).to_numpy();candidate=base.copy();candidate[gate&winter]+=ALPHA*raw_delta[gate&winter]
    out=basef[[ID]].assign(prediction=candidate);out.to_parquet(art/'autoresearch_identity_winter_validation.parquet',index=False)
    late=labels[TIME].dt.day.ge(15).to_numpy();score=lambda p,m:float(np.sqrt(np.mean((y[m]-p[m])**2)))
    daily=pd.DataFrame({'day':labels[TIME].dt.floor('D'),'delta':(y-candidate)**2-(y-base)**2}).loc[late&winter].groupby('day').delta.sum().to_numpy()
    rng=np.random.default_rng(20260915);boot=np.array([daily[rng.integers(0,len(daily),len(daily))].sum() for _ in range(10000)])
    # Strict pre-Nov raw comparison; complete chronological V3 remains pending.
    new=pd.read_parquet(art/'autoresearch_flight_identity_2025-11_raw.parquet')
    old=pd.read_parquet(art/'forward_v3_2025-11_missing_arrival.parquet')
    fold=rows[[ID,'ADEP_mvt','sched']].merge(new,on=ID,validate='one_to_one').merge(old,on=ID,suffixes=('_new','_old'),validate='one_to_one')
    targets=pd.read_parquet(art/'rows.parquet',columns=[ID,TARGET]);fold=fold.merge(targets,on=ID,validate='one_to_one')
    fg=fold.ADEP_mvt.eq('LIRF').to_numpy()&(fold.sched.to_numpy()<24000);fy=fold[TARGET].to_numpy(float);fo=fold.prediction_old.to_numpy(float);fp=fo.copy();fp[fg]+=ALPHA*(fold.prediction_new.to_numpy(float)[fg]-fo[fg])
    fsse=lambda p:float(np.sum((fy[fg]-p[fg])**2))
    report={'december_read':False,'alpha':ALPHA,'scope':'November/January winter only; July unchanged',
      'metadata':['flight_identity','flight_destination'],'target_used_as_feature':False,
      'cached_reconstruction':{'november_matches_allseason_max_abs':float(np.max(np.abs(candidate[winter]-allseason[winter]))),'july_bitwise_v3':bool(np.array_equal(candidate[~winter],base[~winter])),'outside_gate_bitwise_unchanged':bool(np.array_equal(candidate[~gate],base[~gate]))},
      'local':{'full_rmse':score(candidate,np.ones(len(y),bool)),'baseline_full_rmse':score(base,np.ones(len(y),bool)),'late_november':{'baseline':score(base,late&winter),'candidate':score(candidate,late&winter)},'late_july_unchanged':score(candidate,late&~winter)},
      'day_bootstrap':{'late_november_sse_delta':float(daily.sum()),'ci95':[float(x) for x in np.quantile(boot,[.025,.975])],'probability_improvement':float(np.mean(boot<0))},
      'strict_forward_preliminary':{'status':'raw missing-arrival comparison; awaiting complete forward V3','gate_rows':int(fg.sum()),'baseline_rmse':float(np.sqrt(np.mean((fy[fg]-fo[fg])**2))),'candidate_rmse':float(np.sqrt(np.mean((fy[fg]-fp[fg])**2))),'sse_delta':fsse(fp)-fsse(fo)}}
    assert report['cached_reconstruction']['november_matches_allseason_max_abs']<1e-10 and report['cached_reconstruction']['outside_gate_bitwise_unchanged']
    (ROOT/'docs/autoresearch_identity_winter_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
