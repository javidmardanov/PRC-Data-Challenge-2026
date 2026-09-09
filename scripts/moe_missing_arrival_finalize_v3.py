"""Final all-2025 missing-arrival normal expert; reuse frozen v2 identity classifier."""
import ctypes,json,os
import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier,CatBoostRegressor
from threadpoolctl import threadpool_limits
from ensemble import ID,TARGET
from missing_v2 import OUT,PARAMS,ROOT,SCHED
from moe_finalize import load_final_data

def main():
    if os.name=='nt':assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x8000)
    x,rows=load_final_data()
    arrival=pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
    cols9=[c for c in arrival if c.startswith(('arr_stand_','arr_runway_'))];assert len(cols9)==9 and arrival[ID].is_unique
    aligned=rows[[ID]].merge(arrival[[ID]+cols9],on=ID,how='left',validate='one_to_one')
    x=pd.concat([x.reset_index(drop=True),aligned[cols9].fillna(-999999).astype('float32')],axis=1)
    training=rows.source.eq('training').to_numpy();ranking=rows.source.eq('ranking').to_numpy()&rows.missing_aobt.to_numpy()
    y=rows[TARGET].to_numpy(float);schedule=x[SCHED].to_numpy(float);rome=x.ADEP_mvt.eq('LIRF').to_numpy();tail=rome&(schedule>=24000)
    identity=rome&(np.abs(y-schedule)<=6);fit=training&~tail&~identity&(y<12000)
    weight=np.where(rows.missing_aobt,8.,1.);cats=list(x.select_dtypes('object').columns)
    normal=CatBoostRegressor(**{**PARAMS,'thread_count':3},loss_function='RMSE')
    normal.fit(x.loc[fit],y[fit],cat_features=cats,sample_weight=weight[fit]);normal.save_model(str(OUT/'moe_missing_arrival_v3_normal_cap12000_final.cbm'))
    classifier=CatBoostClassifier().load_model(str(OUT/'moe_missing_v2_identity_final.cbm'));cls_cols=classifier.feature_names_
    raw=normal.predict(x.loc[ranking],thread_count=3);prob=classifier.predict_proba(x.loc[ranking,cls_cols],thread_count=3)[:,1]
    rr,rs=rome[ranking],schedule[ranking];prob[~rr|(rs<=0)]=0;raw=np.maximum(prob*rs+(1-prob)*raw,0)
    out=rows.loc[ranking,[ID]].copy();out['prediction']=raw
    destination=OUT/'moe_missing_arrival_v3_raw_missing_ranking.parquet';out.to_parquet(destination,index=False)
    old=pd.read_parquet(OUT/'missing_ranking.parquet',columns=[ID])[ID]
    report={'selected':'missing_arrival_fixed_existing_alphas','selection_frozen':True,'fit_years':[2025],'december_scored':False,
      'normal_training_rows':int(fit.sum()),'arrival_features':cols9,'classifier_reused':'moe_missing_v2_identity_final.cbm',
      'classifier_feature_count':len(cls_cols),'raw_rows':len(out),'exact_missing_id_set':set(out[ID])==set(old),'finite':bool(np.isfinite(raw).all()),
      'protected_ranking_tail_rows':int((rr&(rs>=24000)).sum()),'output':str(destination),
      'apply':'v2 + regional_alpha*(new_raw-v2_cap12000_raw) outside LIRF schedule>=24000; Rome=.4224120030856563, other=1',
      'reproduction_command':"$env:OPENBLAS_NUM_THREADS='1'; & 'C:\\Users\\javid\\anaconda3\\python.exe' scripts/moe_missing_arrival_finalize_v3.py"}
    (ROOT/'docs/research_moe_missing_arrival_final_v3.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    pa.set_cpu_count(3);pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):main()
