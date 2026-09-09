"""One cap-12000 missing expert ablation with validated arrival context."""
import ctypes, json, os
import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits
from ensemble import ID, TARGET, TIME, validation_labels
from missing_v2 import OUT, PARAMS, ROOT, SCHED, load_training, load_validation, rmse


def scalar(base, candidate, y, mask):
    d=candidate[mask]-base[mask]
    return float(np.clip(np.dot(d,y[mask]-base[mask])/max(np.dot(d,d),1),0,1))


def main():
    if os.name=='nt': assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x8000)
    _, valid_missing, meta, missing = load_validation()
    x, rows = load_training(meta, missing)
    arrival=pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
    cols9=[c for c in arrival if c.startswith(('arr_stand_','arr_runway_'))]; assert len(cols9)==9
    aligned=rows[[ID]].merge(arrival[[ID]+cols9],on=ID,how='left',validate='one_to_one')
    x=pd.concat([x.reset_index(drop=True),aligned[cols9].fillna(-999999).astype('float32')],axis=1)
    y=rows[TARGET].to_numpy(float); month=rows[TIME].dt.month.to_numpy()
    training=~np.isin(month,[7,11]); validation=~training
    order=pd.Index(rows.loc[validation,ID]).get_indexer(valid_missing[ID]); assert (order>=0).all()
    schedule=x[SCHED].to_numpy(float); rome=x.ADEP_mvt.eq('LIRF').to_numpy(); tail=rome&(schedule>=24000)
    identity=rome&(np.abs(y-schedule)<=6); normal_train=training&~tail&~identity&(y<12000)
    weight=np.where(rows.missing_aobt,8.,1.); cats=list(x.select_dtypes('object').columns)
    normal=CatBoostRegressor(**{**PARAMS,'thread_count':3},loss_function='RMSE')
    normal.fit(x.loc[normal_train],y[normal_train],cat_features=cats,sample_weight=weight[normal_train])
    normal.save_model(str(OUT/'moe_missing_arrival_v3_normal_cap12000.cbm'))
    classifier=CatBoostClassifier().load_model(str(OUT/'missing_v2_identity.cbm'))
    cls_cols=classifier.feature_names_
    raw=normal.predict(x.loc[validation],thread_count=3)
    prob=classifier.predict_proba(x.loc[validation,cls_cols],thread_count=3)[:,1]
    prob[~rome[validation]|(schedule[validation]<=0)]=0
    raw=np.maximum(prob*schedule[validation]+(1-prob)*raw,0)[order]
    old=pd.read_parquet(OUT/'moe_normal_cap12000_raw_missing_validation.parquet').set_index(ID).loc[valid_missing[ID],'prediction'].to_numpy()
    v2=pd.read_parquet(OUT/'v2_validation.parquet'); full=validation_labels(OUT/'rows.parquet',v2[ID].to_numpy()); full['prediction']=v2.prediction
    base_full=full.prediction.to_numpy(); pos=pd.Index(full[ID]).get_indexer(valid_missing[ID]); base=base_full[pos]
    vrome=valid_missing.ADEP_mvt.eq('LIRF').to_numpy(); vsched=valid_missing[SCHED].to_numpy(); eligible=~(vrome&(vsched>=24000))
    fixed_alpha=np.where(vrome,.4224120030856563,1.0)
    candidate=base.copy(); candidate[eligible]+=fixed_alpha[eligible]*(raw[eligible]-old[eligible])
    fixed_full=base_full.copy(); fixed_full[pos]=candidate
    vy=valid_missing[TARGET].to_numpy(); vday=valid_missing[TIME].dt.day.to_numpy(); vmonth=valid_missing[TIME].dt.month.to_numpy()
    even=eligible&(vday%2==0); beta=scalar(base,candidate,vy,even)
    shrunk=base+beta*(candidate-base); shrunk[~eligible]=base[~eligible]
    shrunk_full=base_full.copy(); shrunk_full[pos]=shrunk
    checks={}
    for label,mask in [('odd_all',eligible&(vday%2==1)),('odd_july',eligible&(vday%2==1)&(vmonth==7)),('odd_november',eligible&(vday%2==1)&(vmonth==11)),('july_all',eligible&(vmonth==7)),('november_all',eligible&(vmonth==11))]:
        checks[label]={'rows':int(mask.sum()),'baseline_rmse':rmse(vy[mask],base[mask]),'fixed_rmse':rmse(vy[mask],candidate[mask]),'even_shrink_rmse':rmse(vy[mask],shrunk[mask])}
    report={'excluded_months':[7,11,12],'december_scored':False,'arrival_features':cols9,
      'baseline_full_rmse':rmse(full[TARGET],base_full),'fixed_full_rmse':rmse(full[TARGET],fixed_full),
      'even_fitted_shrink':beta,'shrunk_full_rmse':rmse(full[TARGET],shrunk_full),'checks':checks,
      'protected_tail_rows':int((~eligible).sum()),'selected':'pending_parent_review'}
    valid_missing[[ID]].assign(prediction=raw).to_parquet(OUT/'moe_missing_arrival_v3_raw_validation.parquet',index=False)
    pd.DataFrame({ID:full[ID],'prediction':fixed_full}).to_parquet(OUT/'moe_missing_arrival_v3_fixed_complete_validation.parquet',index=False)
    pd.DataFrame({ID:full[ID],'prediction':shrunk_full}).to_parquet(OUT/'moe_missing_arrival_v3_shrunk_complete_validation.parquet',index=False)
    (ROOT/'docs/research_moe_missing_arrival_v3.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    pa.set_cpu_count(3);pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):main()
