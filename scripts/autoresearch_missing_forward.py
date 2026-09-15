"""Forward-check and finalize the frozen weighted-d6 non-Rome missing expert."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TIME, TARGET, align
from missing_v2 import ROOT, OUT, SCHED, load_validation, load_training
from moe_finalize import load_final_data

ALPHA = .4425727562679028
PARAMS = dict(iterations=1400, depth=6, learning_rate=.04, l2_leaf_reg=40,
              random_seed=20260915, thread_count=3, verbose=False,
              loss_function='RMSE', one_hot_max_size=20, max_ctr_complexity=1,
              allow_writing_files=False)


def arrival(x, rows):
    a = pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
    cols = [c for c in a if c.startswith(('arr_stand_', 'arr_runway_'))]
    joined = rows[[ID]].merge(a[[ID]+cols], on=ID, how='left', validate='one_to_one')
    x[cols] = joined[cols].fillna(-999999).astype('float32')
    return cols


def train_predict(x, rows, fit, predict, path):
    y = rows[TARGET].to_numpy(float); schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy(); reference = np.where(rome, schedule, 900.)
    cats = x.select_dtypes('object').columns.tolist()
    model = CatBoostRegressor(**PARAMS)
    model.fit(x.loc[fit], (y-reference)[fit], cat_features=cats,
              sample_weight=np.where(rows.loc[fit, 'missing_aobt'], 8., 1.))
    model.save_model(str(path))
    return np.maximum(model.predict(x.loc[predict], thread_count=3)+reference[predict], 0)


def forward(month, meta, missing):
    cutoff = f'2025-{month:02}-01'
    x, rows = load_training(meta, missing, cutoff=cutoff); arrival(x, rows)
    validation = rows[TIME].dt.month.eq(month).to_numpy() & rows.missing_aobt.to_numpy()
    schedule=x[SCHED].to_numpy(float); rome=x.ADEP_mvt.eq('LIRF').to_numpy(); tail=rome&(schedule>=24000)
    fit = rows[TIME].lt(pd.Timestamp(cutoff, tz='UTC')).to_numpy() & ~tail
    raw = train_predict(x, rows, fit, validation, OUT/f'autoresearch_missing_weighted_forward_{month:02}.cbm')
    ids=rows.loc[validation,ID]; base=align(pd.read_parquet(OUT/f'forward_v3_2025-{month:02}_missing_arrival.parquet'),ids,'fold baseline').prediction.to_numpy(float)
    y=rows.loc[validation,TARGET].to_numpy(float); gate=rows.loc[validation,'ADEP_mvt'].ne('LIRF').to_numpy()
    candidate=base.copy();candidate[gate]+=ALPHA*(raw[gate]-base[gate])
    sse=lambda p:float(np.sum((y[gate]-p[gate])**2))
    return {'train_rows':int(fit.sum()),'gate_rows':int(gate.sum()),'baseline_sse':sse(base),'candidate_sse':sse(candidate),
            'sse_delta':sse(candidate)-sse(base),'baseline_rmse':float(np.sqrt(np.mean((y[gate]-base[gate])**2))),
            'candidate_rmse':float(np.sqrt(np.mean((y[gate]-candidate[gate])**2)))}


def finalize():
    x,rows=load_final_data();cols=arrival(x,rows)
    training=rows.source.eq('training').to_numpy(); ranking=rows.source.eq('ranking').to_numpy()&rows.missing_aobt.to_numpy()
    schedule=x[SCHED].to_numpy(float);rome=x.ADEP_mvt.eq('LIRF').to_numpy();fit=training&~(rome&(schedule>=24000))
    raw=train_predict(x,rows,fit,ranking,OUT/'autoresearch_missing_weighted_d6_final.cbm')
    raw_out=rows.loc[ranking,[ID]].copy();raw_out['prediction']=raw;raw_out.to_parquet(OUT/'autoresearch_missing_weighted_d6_raw_ranking.parquet',index=False)
    base=pd.read_parquet(ROOT/'submissions/elegant-alligator_v3.parquet');out=base.copy()
    pos=pd.Index(base[ID]).get_indexer(raw_out[ID]);gate=rows.loc[ranking,'ADEP_mvt'].ne('LIRF').to_numpy();assert (pos>=0).all()
    out.loc[pos[gate],TARGET] += ALPHA*(raw[gate]-base.loc[pos[gate],TARGET].to_numpy(float))
    assert np.array_equal(out.loc[np.setdiff1d(np.arange(len(out)),pos[gate]),TARGET].to_numpy(),base.loc[np.setdiff1d(np.arange(len(out)),pos[gate]),TARGET].to_numpy())
    path=ROOT/'submissions/autoresearch_v3_missing_weighted.parquet';out.to_parquet(path,index=False)
    validation=OUT/'autoresearch_missing_weighted_d6_validation.parquet'
    return {'fit_rows':int(fit.sum()),'arrival_features':cols,'raw_ranking_rows':len(raw_out),'changed_nonrome_rows':int(gate.sum()),
            'validation':str(validation.relative_to(ROOT)).replace('\\','/'),'output':str(path.relative_to(ROOT)).replace('\\','/')}


def main():
    _,_,meta,missing=load_validation();report={'alpha':ALPHA,'params':PARAMS,'folds':{}}
    for month in (7,11):
        if not (OUT/f'forward_v3_2025-{month:02}_missing_arrival.parquet').exists(): raise FileNotFoundError(f'forward month {month} baseline pending')
        report['folds'][str(month)]=forward(month,meta,missing)
    report['eligible']=all(x['sse_delta']<0 for x in report['folds'].values())
    if report['eligible']:report['final']=finalize()
    (ROOT/'docs/autoresearch_missing_forward.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=3):main()
