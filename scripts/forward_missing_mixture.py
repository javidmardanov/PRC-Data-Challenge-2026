"""Strictly chronological fits for the two frozen cap12000 missing experts."""
import argparse
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits
from missing_v2 import load_training, OUT, ROOT, SCHED, PARAMS
from ensemble import ID, TIME, TARGET


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--month', required=True, choices=['2025-07', '2025-11'])
    args = p.parse_args()
    cutoff = pd.Timestamp(args.month+'-01', tz='UTC')
    meta = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'source', 'ADEP_mvt'])
    proxy = pd.read_parquet(OUT/'features.parquet', columns=['mvt_minus_AOBT_3_flt'])
    missing = proxy.iloc[:, 0].lt(-100000).to_numpy()
    x, rows = load_training(meta, missing, cutoff=args.month+'-01')
    train = rows[TIME].lt(cutoff).to_numpy()
    valid = ~train
    if not (rows.loc[valid, TIME].dt.strftime('%Y-%m') == args.month).all():
        raise ValueError('Invalid chronological split')
    y = rows[TARGET].to_numpy(float)
    schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    identity = rome & (np.abs(y-schedule) <= 6)
    weight = np.where(rows.missing_aobt, 8., 1.)
    cats = x.select_dtypes('object').columns.tolist()
    clsfit = train & rome & ~tail
    prefix = 'forward_v3_'+args.month
    classifier = CatBoostClassifier(**{**PARAMS, 'iterations': 400, 'depth': 5}, loss_function='Logloss')
    classifier.fit(x.loc[clsfit], identity[clsfit].astype(int), cat_features=cats,
                   sample_weight=(weight*np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25))[clsfit])
    classifier.save_model(str(OUT/f'{prefix}_missing_identity.cbm'))
    prob = classifier.predict_proba(x.loc[valid], thread_count=3)[:, 1]
    prob[~rome[valid] | (schedule[valid] <= 0)] = 0
    normalfit = train & ~tail & ~identity & (y < 12000)
    for name in ['missing_v2', 'missing_arrival']:
        if name == 'missing_arrival':
            arrival = pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
            cols = [c for c in arrival if c.startswith(('arr_stand_', 'arr_runway_'))]
            extra = rows[[ID]].merge(arrival[[ID]+cols], on=ID, how='left', validate='one_to_one')
            x[cols] = extra[cols].fillna(-999999).astype('float32')
        model = CatBoostRegressor(**PARAMS, loss_function='RMSE')
        model.fit(x.loc[normalfit], y[normalfit], cat_features=cats, sample_weight=weight[normalfit])
        model.save_model(str(OUT/f'{prefix}_{name}.cbm'))
        raw = np.maximum(prob*schedule[valid]+(1-prob)*model.predict(x.loc[valid], thread_count=3), 0)
        rows.loc[valid, [ID]].assign(prediction=raw).to_parquet(OUT/f'{prefix}_{name}.parquet', index=False)
        print(args.month, name, 'completed', int(valid.sum()), flush=True)
    info = {'month': args.month, 'train_max': str(rows.loc[train, TIME].max()),
            'classifier_rows': int(clsfit.sum()), 'normal_rows': int(normalfit.sum()),
            'validation_rows': int(valid.sum()), 'validation_used_for_stopping': False}
    (OUT/f'{prefix}_missing_mixture.json').write_text(json.dumps(info, indent=2)+'\n')


if __name__ == '__main__':
    with threadpool_limits(limits=3): main()
