"""Small independent CPU check of timestamp proxies and exceptional taxi tails."""
import argparse
import json
import gc

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from audit_signals import DATA, ROOT, TIMES, read


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def features(df):
    cat = ['ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt', 'STAND_mvt',
           'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt']
    x = df[cat + ['dt_' + t for t in TIMES]].copy()
    for col in cat:
        x[col] = x[col].astype('object').fillna('missing').astype(str)
    x['flight_prefix'] = df.FLIGHT_mvt.astype('object').fillna('missing').str.extract(r'^([A-Za-z]+)', expand=False).fillna('missing')
    cat.append('flight_prefix')
    x['flight_length'] = df.FLIGHT_mvt.astype('object').fillna('').str.len()
    for col in ['MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt'] + TIMES[:4]:
        for unit in ['hour', 'dayofweek', 'second']:
            x[col + '_' + unit] = getattr(df[col].dt, unit)
    x['month'] = df.MVT_TIME_UTC_mvt.dt.month
    x['aobt_lobt'] = df.dt_LOBT_flt - df.dt_AOBT_3_flt
    x['iobt_lobt'] = df.dt_LOBT_flt - df.dt_IOBT_flt
    for col in x.columns.difference(cat):
        x[col] = x[col].fillna(-999999).astype('float32')
    return x, cat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=450)
    args = parser.parse_args()
    frames = []
    for p in sorted(DATA.glob('training*.parquet')):
        if '2025-12-01_2026' in p.name:
            continue
        df = read(p)
        frames.append(df.loc[df.AOBT_3_flt.isna()].copy())
        del df
        gc.collect()
    train = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    valid = train.MVT_TIME_UTC_mvt.dt.month.isin([7, 11])
    idx_train = np.flatnonzero(~valid)
    assert not np.intersect1d(idx_train, np.flatnonzero(valid)).size
    x, cat = features(train)
    y = train.TAXITIME_SEC_mvt.to_numpy(dtype=float)
    airport_mean = train.loc[~valid].groupby('ADEP_mvt', observed=True).TAXITIME_SEC_mvt.mean()
    proxy = train.dt_AOBT_3_flt.fillna(train.ADEP_mvt.map(airport_mean).astype(float)).to_numpy(dtype=float)
    # Preserve extrapolation on exceptional scheduled-time records in Rome.
    rome_missing = (train.ADEP_mvt == 'LIRF') & train.AOBT_3_flt.isna()
    proxy[rome_missing] = train.loc[rome_missing, 'dt_SCHED_TIME_UTC_mvt'].clip(lower=0)
    residual = y - proxy
    model = CatBoostRegressor(iterations=args.iterations, depth=5, learning_rate=.04, loss_function='RMSE',
                              thread_count=4, random_seed=2026, verbose=100, l2_leaf_reg=25)
    model.fit(x.iloc[idx_train], residual[idx_train], cat_features=cat,
              eval_set=(x.loc[valid], residual[valid]), early_stopping_rounds=100)
    pred = proxy[valid] + model.predict(x.loc[valid])
    out = train.loc[valid, ['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']].copy()
    out['prediction'] = pred
    out['proxy'] = proxy[valid]
    out['dt_sched'] = train.loc[valid, 'dt_SCHED_TIME_UTC_mvt'].to_numpy()
    out['prediction_schedule_rule'] = out.prediction
    rule = (out.ADEP_mvt == 'LIRF') & out.dt_sched.between(24000, 48000)
    out.loc[rule, 'prediction_schedule_rule'] = out.loc[rule, 'dt_sched']
    out.to_parquet(ROOT / 'data/proxy_validation.parquet', index=False)
    result = {'train_rows': len(idx_train), 'valid_rows': int(valid.sum()), 'rmse': rmse(y[valid], pred),
        'proxy_rmse': rmse(y[valid], proxy[valid]), 'best_iteration': model.best_iteration_,
        'airports': {a: {'rmse': rmse(g.TAXITIME_SEC_mvt, g.prediction), 'proxy_rmse': rmse(g.TAXITIME_SEC_mvt, g.proxy)} for a, g in out.groupby('ADEP_mvt', observed=True)},
        'importance': dict(zip(x.columns, model.feature_importances_.tolist()))}
    result['schedule_rule_rmse'] = rmse(out.TAXITIME_SEC_mvt, out.prediction_schedule_rule)
    result['schedule_rule_rows'] = int(rule.sum())
    result['tail_rmse'] = rmse(out.loc[out.TAXITIME_SEC_mvt > 5000, 'TAXITIME_SEC_mvt'], out.loc[out.TAXITIME_SEC_mvt > 5000, 'prediction'])
    (ROOT / 'docs/proxy_baseline.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    model.save_model(str(ROOT / 'data/proxy_baseline.cbm'))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
