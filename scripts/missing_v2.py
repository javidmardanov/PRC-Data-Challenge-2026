"""Two bounded CPU experiments for missing AOBT; July/November validation only."""
# SPDX-License-Identifier: GPL-3.0-only
import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ROOT, ID, TIME, TARGET, frozen_predictions, validation_labels

OUT = ROOT / 'artifacts'
SCHED = 'mvt_minus_SCHED_TIME_UTC_mvt'
SEED = 20260909
PARAMS = dict(iterations=600, depth=6, learning_rate=.06, l2_leaf_reg=30,
              thread_count=3, random_seed=SEED, verbose=False,
              one_hot_max_size=20, max_ctr_complexity=1, allow_writing_files=False)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def summarize(d, pred):
    error = np.asarray(pred)-d[TARGET].to_numpy()
    return dict(rows=len(d), rmse=float(np.sqrt(np.mean(error**2))),
                bias=float(error.mean()), sse=float(np.dot(error, error)))


def load_validation():
    config = json.loads((OUT/'v1_ensemble.json').read_text())
    ids, prediction = frozen_predictions(config, config['models'],
        OUT/'missing_validation.parquet', rome=OUT/'lirf_identity_expert_validation.parquet')
    full = validation_labels(OUT/'rows.parquet', ids)
    full['prediction'] = prediction
    meta = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'source'])
    small = pd.read_parquet(OUT/'features.parquet', columns=['ADEP_mvt', SCHED, 'mvt_minus_AOBT_3_flt'])
    assert len(meta) == len(small)
    meta['ADEP_mvt'] = small.ADEP_mvt.to_numpy()
    small[ID] = meta[ID].to_numpy()
    missing = small.mvt_minus_AOBT_3_flt.lt(-100000)
    valid = full.merge(small.loc[missing], on=ID, validate='one_to_one')
    assert len(valid) == 5294
    return full, valid, meta, missing.to_numpy()


def analyze(full, valid):
    report = {'december_read': False, 'training_excluded_months': [7, 11, 12],
        'v1': summarize(full, full.prediction), 'missing_v1': summarize(valid, valid.prediction)}
    report['missing_airport_month'] = {f'{a}/{int(m)}': summarize(g, g.prediction)
        for (a, m), g in valid.groupby(['ADEP_mvt', valid[TIME].dt.month])}
    error = (valid.prediction-valid[TARGET])**2
    report['missing_sse_concentration'] = {str(n): float(error.nlargest(n).sum()/error.sum()) for n in [1, 5, 10, 25, 100, 500]}
    report['missing_schedule_regimes'] = {f'{a}/{b}': summarize(g, g.prediction)
        for (a, b), g in valid.groupby(['ADEP_mvt', pd.cut(valid[SCHED], [-np.inf, 0, 1200, 3600, 6000, 24000, np.inf])], observed=True)}
    report['missing_target_regimes'] = {str(b): summarize(g, g.prediction)
        for b, g in valid.groupby(pd.cut(valid[TARGET], [-np.inf, 600, 1200, 2400, 6000, 24000, np.inf]), observed=True)}
    return report


def load_training(meta, missing):
    month = meta[TIME].dt.month
    development = meta.source.eq('training') & ~month.isin([7, 11, 12])
    valid = meta.source.eq('training') & month.isin([7, 11]) & missing
    rng = np.random.default_rng(SEED)
    known = development & ~missing
    selected = np.concatenate([rng.choice(np.flatnonzero(known & mask),
        min(count, int((known & mask).sum())), replace=False)
        for mask, count in [(meta.ADEP_mvt.eq('LIRF'), 120000),
                            (meta.ADEP_mvt.ne('LIRF'), 80000)]])
    keep = (development & missing).to_numpy() | valid.to_numpy()
    keep[selected] = True
    names = pq.read_schema(OUT/'features.parquet').names
    columns = [c for c in names if '_flt' not in c and not c.startswith('nm_')
               and c not in ['FLIGHT_mvt', 'callsign_prefix', 'diverted']]
    pieces, start = [], 0
    for batch in pq.ParquetFile(OUT/'features.parquet').iter_batches(batch_size=65536, columns=columns):
        mask = keep[start:start+len(batch)]
        if mask.any():
            pieces.append(batch.filter(mask).to_pandas())
        start += len(batch)
    x = pd.concat(pieces, ignore_index=True)
    rows = meta.loc[keep].reset_index(drop=True)
    labels = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TARGET],
        filters=[(TIME, '<', pd.Timestamp('2025-12-01', tz='UTC')), (ID, 'in', rows[ID].tolist())])
    rows = rows.merge(labels, on=ID, validate='one_to_one')
    rows['missing_aobt'] = missing[keep]
    assert len(rows) == len(x) and not rows[TIME].dt.month.eq(12).any()
    x[ID] = rows[ID].to_numpy()
    queue_cols = [c for c in pq.read_schema(ROOT/'data/processed/queue_features.parquet').names if not c.endswith('_at_aobt')]
    queues = pd.read_parquet(ROOT/'data/processed/queue_features.parquet', columns=queue_cols)
    x = x.merge(queues, on=ID, how='left', validate='one_to_one').drop(columns=ID)
    del pieces, queues, labels
    keys = rows[['ADEP_mvt']].copy() if 'ADEP_mvt' in rows else x[['ADEP_mvt']].copy()
    keys['hour'] = rows[TIME].dt.floor('h')
    for path in ['weather_hourly.parquet', 'timesfm_hourly_dev.parquet']:
        extra = pd.read_parquet(ROOT/'data/processed'/path)
        joined = keys.merge(extra, on=['ADEP_mvt', 'hour'], how='left', validate='many_to_one')
        for col in joined.columns.difference(['ADEP_mvt', 'hour']):
            x[col] = joined[col].to_numpy()
    x['stand_runway'] = x.airport_stand+'_'+x.RUNWAY_mvt
    x['airport_flight_prefix'] = x.ADEP_mvt+'_'+x.flight_prefix
    x['flight_prefix_length'] = x.flight_prefix.str.len()
    for col in x.select_dtypes('number'):
        x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
    assert all('_flt' not in c for c in x) and TARGET not in x
    gc.collect()
    return x, rows


def evaluate(full, valid, raw, name, report):
    raw = np.maximum(raw, 0)
    d = raw-valid.prediction.to_numpy()
    y = valid[TARGET].to_numpy()
    alpha = float(np.clip(np.dot(d, y-valid.prediction)/max(np.dot(d, d), 1), 0, 1))
    combined = valid.prediction.to_numpy()+alpha*d
    full_pred = full.prediction.copy()
    positions = pd.Index(full[ID]).get_indexer(valid[ID])
    full_pred.iloc[positions] = combined
    result = {'alpha': alpha, 'raw_missing': summarize(valid, raw),
        'blended_missing': summarize(valid, combined), 'blended_full': summarize(full, full_pred),
        'months': {str(m): {'raw_missing': rmse(y[mask], raw[mask]),
            'blended_missing': rmse(y[mask], combined[mask]),
            'blended_full': rmse(full.loc[full[TIME].dt.month.eq(m), TARGET], full_pred.loc[full[TIME].dt.month.eq(m)])}
            for m in [7, 11] for mask in [valid[TIME].dt.month.eq(m).to_numpy()]},
        'airports': {a: {'raw_missing': rmse(y[mask], raw[mask]),
            'blended_missing': rmse(y[mask], combined[mask])}
            for a in valid.ADEP_mvt.unique() for mask in [valid.ADEP_mvt.eq(a).to_numpy()]}}
    regional, alphas = valid.prediction.to_numpy().copy(), {}
    for region, mask in [('Rome', valid.ADEP_mvt.eq('LIRF').to_numpy()),
                         ('other', valid.ADEP_mvt.ne('LIRF').to_numpy())]:
        a = float(np.clip(np.dot(d[mask], y[mask]-regional[mask])/max(np.dot(d[mask], d[mask]), 1), 0, 1))
        regional[mask] += a*d[mask]
        alphas[region] = a
    regional_full = full.prediction.copy()
    regional_full.iloc[positions] = regional
    result.update(regional_alphas=alphas, regional_missing=summarize(valid, regional),
        regional_full=summarize(full, regional_full),
        regional_months={str(m): {'missing': rmse(y[mask], regional[mask]),
            'full': rmse(full.loc[full[TIME].dt.month.eq(m), TARGET], regional_full.loc[full[TIME].dt.month.eq(m)])}
            for m in [7, 11] for mask in [valid[TIME].dt.month.eq(m).to_numpy()]})
    prediction = valid[[ID]].copy()
    prediction['prediction'] = raw
    assert prediction[ID].is_unique and np.isfinite(raw).all()
    prediction.to_parquet(OUT/f'missing_v2_{name}_validation.parquet', index=False)
    report.setdefault('candidates', {})[name] = result
    print(name, json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        import ctypes
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    full, valid, metadata, missing = load_validation()
    path = ROOT/'docs/missing_v2.json'
    if args.evaluate_only:
        report = json.loads(path.read_text())
        for name in ['direct', 'mixture']:
            pred = pd.read_parquet(OUT/f'missing_v2_{name}_validation.parquet').set_index(ID)
            evaluate(full, valid, pred.loc[valid[ID], 'prediction'].to_numpy(), name, report)
        report['regional_routing'] = 'Two closed-form validation alphas after complete v1: Rome vs other missing-AOBT rows; never combine blindly before old Rome routing.'
        path.write_text(json.dumps(report, indent=2))
        return
    report = analyze(full, valid)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['v1', 'missing_v1', 'missing_sse_concentration']}), flush=True)
    if args.analyze_only:
        return
    x, rows = load_training(metadata, missing)
    y = rows[TARGET].to_numpy(dtype=float)
    training = ~rows[TIME].dt.month.isin([7, 11]).to_numpy()
    validation = ~training
    order = pd.Index(rows.loc[validation, ID]).get_indexer(valid[ID])
    assert (order >= 0).all()
    schedule = x[SCHED].to_numpy(dtype=float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    # Long Rome timestamp anomalies retain the frozen v1 specialist.
    tail = rome & (schedule >= 24000)
    train = training & ~tail
    cats = list(x.select_dtypes('object').columns)
    weight = np.where(rows.missing_aobt, 8., 1.)
    report.update(feature_columns=list(x.columns), params=PARAMS, train_rows=int(training.sum()),
        train_missing_rows=int(rows.loc[training, 'missing_aobt'].sum()),
        sample_seed=SEED, known_sample_rows=200000, known_allocation={'LIRF': 120000, 'other': 80000}, missing_weight=8,
        retained_tail='LIRF schedule delta >=24000: frozen v1 complete missing+Rome pipeline')
    direct = CatBoostRegressor(**PARAMS, loss_function='RMSE')
    direct.fit(x.loc[train], y[train], cat_features=cats, sample_weight=weight[train])
    direct.save_model(str(OUT/'missing_v2_direct.cbm'))
    raw = direct.predict(x.loc[validation], thread_count=3)[order]
    vtail = tail[validation][order]
    raw[vtail] = valid.loc[vtail, 'prediction']
    evaluate(full, valid, raw, 'direct', report)
    path.write_text(json.dumps(report, indent=2))
    del direct
    # Second candidate explicitly separates schedule-as-block Rome records.
    identity = rome & (np.abs(y-schedule) <= 6)
    normal_train = train & ~identity & (y < 6000)
    report['normal_training_rows'] = int(normal_train.sum())
    report['identity_training_rows'] = int((train & rome).sum())
    normal = CatBoostRegressor(**PARAMS, loss_function='RMSE')
    normal.fit(x.loc[normal_train], y[normal_train], cat_features=cats, sample_weight=weight[normal_train])
    normal.save_model(str(OUT/'missing_v2_normal.cbm'))
    raw = normal.predict(x.loc[validation], thread_count=3)
    cls = CatBoostClassifier(**{**PARAMS, 'iterations': 400, 'depth': 5}, loss_function='Logloss')
    cls_weight = weight*np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25)
    cls.fit(x.loc[train & rome], identity[train & rome].astype(int), cat_features=cats,
            sample_weight=cls_weight[train & rome])
    cls.save_model(str(OUT/'missing_v2_identity.cbm'))
    probability = cls.predict_proba(x.loc[validation], thread_count=3)[:, 1]
    probability[~rome[validation] | (schedule[validation] <= 0)] = 0
    raw = (probability*schedule[validation]+(1-probability)*raw)[order]
    raw[vtail] = valid.loc[vtail, 'prediction']
    evaluate(full, valid, raw, 'mixture', report)
    report['selected'] = min(report['candidates'], key=lambda k: report['candidates'][k]['blended_full']['rmse'])
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):
        main()
