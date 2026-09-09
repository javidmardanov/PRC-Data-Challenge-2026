"""Final all-2025 fit and ranking inference for the frozen missing-v2 MoE recipe."""
import ctypes
import gc
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, TIME, frozen_predictions
from missing_v2 import OUT, PARAMS, ROOT, SCHED, SEED


def load_final_data():
    meta = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'source', TARGET])
    small = pd.read_parquet(OUT/'features.parquet', columns=['ADEP_mvt', SCHED, 'mvt_minus_AOBT_3_flt'])
    assert len(meta) == len(small)
    missing = small.mvt_minus_AOBT_3_flt.lt(-100000).to_numpy()
    training = meta.source.eq('training').to_numpy()
    ranking = meta.source.eq('ranking').to_numpy()
    rng = np.random.default_rng(SEED)
    known = training & ~missing
    selected = np.concatenate([rng.choice(np.flatnonzero(known & small.ADEP_mvt.eq(airport).to_numpy()),
        min(count, int((known & small.ADEP_mvt.eq(airport).to_numpy()).sum())), replace=False)
        for airport, count in [('LIRF', 120000)]])
    other = known & small.ADEP_mvt.ne('LIRF').to_numpy()
    selected = np.concatenate([selected, rng.choice(np.flatnonzero(other), min(80000, int(other.sum())), replace=False)])
    keep = (training & missing) | (ranking & missing)
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
    rows['missing_aobt'] = missing[keep]
    x[ID] = rows[ID].to_numpy()
    queue_cols = [c for c in pq.read_schema(ROOT/'data/processed/queue_features.parquet').names if not c.endswith('_at_aobt')]
    x = x.merge(pd.read_parquet(ROOT/'data/processed/queue_features.parquet', columns=queue_cols),
                on=ID, how='left', validate='one_to_one').drop(columns=ID)
    keys = x[['ADEP_mvt']].copy()
    keys['hour'] = rows[TIME].dt.floor('h')
    weather = pd.read_parquet(ROOT/'data/processed/weather_hourly.parquet')
    joined = keys.merge(weather, on=['ADEP_mvt', 'hour'], how='left', validate='many_to_one')
    for col in joined.columns.difference(['ADEP_mvt', 'hour']):
        x[col] = joined[col].to_numpy()
    tfm = pd.concat([pd.read_parquet(ROOT/'data/processed/timesfm_hourly_dev.parquet'),
                     pd.read_parquet(ROOT/'data/processed/timesfm_hourly_final.parquet')], ignore_index=True)
    tfm = tfm.drop_duplicates(['ADEP_mvt', 'hour'], keep='last')
    joined = keys.merge(tfm, on=['ADEP_mvt', 'hour'], how='left', validate='many_to_one')
    for col in joined.columns.difference(['ADEP_mvt', 'hour']):
        x[col] = joined[col].to_numpy()
    x['stand_runway'] = x.airport_stand+'_'+x.RUNWAY_mvt
    x['airport_flight_prefix'] = x.ADEP_mvt+'_'+x.flight_prefix
    x['flight_prefix_length'] = x.flight_prefix.str.len()
    for col in x.select_dtypes('number'):
        x[col] = x[col].replace([np.inf, -np.inf], np.nan).fillna(-999999).astype('float32')
    assert len(rows) == len(x) and rows.loc[rows.source.eq('training'), TIME].dt.year.eq(2025).all()
    del pieces, weather, tfm, joined
    gc.collect()
    return x, rows


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    x, rows = load_final_data()
    training = rows.source.eq('training').to_numpy()
    ranking = rows.source.eq('ranking').to_numpy() & rows.missing_aobt.to_numpy()
    y = rows[TARGET].to_numpy(float)
    schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    identity = rome & (np.abs(y-schedule) <= 6)
    normal_train = training & ~tail & ~identity & (y < 6000)
    cats = list(x.select_dtypes('object').columns)
    weight = np.where(rows.missing_aobt, 8., 1.)
    normal = CatBoostRegressor(**{**PARAMS, 'thread_count': 3}, loss_function='RMSE')
    normal.fit(x.loc[normal_train], y[normal_train], cat_features=cats, sample_weight=weight[normal_train])
    normal.save_model(str(OUT/'moe_missing_v2_normal_final.cbm'))
    classifier = CatBoostClassifier(**{**PARAMS, 'iterations': 400, 'depth': 5, 'thread_count': 3}, loss_function='Logloss')
    cls_train = training & rome & ~tail
    cls_weight = weight*np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25)
    classifier.fit(x.loc[cls_train], identity[cls_train].astype(int), cat_features=cats,
                   sample_weight=cls_weight[cls_train])
    classifier.save_model(str(OUT/'moe_missing_v2_identity_final.cbm'))
    raw = normal.predict(x.loc[ranking], thread_count=3)
    probability = classifier.predict_proba(x.loc[ranking], thread_count=3)[:, 1]
    rank_rome, rank_schedule = rome[ranking], schedule[ranking]
    probability[~rank_rome | (rank_schedule <= 0)] = 0
    raw = np.maximum(probability*rank_schedule+(1-probability)*raw, 0)
    raw_out = rows.loc[ranking, [ID]].copy()
    raw_out['prediction'] = raw
    raw_out.to_parquet(OUT/'moe_missing_v2_raw_missing_ranking.parquet', index=False)

    config = json.loads((OUT/'v1_ensemble.json').read_text())
    template = pd.read_parquet(ROOT/'data/raw/prc-2026-datasets/submitting.parquet', columns=[ID])
    ids, complete = frozen_predictions(config,
        [OUT/'tfm_queue_final_predictions.parquet', OUT/'no_tfm_final_predictions.parquet', OUT/'lgbm_tfm_ranking.parquet'],
        specialist=OUT/'missing_ranking.parquet', ids=template[ID].to_numpy(),
        rome=OUT/'lirf_identity_expert_ranking.parquet')
    pos = pd.Index(ids).get_indexer(raw_out[ID])
    assert (pos >= 0).all()
    apply = ~(rank_rome & (rank_schedule >= 24000))
    alpha = np.where(rank_rome, 0.4224120030856563, 1.0)
    complete[pos[apply]] += alpha[apply]*(raw[apply]-complete[pos[apply]])
    complete_out = pd.DataFrame({ID: ids, 'prediction': complete})
    assert len(complete_out) == 344841 and complete_out[ID].is_unique and np.isfinite(complete).all()
    complete_out.to_parquet(OUT/'moe_missing_v2_complete_ranking.parquet', index=False)
    report = {'selected': 'incumbent_d5_l30_i400', 'selection_frozen': True,
        'fit_years': [2025], 'december_scored': False, 'normal_training_rows': int(normal_train.sum()),
        'classifier_training_rows': int(cls_train.sum()), 'ranking_missing_rows': int(ranking.sum()),
        'ranking_changed_rows': int(apply.sum()), 'ranking_tail_fallback_rows': int((~apply).sum()),
        'alphas': {'Rome': 0.4224120030856563, 'other': 1.0},
        'raw_output_semantics': 'All missing-AOBT ranking rows contain raw expert outputs; config applies v1 fallback to LIRF schedule>=24000.',
        'raw_output': str(OUT/'moe_missing_v2_raw_missing_ranking.parquet'),
        'complete_output': str(OUT/'moe_missing_v2_complete_ranking.parquet'), 'cpu_threads': 3}
    (ROOT/'docs'/'research_moe_final.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):
        main()
