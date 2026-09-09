"""Bounded Rome schedule-identity experiment; December labels are never read."""
import gc
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from audit_signals import ROOT

OUT = ROOT/'artifacts'
ID, TARGET, TIME = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt'
SCHEDULE = 'mvt_minus_SCHED_TIME_UTC_mvt'


def load():
    metadata = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'source'], filters=[('ADEP_mvt', '==', 'LIRF')])
    metadata = metadata.loc[metadata[TIME].dt.month.ne(12)].reset_index(drop=True)
    x = pd.read_parquet(OUT/'features.parquet', filters=[('ADEP_mvt', '==', 'LIRF'), ('month', '!=', 12)]).reset_index(drop=True)
    assert len(x) == len(metadata)
    x[ID] = metadata[ID].to_numpy()
    x = x.loc[metadata.source.eq('training')].copy()
    labels = pd.read_parquet(OUT/'rows.parquet', filters=[('ADEP_mvt', '==', 'LIRF'),
        (TIME, '<', pd.Timestamp('2025-12-01', tz='UTC'))])
    labels = labels.set_index(ID).loc[x[ID]].reset_index()
    assert not labels[TIME].dt.month.eq(12).any()
    return augment(x, labels), labels


def augment(x, metadata):
    """Same ranking-visible inputs for development, audit, and final inference."""
    queues = pd.read_parquet(ROOT/'data/processed/queue_features.parquet')
    x = x.merge(queues, on=ID, how='left', validate='one_to_one')
    del queues
    weather = pd.read_parquet(ROOT/'data/processed/weather_hourly.parquet', filters=[('ADEP_mvt', '==', 'LIRF')])
    keys = pd.DataFrame({'ADEP_mvt': 'LIRF', 'hour': metadata[TIME].dt.floor('h')})
    wx = keys.merge(weather, on=['ADEP_mvt', 'hour'], how='left', validate='many_to_one')
    for col in wx.columns.difference(['ADEP_mvt', 'hour']):
        x[col] = wx[col].to_numpy()
    x['flight_prefix_length'] = x.flight_prefix.str.len()
    x = x.drop(columns=[ID, 'FLIGHT_mvt', 'CALLSIGN_flt'])
    x = x.replace([np.inf, -np.inf], np.nan)
    for col in x.select_dtypes('number'):
        x[col] = x[col].fillna(-999999).astype('float32')
    gc.collect()
    assert TARGET not in x and 'BLOCK_TIME_UTC_mvt' not in x
    return x


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))


def score(y, p, month, missing):
    return {'pooled': rmse(y, p), 'known': rmse(y[~missing], p[~missing]),
            'missing': rmse(y[missing], p[missing]),
            'months': {str(m): {'pooled': rmse(y[month==m], p[month==m]),
                'known': rmse(y[(month==m)&~missing], p[(month==m)&~missing]),
                'missing': rmse(y[(month==m)&missing], p[(month==m)&missing])} for m in [7, 11]}}


def main():
    if os.name == 'nt':
        import ctypes
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    x, meta = load()
    month = meta[TIME].dt.month.to_numpy()
    train, valid = ~np.isin(month, [7, 11]), np.isin(month, [7, 11])
    assert not np.intersect1d(np.flatnonzero(train), np.flatnonzero(valid)).size
    y = meta[TARGET].to_numpy(dtype=float)
    schedule = x[SCHEDULE].to_numpy(dtype=float)
    missing = x.mvt_minus_AOBT_3_flt.lt(-100000).to_numpy()
    identity = np.abs(y-schedule)<=6
    # Timestamp-free normal component must generalize to rows with no NM timing.
    nm_time = ('LOBT', 'IOBT', 'EOBT', 'AOBT', 'ARVT')
    normal_cols = [c for c in x if not any(token in c for token in nm_time)]
    normal_cols = [c for c in normal_cols if x.loc[train, c].nunique()>1]
    normal_train = train & ~identity & (y<6000)
    cats = list(x[normal_cols].select_dtypes('object').columns)
    normal = CatBoostRegressor(iterations=500, depth=6, learning_rate=.08, l2_leaf_reg=30,
        thread_count=4, random_seed=2026, loss_function='RMSE', verbose=False,
        one_hot_max_size=20, max_ctr_complexity=1)
    normal.fit(x.loc[normal_train, normal_cols], y[normal_train], cat_features=cats)
    normal_prediction = normal.predict(x.loc[valid, normal_cols])
    print('Conditional normal regressor trained', int(normal_train.sum()), flush=True)

    vm = meta.loc[valid].reset_index(drop=True)
    baseline = pd.read_parquet(OUT/'residual_d8_predictions.parquet').set_index(ID).loc[vm[ID], 'prediction'].to_numpy()
    specialist = pd.read_parquet(OUT/'missing_validation.parquet').set_index(ID).prediction.reindex(vm[ID]).to_numpy()
    alpha = json.loads((OUT/'baseline_mix.json').read_text())['specialist_alpha']
    baseline[missing[valid]] += alpha*(specialist[missing[valid]]-baseline[missing[valid]])
    lobt = x.loc[valid, 'mvt_minus_LOBT_flt'].to_numpy()
    known = lobt>-100000
    baseline[known] = np.clip(baseline[known], lobt[known]-3606, lobt[known]+3606)
    baseline = np.maximum(baseline, 0)
    vy, vmonth, vmissing = y[valid], month[valid], missing[valid]
    report = {'training_rows': int(train.sum()), 'normal_training_rows': int(normal_train.sum()),
        'validation_rows': int(valid.sum()), 'training_identity_fraction': float(identity[train].mean()),
        'excluded_months': [7, 11, 12], 'baseline': score(vy, baseline, vmonth, vmissing), 'candidates': {}}
    classifier_cols = [c for c in x if x.loc[train, c].nunique()>1]
    cats = list(x[classifier_cols].select_dtypes('object').columns)
    best_score, best = rmse(vy, baseline), None
    for weighted in [False, True]:
        weight = np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25) if weighted else np.ones(len(y))
        model = CatBoostClassifier(iterations=400, depth=5, learning_rate=.08, l2_leaf_reg=30,
            thread_count=4, random_seed=2026, loss_function='Logloss', verbose=False,
            one_hot_max_size=20, max_ctr_complexity=1)
        model.fit(x.loc[train, classifier_cols], identity[train].astype(int), cat_features=cats,
                  sample_weight=weight[train])
        probability = model.predict_proba(x.loc[valid, classifier_cols])[:, 1]
        expert = probability*schedule[valid]+(1-probability)*normal_prediction
        # Keep the established long-tail pipeline unchanged.
        expert[schedule[valid]>=24000] = baseline[schedule[valid]>=24000]
        expert = np.maximum(expert, 0)
        blended = baseline.copy()
        blend_weights = {}
        for flag in [False, True]:
            mask = vmissing==flag
            delta = expert[mask]-baseline[mask]
            a = float(np.clip(np.dot(delta, vy[mask]-baseline[mask])/max(np.dot(delta, delta), 1), 0, 1))
            blended[mask] += a*delta
            blend_weights['missing' if flag else 'known'] = a
        name = 'schedule_weighted' if weighted else 'unweighted'
        result = {'raw_expert': score(vy, expert, vmonth, vmissing),
            'blend_weights': blend_weights, 'blended': score(vy, blended, vmonth, vmissing)}
        report['candidates'][name] = result
        print(name, json.dumps(result), flush=True)
        if rmse(vy, blended)<best_score:
            best_score, best = rmse(vy, blended), (name, blended, expert)
            model.save_model(str(OUT/'lirf_identity_classifier.cbm'))
    report['selected'] = best[0] if best else None
    report['december_scored'] = False
    report['normal_features'], report['classifier_features'] = normal_cols, classifier_cols
    if best:
        normal.save_model(str(OUT/'lirf_identity_normal.cbm'))
        out = vm[[ID, TIME, 'ADEP_mvt', TARGET]].copy()
        out['prediction'], out['expert_prediction'] = best[1], best[2]
        out.to_parquet(OUT/'lirf_identity_validation.parquet', index=False)
        regime = schedule[valid]<24000
        standalone = vm.loc[regime, [ID]].copy()
        standalone['prediction'] = best[2][regime]
        assert standalone[ID].is_unique and np.isfinite(standalone.prediction).all()
        standalone.to_parquet(OUT/'lirf_identity_expert_validation.parquet', index=False)
        report['standalone_regime'] = 'Only LIRF rows with MVT-SCHED <24000 seconds; omitted rows receive no expert prediction or fallback.'
        report['standalone_rows'] = len(standalone)
        report['model_recipe'] = {'normal_model': 'artifacts/lirf_identity_normal.cbm',
            'classifier_model': 'artifacts/lirf_identity_classifier.cbm',
            'prediction': 'P(schedule identity)*(MVT-SCHED)+(1-P)*normal_prediction, clipped at zero',
            'normal_training': 'Nonidentity development rows with target<6000;500trees/depth6/lr0.08/l2=30',
            'classifier_training': 'All LIRF development rows;400trees/depth5/lr0.08/l2=30; selected schedule-magnitude weights',
            'development_excluded_months': [7, 11, 12]}
    (ROOT/'docs/lirf_identity_experiment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Selected', report['selected'], 'pooled RMSE', best_score, flush=True)


if __name__ == '__main__':
    main()
