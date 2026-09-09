"""Small known-AOBT Rome timestamp-source experts on top of frozen v2."""
import ctypes
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, TIME, validation_labels
from lirf_identity_experiment import OUT, ROOT, SCHEDULE, load, rmse


PARAMS = dict(iterations=400, depth=5, learning_rate=.08, l2_leaf_reg=30,
              thread_count=3, random_seed=20260909, loss_function='Logloss', verbose=False,
              one_hot_max_size=20, max_ctr_complexity=1, allow_writing_files=False)


def alpha(base, expert, y, mask):
    d = expert[mask]-base[mask]
    return float(np.clip(np.dot(d, y[mask]-base[mask])/max(np.dot(d, d), 1), 0, 1))


def evaluate(name, vm, y, base, expert, eligible, full, report):
    month, day = vm[TIME].dt.month.to_numpy(), vm[TIME].dt.day.to_numpy()
    pooled_alpha = alpha(base, expert, y, eligible)
    pred = base.copy(); pred[eligible] += pooled_alpha*(expert[eligible]-base[eligible])
    complete = full.prediction.to_numpy().copy()
    pos = pd.Index(full[ID]).get_indexer(vm[ID]); assert (pos>=0).all()
    complete[pos] = pred
    result = {'pooled_alpha': pooled_alpha, 'known_rome_rmse': rmse(y[eligible], pred[eligible]),
              'full_rmse': rmse(full[TARGET], complete), 'months': {}, 'held_checks': {}}
    for m in [7, 11]:
        fm = full[TIME].dt.month.eq(m).to_numpy()
        result['months'][str(m)] = {'full_rmse': rmse(full.loc[fm, TARGET], complete[fm]),
            'known_rome_rmse': rmse(y[eligible & (month==m)], pred[eligible & (month==m)])}
    for label, fit, check in [('fit_even_check_odd', day%2==0, day%2==1),
                              ('fit_odd_check_even', day%2==1, day%2==0),
                              ('fit_july_check_november', month==7, month==11),
                              ('fit_november_check_july', month==11, month==7)]:
        a = alpha(base, expert, y, eligible & fit)
        check_mask = eligible & check
        check_pred = base[check_mask]+a*(expert[check_mask]-base[check_mask])
        result['held_checks'][label] = {'alpha': a, 'rows': int(check_mask.sum()),
            'rmse': rmse(y[check_mask], check_pred), 'baseline_rmse': rmse(y[check_mask], base[check_mask])}
    out = vm[[ID]].copy(); out['expert_prediction'] = expert
    out.to_parquet(OUT/f'moe_known_v3_{name}_raw_validation.parquet', index=False)
    pd.DataFrame({ID: full[ID], 'prediction': complete}).to_parquet(OUT/f'moe_known_v3_{name}_complete_validation.parquet', index=False)
    report['trials'][name] = result
    print(name, json.dumps(result), flush=True)


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    x, meta = load()
    month = meta[TIME].dt.month.to_numpy()
    train, valid = ~np.isin(month, [7, 11]), np.isin(month, [7, 11])
    y = meta[TARGET].to_numpy(float)
    schedule = x[SCHEDULE].to_numpy(float)
    candidates = np.column_stack([schedule, x.mvt_minus_AOBT_3_flt.to_numpy(float),
                                  x.mvt_minus_LOBT_flt.to_numpy(float)])
    available = candidates > -100000
    error = np.where(available, np.abs(candidates-y[:, None]), np.inf)
    nearest = error.argmin(axis=1)
    source = np.where(error.min(axis=1)<=6, nearest+1, 0)  # normal,schedule,AOBT,LOBT
    known = available[:, 1]
    recipe = json.loads((ROOT/'docs/lirf_identity_experiment.json').read_text())
    normal_cols = recipe['normal_features']
    classifier_cols = recipe['classifier_features']
    normal = CatBoostRegressor().load_model(str(OUT/'lirf_identity_normal.cbm'))
    normal_all = normal.predict(x[normal_cols], thread_count=3)
    cats = list(x[classifier_cols].select_dtypes('object').columns)
    vm = meta.loc[valid].reset_index(drop=True)
    v2 = pd.read_parquet(OUT/'v2_validation.parquet')
    full = validation_labels(OUT/'rows.parquet', v2[ID].to_numpy())
    full['prediction'] = v2.prediction.to_numpy()
    assert np.isclose(rmse(full[TARGET], full.prediction), 262.526361, atol=1e-6)
    base = full.set_index(ID).loc[vm[ID], 'prediction'].to_numpy()
    vy = y[valid]
    eligible = known[valid] & (schedule[valid] < 24000)
    report = {'excluded_months': [7, 11, 12], 'december_scored': False,
              'source_classes': ['normal', 'schedule', 'AOBT', 'LOBT'],
              'training_source_counts': {str(k): int(((source==k)&train&known).sum()) for k in range(4)},
              'validation_source_counts': {str(k): int(((source==k)&valid&known).sum()) for k in range(4)},
              'eligible_validation_rows': int(eligible.sum()), 'baseline_full_rmse': rmse(full[TARGET], full.prediction),
              'trials': {}}

    # Binary schedule identity, with candidate separation rather than schedule magnitude.
    binary = (source==1).astype(int)
    gap_weight = np.clip(1+((schedule-normal_all)/1800)**2, 1, 25)
    model = CatBoostClassifier(**PARAMS)
    model.fit(x.loc[train & known, classifier_cols], binary[train & known], cat_features=cats,
              sample_weight=gap_weight[train & known])
    model.save_model(str(OUT/'moe_known_v3b_gap_binary.cbm'))
    probability = model.predict_proba(x.loc[valid, classifier_cols], thread_count=3)[:, 1]
    expert = probability*schedule[valid]+(1-probability)*normal_all[valid]
    expert[~eligible] = base[~eligible]
    evaluate('gap_binary_v3b', vm, vy, base, np.maximum(expert, 0), eligible, full, report)

    # Four sources: normal, schedule, AOBT, and LOBT.
    separation = np.nanmax(np.where(available, candidates, np.nan), axis=1)-np.nanmin(np.where(available, candidates, np.nan), axis=1)
    multi_weight = np.clip(1+(np.nan_to_num(separation)/1800)**2, 1, 25)
    multi = CatBoostClassifier(**{**PARAMS, 'loss_function': 'MultiClass'})
    multi.fit(x.loc[train & known, classifier_cols], source[train & known], cat_features=cats,
              sample_weight=multi_weight[train & known])
    multi.save_model(str(OUT/'moe_known_v3b_multisource.cbm'))
    probability = multi.predict_proba(x.loc[valid, classifier_cols], thread_count=3)
    values = np.column_stack([normal_all[valid], candidates[valid]])
    values[:, 1:] = np.where(available[valid], values[:, 1:], normal_all[valid, None])
    expert = np.sum(probability*values, axis=1)
    expert[~eligible] = base[~eligible]
    evaluate('multisource_v3b', vm, vy, base, np.maximum(expert, 0), eligible, full, report)
    report['selected'] = 'frozen_v2_no_correction'
    report['selection_reason'] = ('Gap binary gains only 0.006 pooled RMSE and loses even-to-odd and '
                                  'November-to-July transfer; multisource has pooled alpha zero.')
    (ROOT/'docs/research_moe_v3.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    pa.set_cpu_count(3); pa.set_io_thread_count(3)
    with threadpool_limits(limits=3): main()
