"""Direct known-Rome TARGET-minus-AOBT residual trial on frozen v2."""
import ctypes
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, TIME, validation_labels
from lirf_identity_experiment import OUT, ROOT, SCHEDULE, load, rmse


def fit_alpha(base, expert, y, mask):
    d = expert[mask]-base[mask]
    return float(np.clip(np.dot(d, y[mask]-base[mask])/max(np.dot(d, d), 1), 0, 1))


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    x, meta = load()
    month, day = meta[TIME].dt.month.to_numpy(), meta[TIME].dt.day.to_numpy()
    train, valid = ~np.isin(month, [7, 11]), np.isin(month, [7, 11])
    y = meta[TARGET].to_numpy(float)
    aobt = x.mvt_minus_AOBT_3_flt.to_numpy(float)
    schedule = x[SCHEDULE].to_numpy(float)
    known = aobt > -100000
    train &= known
    cols = [c for c in x if x.loc[train, c].nunique() > 1]
    cats = list(x[cols].select_dtypes('object').columns)
    model = CatBoostRegressor(iterations=1000, depth=6, learning_rate=.06, l2_leaf_reg=30,
        thread_count=3, random_seed=20260909, loss_function='RMSE', verbose=False,
        one_hot_max_size=20, max_ctr_complexity=1, allow_writing_files=False)
    model.fit(x.loc[train, cols], y[train]-aobt[train], cat_features=cats)
    model.save_model(str(OUT/'moe_known_residual_v3_d6_i1000.cbm'))

    vm = meta.loc[valid].reset_index(drop=True)
    v2 = pd.read_parquet(OUT/'v2_validation.parquet')
    full = validation_labels(OUT/'rows.parquet', v2[ID].to_numpy())
    full['prediction'] = v2.prediction.to_numpy()
    assert np.isclose(rmse(full[TARGET], full.prediction), 262.526361, atol=1e-6)
    base = full.set_index(ID).loc[vm[ID], 'prediction'].to_numpy()
    expert = np.maximum(aobt[valid]+model.predict(x.loc[valid, cols], thread_count=3), 0)
    eligible = known[valid] & (schedule[valid] < 24000)
    expert[~eligible] = base[~eligible]
    vy, vmonth, vday = y[valid], month[valid], day[valid]
    pooled_alpha = fit_alpha(base, expert, vy, eligible)
    pred = base.copy(); pred[eligible] += pooled_alpha*(expert[eligible]-base[eligible])
    complete = full.prediction.to_numpy().copy()
    pos = pd.Index(full[ID]).get_indexer(vm[ID]); complete[pos] = pred
    checks = {}
    for label, fit, check in [('fit_even_check_odd', vday%2==0, vday%2==1),
                              ('fit_odd_check_even', vday%2==1, vday%2==0),
                              ('fit_july_check_november', vmonth==7, vmonth==11),
                              ('fit_november_check_july', vmonth==11, vmonth==7)]:
        alpha = fit_alpha(base, expert, vy, eligible & fit)
        mask = eligible & check
        p = base[mask]+alpha*(expert[mask]-base[mask])
        checks[label] = {'alpha': alpha, 'rows': int(mask.sum()), 'rmse': rmse(vy[mask], p),
                         'baseline_rmse': rmse(vy[mask], base[mask])}
    report = {'excluded_months': [7, 11, 12], 'december_scored': False,
        'training_rows': int(train.sum()), 'validation_eligible_rows': int(eligible.sum()),
        'target': 'TAXITIME_SEC_mvt - mvt_minus_AOBT_3_flt', 'params': {'iterations': 1000, 'depth': 6, 'learning_rate': .06, 'l2_leaf_reg': 30},
        'pooled_alpha': pooled_alpha, 'baseline_full_rmse': rmse(full[TARGET], full.prediction),
        'corrected_full_rmse': rmse(full[TARGET], complete),
        'months': {str(m): {'full_rmse': rmse(full.loc[full[TIME].dt.month.eq(m), TARGET], complete[full[TIME].dt.month.eq(m)])} for m in [7, 11]},
        'held_checks': checks, 'selected': 'pending_parent_review'}
    vm[[ID]].assign(expert_prediction=expert).to_parquet(OUT/'moe_known_residual_v3_raw_validation.parquet', index=False)
    pd.DataFrame({ID: full[ID], 'prediction': complete}).to_parquet(OUT/'moe_known_residual_v3_complete_validation.parquet', index=False)
    (ROOT/'docs/research_moe_known_residual_v3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    pa.set_cpu_count(3); pa.set_io_thread_count(3)
    with threadpool_limits(limits=3): main()
