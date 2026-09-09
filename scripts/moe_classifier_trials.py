"""Two bounded classifier-regularization trials for the missing-AOBT MoE."""
import ctypes
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, TIME
from missing_v2 import OUT, PARAMS, ROOT, load_training, load_validation, rmse
from moe_calibrate import complete_prediction, fit_alpha, routed, scores


TRIALS = {
    'moe_cls_d4_l50_i600': {'iterations': 600, 'depth': 4, 'l2_leaf_reg': 50},
    'moe_cls_d6_l60_i400': {'iterations': 400, 'depth': 6, 'l2_leaf_reg': 60},
}


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    full, valid, meta, missing = load_validation()
    x, rows = load_training(meta, missing)
    y = rows[TARGET].to_numpy(float)
    training = ~rows[TIME].dt.month.isin([7, 11]).to_numpy()
    validation = ~training
    order = pd.Index(rows.loc[validation, ID]).get_indexer(valid[ID])
    schedule = x['mvt_minus_SCHED_TIME_UTC_mvt'].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    train = training & ~tail
    identity = rome & (np.abs(y-schedule) <= 6)
    weight = np.where(rows.missing_aobt, 8., 1.)
    cls_weight = weight*np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25)
    cats = list(x.select_dtypes('object').columns)
    normal = CatBoostRegressor().load_model(str(OUT/'missing_v2_normal.cbm'))
    normal_raw = normal.predict(x.loc[validation], thread_count=3)
    vrome = rome[validation]
    vschedule = schedule[validation]
    vtail = tail[validation][order]
    base, target = valid.prediction.to_numpy(), valid[TARGET].to_numpy()
    region_masks = {'Rome': valid.ADEP_mvt.eq('LIRF').to_numpy() & ~vtail,
                    'other': valid.ADEP_mvt.ne('LIRF').to_numpy()}
    report = {'december_read': False, 'training_excluded_months': [7, 11, 12], 'trials': {}}
    for name, override in TRIALS.items():
        params = {**PARAMS, **override, 'thread_count': 3}
        model = CatBoostClassifier(**params, loss_function='Logloss')
        model.fit(x.loc[train & rome], identity[train & rome].astype(int), cat_features=cats,
                  sample_weight=cls_weight[train & rome])
        model.save_model(str(OUT/f'{name}.cbm'))
        probability = model.predict_proba(x.loc[validation], thread_count=3)[:, 1]
        probability[~vrome | (vschedule <= 0)] = 0
        raw = (probability*vschedule + (1-probability)*normal_raw)[order]
        raw = np.maximum(raw, 0)
        raw[vtail] = base[vtail]
        valid[[ID]].assign(prediction=raw).to_parquet(OUT/f'{name}_raw_missing_validation.parquet', index=False)
        alphas = {k: fit_alpha(base, raw, target, mask) for k, mask in region_masks.items()}
        pred = routed(base, raw, valid, alphas)
        complete = complete_prediction(full, valid, pred)
        complete.to_parquet(OUT/f'{name}_complete_validation.parquet', index=False)
        report['trials'][name] = {'params': override, 'alphas': alphas,
            'scores': scores(full, valid, pred),
            'raw_missing_rmse': rmse(target, raw),
            'tail_exact': bool(np.array_equal(pred[vtail], base[vtail]))}
        print(name, json.dumps(report['trials'][name]), flush=True)
    (ROOT/'docs'/'research_moe_classifier_trials.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):
        main()
