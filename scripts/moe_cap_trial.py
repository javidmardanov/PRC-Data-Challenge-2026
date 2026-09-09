"""One bounded normal-expert target-cap trial; classifier remains frozen."""
import ctypes
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, TIME
from missing_v2 import OUT, PARAMS, ROOT, SCHED, load_training, load_validation, rmse
from moe_calibrate import complete_prediction, fit_alpha, routed, scores


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    full, valid, meta, missing = load_validation()
    x, rows = load_training(meta, missing)
    y = rows[TARGET].to_numpy(float)
    training = ~rows[TIME].dt.month.isin([7, 11]).to_numpy()
    validation = ~training
    order = pd.Index(rows.loc[validation, ID]).get_indexer(valid[ID])
    schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    identity = rome & (np.abs(y-schedule) <= 6)
    train = training & ~tail & ~identity & (y < 12000)
    weight = np.where(rows.missing_aobt, 8., 1.)
    cats = list(x.select_dtypes('object').columns)
    normal = CatBoostRegressor(**{**PARAMS, 'thread_count': 3}, loss_function='RMSE')
    normal.fit(x.loc[train], y[train], cat_features=cats, sample_weight=weight[train])
    normal.save_model(str(OUT/'moe_normal_cap12000_d6_i600.cbm'))
    raw = normal.predict(x.loc[validation], thread_count=3)
    cls = CatBoostClassifier().load_model(str(OUT/'missing_v2_identity.cbm'))
    probability = cls.predict_proba(x.loc[validation], thread_count=3)[:, 1]
    probability[~rome[validation] | (schedule[validation] <= 0)] = 0
    raw = np.maximum(probability*schedule[validation]+(1-probability)*raw, 0)[order]
    base, target = valid.prediction.to_numpy(), valid[TARGET].to_numpy()
    vtail = tail[validation][order]
    raw[vtail] = base[vtail]
    valid[[ID]].assign(prediction=raw).to_parquet(OUT/'moe_normal_cap12000_raw_missing_validation.parquet', index=False)
    masks = {'Rome': valid.ADEP_mvt.eq('LIRF').to_numpy() & ~vtail,
             'other': valid.ADEP_mvt.ne('LIRF').to_numpy()}
    alpha = {k: fit_alpha(base, raw, target, mask) for k, mask in masks.items()}
    pred = routed(base, raw, valid, alpha)
    complete_prediction(full, valid, pred).to_parquet(OUT/'moe_normal_cap12000_complete_validation.parquet', index=False)
    result = {'training_rows': int(train.sum()), 'target_cap': 12000, 'alphas': alpha,
              'scores': scores(full, valid, pred), 'raw_missing_rmse': rmse(target, raw),
              'months_raw_missing': {str(m): rmse(target[valid[TIME].dt.month.eq(m)], raw[valid[TIME].dt.month.eq(m)]) for m in [7, 11]}}
    (ROOT/'docs'/'research_moe_cap12000.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):
        main()
