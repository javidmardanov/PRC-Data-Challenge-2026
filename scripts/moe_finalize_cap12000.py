"""Final cap-12000 normal refit, reusing the frozen final identity classifier."""
import ctypes
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits

from ensemble import ID, TARGET, frozen_predictions
from missing_v2 import OUT, PARAMS, ROOT, SCHED
from moe_finalize import load_final_data


def main():
    if os.name == 'nt':
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    x, rows = load_final_data()
    training = rows.source.eq('training').to_numpy()
    ranking = rows.source.eq('ranking').to_numpy() & rows.missing_aobt.to_numpy()
    y, schedule = rows[TARGET].to_numpy(float), x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    identity = rome & (np.abs(y-schedule) <= 6)
    normal_train = training & ~tail & ~identity & (y < 12000)
    weight = np.where(rows.missing_aobt, 8., 1.)
    cats = list(x.select_dtypes('object').columns)
    normal = CatBoostRegressor(**{**PARAMS, 'thread_count': 3}, loss_function='RMSE')
    normal.fit(x.loc[normal_train], y[normal_train], cat_features=cats, sample_weight=weight[normal_train])
    normal.save_model(str(OUT/'moe_normal_cap12000_final.cbm'))
    classifier = CatBoostClassifier().load_model(str(OUT/'moe_missing_v2_identity_final.cbm'))
    raw = normal.predict(x.loc[ranking], thread_count=3)
    probability = classifier.predict_proba(x.loc[ranking], thread_count=3)[:, 1]
    rank_rome, rank_schedule = rome[ranking], schedule[ranking]
    probability[~rank_rome | (rank_schedule <= 0)] = 0
    raw = np.maximum(probability*rank_schedule+(1-probability)*raw, 0)
    raw_out = rows.loc[ranking, [ID]].copy()
    raw_out['prediction'] = raw
    raw_path = OUT/'moe_normal_cap12000_raw_missing_ranking.parquet'
    raw_out.to_parquet(raw_path, index=False)

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
    complete_path = OUT/'moe_normal_cap12000_complete_ranking.parquet'
    pd.DataFrame({ID: ids, 'prediction': complete}).to_parquet(complete_path, index=False)
    old_ids = pd.read_parquet(OUT/'missing_ranking.parquet', columns=[ID])[ID]
    report = {'selected': 'normal_cap12000_fixed_original_alphas', 'selection_frozen': True,
        'normal_training_rows': int(normal_train.sum()), 'classifier_reused': 'moe_missing_v2_identity_final.cbm',
        'raw_rows': len(raw_out), 'exact_missing_id_set': set(raw_out[ID]) == set(old_ids),
        'complete_rows': len(ids), 'finite': bool(np.isfinite(raw).all() and np.isfinite(complete).all()),
        'changed_rows': int(apply.sum()), 'tail_fallback_rows': int((~apply).sum()),
        'alphas': {'Rome': 0.4224120030856563, 'other': 1.0},
        'routing': 'After complete v1; LIRF schedule>=24000 remains complete-v1 prediction.',
        'raw_output': str(raw_path), 'complete_output': str(complete_path), 'december_scored': False}
    (ROOT/'docs'/'research_moe_cap12000_final.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    pa.set_cpu_count(3)
    pa.set_io_thread_count(3)
    with threadpool_limits(limits=3):
        main()
