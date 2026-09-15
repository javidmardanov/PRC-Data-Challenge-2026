"""Bounded missing-timestamp residual trials against the complete frozen V3."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
from missing_v2 import load_validation, load_training, OUT, ROOT, SCHED
from ensemble import ID, TIME, TARGET, align, blend_alpha


def main():
    _, valid, meta, missing = load_validation()
    x, rows = load_training(meta, missing)
    arrival = pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
    ac = [c for c in arrival if c.startswith(('arr_stand_', 'arr_runway_'))]
    extra = rows[[ID]].merge(arrival[[ID]+ac], on=ID, how='left', validate='one_to_one')
    x[ac] = extra[ac].fillna(-999999).astype('float32')
    y = rows[TARGET].to_numpy(float)
    month = rows[TIME].dt.month
    schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & (schedule >= 24000)
    training = ~month.isin([7, 11]).to_numpy() & ~tail
    validation = month.isin([7, 11]).to_numpy()
    order = pd.Index(rows.loc[validation, ID]).get_indexer(valid[ID])
    if (order < 0).any(): raise ValueError('Missing validation IDs')
    v3 = pd.read_parquet(OUT/'v3_validation.parquet')
    labels = pd.read_parquet(OUT/'rows.parquet', filters=[(ID, 'in', v3[ID].tolist())])
    labels = align(labels, v3[ID], 'V3 labels')
    pos = pd.Index(v3[ID]).get_indexer(valid[ID])
    incumbent = v3.prediction.to_numpy()
    baseline = incumbent[pos]
    target = labels[TARGET].to_numpy()
    check = labels[TIME].dt.day.ge(15).to_numpy()
    calibration = valid[TIME].dt.day.le(14).to_numpy()
    regions = {'rome': valid.ADEP_mvt.eq('LIRF').to_numpy() & (valid[SCHED].to_numpy() < 24000),
               'other': valid.ADEP_mvt.ne('LIRF').to_numpy()}
    reference = np.where(rome & (schedule > 0) & ~tail, schedule, 900.)
    cats = x.select_dtypes('object').columns.tolist()
    report = {'baseline_rmse': float(np.sqrt(np.mean((incumbent-target)**2))), 'trials': {}}
    for name, only_missing, depth in [('missing_only_d6', True, 6), ('missing_weighted_d6', False, 6), ('missing_only_d4', True, 4)]:
        train = training & (rows.missing_aobt.to_numpy() if only_missing else True)
        model = CatBoostRegressor(iterations=1400, depth=depth, learning_rate=.04,
            l2_leaf_reg=40, thread_count=3, random_seed=20260915, verbose=False,
            loss_function='RMSE', one_hot_max_size=20, max_ctr_complexity=1,
            allow_writing_files=False)
        model.fit(x.loc[train], (y-reference)[train], cat_features=cats,
                  sample_weight=np.where(rows.loc[train, 'missing_aobt'], 8., 1.))
        model.save_model(str(OUT/f'autoresearch_{name}.cbm'))
        raw = np.maximum(model.predict(x.loc[validation], thread_count=3)+reference[validation], 0)[order]
        pred = baseline.copy()
        alphas = {}
        for region, mask in regions.items():
            fit = mask & calibration
            alpha = blend_alpha(baseline[fit], raw[fit], valid.loc[fit, TARGET].to_numpy())
            pred[mask] += alpha*(raw[mask]-baseline[mask])
            alphas[region] = alpha
        full = incumbent.copy(); full[pos] = pred
        scores = {}
        for m in [7, 11]:
            mask = check & labels[TIME].dt.month.eq(m).to_numpy()
            scores[str(m)] = {'baseline': float(np.sqrt(np.mean((incumbent[mask]-target[mask])**2))),
                             'candidate': float(np.sqrt(np.mean((full[mask]-target[mask])**2)))}
        result = {'alphas': alphas, 'late_dates': scores, 'full_rmse': float(np.sqrt(np.mean((full-target)**2))),
                  'train_rows': int(train.sum()), 'eligible': all(s['candidate'] < s['baseline'] for s in scores.values())}
        report['trials'][name] = result
        v3[[ID]].assign(prediction=full).to_parquet(OUT/f'autoresearch_{name}_validation.parquet', index=False)
        valid[[ID]].assign(prediction=raw).to_parquet(OUT/f'autoresearch_{name}_raw.parquet', index=False)
        (ROOT/'docs/autoresearch_missing.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=3): main()
