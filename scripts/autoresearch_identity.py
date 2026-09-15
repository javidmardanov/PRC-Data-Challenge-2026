"""One flight-identity feature trial for missing Rome schedule/normal routing."""
import argparse
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from threadpoolctl import threadpool_limits
from missing_v2 import ROOT, OUT, SCHED, PARAMS, load_validation, load_training
from ensemble import ID, TIME, TARGET, align, blend_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rolling-month', choices=['2025-11'])
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if args.final and args.rolling_month:
        parser.error('Choose final or rolling mode')
    if args.final:
        from moe_finalize import load_final_data
        x, rows = load_final_data()
    else:
        _, valid, meta, missing = load_validation()
        x, rows = load_training(meta, missing, cutoff=args.rolling_month+'-01' if args.rolling_month else None)
    arrival = pd.read_parquet(ROOT/'data/processed/arrival_context_v3.parquet')
    cols = [c for c in arrival if c.startswith(('arr_stand_', 'arr_runway_'))]
    extra = rows[[ID]].merge(arrival[[ID]+cols], on=ID, how='left', validate='one_to_one')
    x[cols] = extra[cols].fillna(-999999).astype('float32')
    y = rows[TARGET].to_numpy(float); schedule = x[SCHED].to_numpy(float)
    rome = x.ADEP_mvt.eq('LIRF').to_numpy()
    training = ~rows[TIME].dt.month.isin([7, 11]).to_numpy()
    if args.rolling_month:
        training = rows[TIME].lt(pd.Timestamp(args.rolling_month+'-01', tz='UTC')).to_numpy()
    validation = ~training
    if args.final:
        training = rows.source.eq('training').to_numpy()
        validation = rows.source.eq('ranking').to_numpy() & rows.missing_aobt.to_numpy()
    normal_path = ('forward_v3_'+args.rolling_month+'_missing_arrival.cbm' if args.rolling_month
                   else 'moe_missing_arrival_v3_normal_cap12000.cbm')
    if args.final:
        normal_path = 'moe_missing_arrival_v3_normal_cap12000_final.cbm'
    model = CatBoostRegressor().load_model(str(OUT/normal_path))
    normal = model.predict(x.loc[validation, model.feature_names_], thread_count=3)
    # Recurring flight/route identifiers are available in ranking; targets never enter these keys.
    x['flight_identity'] = x.flight_prefix+'_'+x.flight_number.astype(str)
    x['flight_destination'] = x.flight_identity+'_'+x.ADES_mvt
    train = training & rome & (schedule < 24000)
    identity = np.abs(y-schedule) <= 6
    weight = np.where(rows.missing_aobt, 8., 1.)*np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25)
    cls = CatBoostClassifier(**{**PARAMS, 'iterations': 600, 'depth': 6}, loss_function='Logloss')
    cls.fit(x.loc[train], identity[train].astype(int), cat_features=x.select_dtypes('object').columns.tolist(), sample_weight=weight[train])
    name = 'autoresearch_flight_identity'+('_final' if args.final else '_'+args.rolling_month if args.rolling_month else '')
    cls.save_model(str(OUT/(name+'.cbm')))
    prob = cls.predict_proba(x.loc[validation], thread_count=3)[:, 1]
    prob[~rome[validation] | (schedule[validation] <= 0)] = 0
    if args.rolling_month or args.final:
        raw = np.maximum(prob*schedule[validation]+(1-prob)*normal, 0)
        rows.loc[validation, [ID]].assign(prediction=raw).to_parquet(OUT/(name+'_raw.parquet'), index=False)
        print(json.dumps({'month': args.rolling_month, 'train_max': str(rows.loc[train, TIME].max()),
                          'train_rows': int(train.sum()), 'output_rows': int(validation.sum()),
                          'coefficient_to_check': .8788581367641393}), flush=True)
        return
    order = pd.Index(rows.loc[validation, ID]).get_indexer(valid[ID])
    raw = np.maximum(prob*schedule[validation]+(1-prob)*normal, 0)[order]
    v3 = pd.read_parquet(OUT/'v3_validation.parquet')
    labels = pd.read_parquet(OUT/'rows.parquet', filters=[(ID, 'in', v3[ID].tolist())])
    labels = align(labels, v3[ID], 'labels'); target = labels[TARGET].to_numpy(float)
    pos = pd.Index(v3[ID]).get_indexer(valid[ID]); base = v3.prediction.to_numpy()
    gate = valid.ADEP_mvt.eq('LIRF').to_numpy() & (valid[SCHED].to_numpy() < 24000)
    early = valid[TIME].dt.day.le(14).to_numpy()
    fit = gate & early
    alpha = blend_alpha(base[pos][fit], raw[fit], valid.loc[fit, TARGET].to_numpy())
    result = base.copy(); result[pos[gate]] += alpha*(raw[gate]-base[pos[gate]])
    report = {'alpha': alpha, 'new_features': ['flight_identity', 'flight_destination'],
              'full_rmse': float(np.sqrt(np.mean((result-target)**2))), 'late': {}}
    for m in [7, 11]:
        mask = labels[TIME].dt.month.eq(m) & labels[TIME].dt.day.ge(15)
        report['late'][str(m)] = {'baseline': float(np.sqrt(np.mean((base[mask]-target[mask])**2))),
                                  'candidate': float(np.sqrt(np.mean((result[mask]-target[mask])**2)))}
    report['eligible'] = all(v['candidate'] < v['baseline'] for v in report['late'].values())
    v3[[ID]].assign(prediction=result).to_parquet(OUT/'autoresearch_identity_validation.parquet', index=False)
    (ROOT/'docs/autoresearch_identity.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=3): main()
