"""Missing-AOBT specialist; tune July/November, then freeze before December fitting.

Run normally to reproduce validation, an audit model excluding December, and final
ranking predictions. --predict-audit-december only predicts; it never scores December.
"""
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from catboost import CatBoostRegressor

from audit_signals import COLS, DATA, ROOT, TIMES
from proxy_baseline import features, rmse

OUT = ROOT / 'artifacts'
ID, TARGET, TIME = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt'
PARAMS = dict(iterations=406, depth=5, learning_rate=.04, l2_leaf_reg=25,
              thread_count=4, random_seed=2026, verbose=False, loss_function='RMSE')
# Frozen after exploratory July/November checks; no subsequent threshold sweep.
TAIL_LOW, TAIL_HIGH, SHRINK_SUPPORT = 54000, 86400, 3


def read_missing(path):
    columns = [c for c in COLS if c != 'BLOCK_TIME_UTC_mvt']
    d = pd.read_parquet(path, columns=columns,
                        filters=(ds.field('PHASE_mvt') == 'DEP') & ds.field('AOBT_3_flt').is_null())
    for col in TIMES:
        d['dt_' + col] = (d[TIME] - d[col]).dt.total_seconds().astype('float32')
    return d


def base(d, means):
    p = d.ADEP_mvt.map(means).to_numpy(dtype=float)
    rome = d.ADEP_mvt.eq('LIRF').to_numpy()
    p[rome] = np.maximum(d.loc[rome, 'dt_SCHED_TIME_UTC_mvt'], 0)
    return p


def fit(d, path):
    x, cats = features(d)
    means = d.groupby('ADEP_mvt')[TARGET].mean().to_dict()
    model = CatBoostRegressor(**PARAMS)
    model.fit(x, d[TARGET].to_numpy() - base(d, means), cat_features=cats)
    model.save_model(str(path))
    return model, means


def tail_parameters(d):
    rome = d.ADEP_mvt.eq('LIRF')
    ordinary = float(d.loc[rome & d[TARGET].lt(4000), TARGET].median())
    day_level = 86400 + ordinary
    sample = d.loc[rome & d.dt_SCHED_TIME_UTC_mvt.between(TAIL_LOW, TAIL_HIGH)]
    delta = day_level - sample.dt_SCHED_TIME_UTC_mvt.to_numpy()
    residual = sample[TARGET].to_numpy() - sample.dt_SCHED_TIME_UTC_mvt.to_numpy()
    # Closed-form steepest-descent optimum for one quadratic coefficient, shrunk
    # toward the schedule timestamp by three zero-offset pseudo-observations.
    raw = float(np.clip(np.dot(delta, residual) / max(np.dot(delta, delta), 1), 0, 1))
    weight = raw * len(sample) / (len(sample) + SHRINK_SUPPORT)
    return dict(support=len(sample), day_level=day_level, raw_weight=raw, weight=weight,
                lower=TAIL_LOW, upper=TAIL_HIGH, shrink_support=SHRINK_SUPPORT)


def predict(model, d, means, tail=None):
    x, _ = features(d)
    p = base(d, means) + model.predict(x)
    rome = d.ADEP_mvt.eq('LIRF').to_numpy()
    schedule = d.dt_SCHED_TIME_UTC_mvt.to_numpy()
    band = rome & (schedule >= 24000) & (schedule <= 48000)
    p[band] = schedule[band]
    if tail:
        band = rome & (schedule >= tail['lower']) & (schedule <= tail['upper'])
        p[band] = schedule[band] + tail['weight'] * (tail['day_level'] - schedule[band])
    assert np.isfinite(p).all()
    return p


def metrics(d, p):
    y = d[TARGET].to_numpy()
    result = {'pooled': rmse(y, p), 'months': {}, 'airports': {}}
    for col, key in [(d[TIME].dt.month, 'months'), (d.ADEP_mvt, 'airports')]:
        for value in col.unique():
            mask = col.eq(value).to_numpy()
            result[key][str(value)] = rmse(y[mask], p[mask])
    return result


def save_predictions(d, p, path, labels=False):
    columns = [ID, TIME, 'ADEP_mvt'] + ([TARGET] if labels else [])
    out = d[columns].copy()
    out['prediction'] = p
    assert out[ID].is_unique
    out.to_parquet(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict-audit-december', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.predict_audit_december:
        config = json.loads((OUT/'missing_audit_config.json').read_text())
        model = CatBoostRegressor().load_model(str(OUT/'missing_audit_model.cbm'))
        december = read_missing(next(DATA.glob('training_2025-12-01*.parquet')))
        p = predict(model, december, config['means'], config['tail'])
        save_predictions(december, p, OUT/'missing_december_audit.parquet')
        print('Saved December audit predictions; no December score calculated.', flush=True)
        return

    paths = sorted(DATA.glob('training*.parquet'))
    development = pd.concat([read_missing(p) for p in paths if '2025-12-01_2026' not in p.name], ignore_index=True)
    valid = development[TIME].dt.month.isin([7, 11])
    train, validation = development.loc[~valid].copy(), development.loc[valid].copy()
    assert len(validation) == 5294 and not train[TIME].dt.month.isin([7, 11, 12]).any()
    model, means = fit(train, OUT/'missing_development_model.cbm')
    tail = tail_parameters(train)
    regular = predict(model, validation, means)
    adjusted = predict(model, validation, means, tail)
    use_tail = rmse(validation[TARGET], adjusted) < rmse(validation[TARGET], regular)
    selected = adjusted if use_tail else regular
    report = dict(train_rows=len(train), validation_rows=len(validation), params=PARAMS,
                  selection_frozen=True, tail_enabled=use_tail, tail=tail,
                  baseline=metrics(validation, regular), validation=metrics(validation, selected),
                  validation_months=[7, 11], december_scored=False,
                  rejected_context_experiment='missing_context_experiment.json')
    (OUT/'missing_specialist_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save_predictions(validation, selected, OUT/'missing_validation.parquet', labels=True)
    print('FROZEN', json.dumps(report), flush=True)

    # Preserve the original development fit for the ensemble's independent audit.
    shutil.copyfile(OUT/'missing_development_model.cbm', OUT/'missing_audit_model.cbm')
    audit_tail = tail if use_tail else None
    (OUT/'missing_audit_config.json').write_text(json.dumps(dict(means=means, tail=audit_tail), indent=2))
    del model

    # Only after selection is frozen do December labels participate in fitting.
    december = read_missing(next(DATA.glob('training_2025-12-01*.parquet')))
    all_train = pd.concat([development, december], ignore_index=True)
    final_model, final_means = fit(all_train, OUT/'missing_regression.cbm')
    final_tail = tail_parameters(all_train) if use_tail else None
    ranking = read_missing(DATA/'ranking.parquet')
    save_predictions(ranking, predict(final_model, ranking, final_means, final_tail), OUT/'missing_ranking.parquet')
    report.update(final_fit_rows=len(all_train), ranking_rows=len(ranking), final_tail=final_tail,
                  audit_fit_rows=len(train), audit_excluded_months=[7, 11, 12])
    (OUT/'missing_specialist_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Saved final ranking predictions and unscored December audit capability.', flush=True)


if __name__ == '__main__':
    main()
