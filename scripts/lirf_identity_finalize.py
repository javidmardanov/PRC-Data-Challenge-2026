"""Frozen LIRF expert: unscored December audit or final all-2025 refit/ranking."""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from lirf_identity_experiment import ID, OUT, ROOT, SCHEDULE, TARGET, TIME, augment


def hashes():
    return {name: hashlib.sha256((OUT/name).read_bytes()).hexdigest()
            for name in ['lirf_identity_classifier.cbm', 'lirf_identity_normal.cbm']}


def inputs(audit):
    # Audit metadata does not contain departure labels.
    meta = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'ADEP_mvt', 'source'], filters=[('ADEP_mvt', '==', 'LIRF')])
    filters = [('ADEP_mvt', '==', 'LIRF')]
    if audit:
        meta = meta.loc[meta[TIME].dt.month.eq(12)].reset_index(drop=True)
        filters += [('month', '==', 12)]
        assert meta.source.eq('training').all() and meta[TIME].dt.year.eq(2025).all()
    x = pd.read_parquet(OUT/'features.parquet', filters=filters).reset_index(drop=True)
    assert len(x) == len(meta)
    x[ID] = meta[ID].to_numpy()
    return augment(x, meta), meta


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--final', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        import ctypes
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x8000)
    original_hashes = hashes()
    recipe = json.loads((ROOT/'docs/lirf_identity_experiment.json').read_text())
    assert recipe['selected'] == 'schedule_weighted'
    normal_cols, classifier_cols = recipe['normal_features'], recipe['classifier_features']
    x, meta = inputs(args.audit)
    schedule = x[SCHEDULE].to_numpy(dtype=float)
    if args.audit:
        normal = CatBoostRegressor().load_model(str(OUT/'lirf_identity_normal.cbm'))
        classifier = CatBoostClassifier().load_model(str(OUT/'lirf_identity_classifier.cbm'))
        select = schedule<24000
        training_rows = recipe['training_rows']
        normal_training_rows = recipe['normal_training_rows']
        training_months = [1, 2, 3, 4, 5, 6, 8, 9, 10]
    else:
        train = meta.source.eq('training').to_numpy()
        assert meta.loc[train, TIME].dt.year.eq(2025).all()
        # December departure labels are read only in this final-fit branch.
        labels = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TARGET],
                                  filters=[('ADEP_mvt', '==', 'LIRF'), ('source', '==', 'training')])
        y = labels.set_index(ID)[TARGET].reindex(meta[ID]).to_numpy(dtype=float)
        identity = np.abs(y-schedule)<=6
        ordinary = train & ~identity & (y<6000)
        common = dict(thread_count=4, random_seed=2026, verbose=False, one_hot_max_size=20,
                      max_ctr_complexity=1, learning_rate=.08, l2_leaf_reg=30)
        normal = CatBoostRegressor(**common, iterations=500, depth=6, loss_function='RMSE')
        normal.fit(x.loc[ordinary, normal_cols], y[ordinary],
                   cat_features=list(x[normal_cols].select_dtypes('object').columns))
        normal.save_model(str(OUT/'lirf_identity_normal_final.cbm'))
        print('Final conditional normal fit complete', int(ordinary.sum()), 'rows', flush=True)
        classifier = CatBoostClassifier(**common, iterations=400, depth=5, loss_function='Logloss')
        weight = np.clip(1+(np.maximum(schedule, 0)/3600)**2, 1, 25)
        classifier.fit(x.loc[train, classifier_cols], identity[train].astype(int),
            sample_weight=weight[train], cat_features=list(x[classifier_cols].select_dtypes('object').columns))
        classifier.save_model(str(OUT/'lirf_identity_classifier_final.cbm'))
        select = ~train & (schedule<24000)
        training_rows, normal_training_rows = int(train.sum()), int(ordinary.sum())
        training_months = sorted(meta.loc[train, TIME].dt.month.unique().tolist())
    assert list(normal.feature_names_) == normal_cols and list(classifier.feature_names_) == classifier_cols
    p = classifier.predict_proba(x.loc[select, classifier_cols])[:, 1]
    prediction = np.maximum(p*schedule[select]+(1-p)*normal.predict(x.loc[select, normal_cols]), 0)
    out = meta.loc[select, [ID]].copy()
    out['prediction'] = prediction
    assert out[ID].is_unique and np.isfinite(prediction).all()
    if args.final:
        template = pd.read_parquet(ROOT/'data/raw/prc-2026-datasets/submitting.parquet', columns=[ID])
        assert out[ID].isin(template[ID]).all()
    destination = OUT/('lirf_identity_expert_audit.parquet' if args.audit else 'lirf_identity_expert_ranking.parquet')
    out.to_parquet(destination, index=False)
    assert hashes() == original_hashes, 'Original development weights changed.'
    report_path = ROOT/'docs/lirf_identity_finalization.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report['audit' if args.audit else 'final'] = dict(output=str(destination), rows=len(out),
        classifier_training_rows=training_rows, normal_training_rows=normal_training_rows,
        training_months=training_months, december_scored=False,
        regime='LIRF and MVT-SCHED <24000 seconds; no fallback for omitted rows',
        original_development_sha256=original_hashes,
        original_development_weights_unchanged=True, cpu_threads=4, gpu=False)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['audit' if args.audit else 'final'], indent=2), flush=True)


if __name__ == '__main__':
    main()
