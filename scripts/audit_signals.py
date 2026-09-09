"""Audit supplied fields only; never use ranking departure labels or credentials."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/raw/prc-2026-datasets'
TIMES = ['AOBT_3_flt', 'LOBT_flt', 'IOBT_flt', 'EOBT_1_flt', 'SCHED_TIME_UTC_mvt']
COLS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'ADEP_mvt', 'ADES_mvt', 'RUNWAY_mvt',
        'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'AIRCRAFT_OPERATOR_flt', 'FLIGHT_mvt',
        'MVT_TIME_UTC_mvt', 'BLOCK_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'] + TIMES


def read(path):
    df = pd.read_parquet(path, columns=COLS, filters=[('PHASE_mvt', '==', 'DEP')])
    for col in df.select_dtypes('object'):
        df[col] = df[col].astype('category')
    for col in TIMES:
        df['dt_' + col] = (df.MVT_TIME_UTC_mvt - df[col]).dt.total_seconds().astype('float32')
    return df


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return dict(count=len(x), mean=float(x.mean()), rmse=float(np.sqrt(np.mean(x*x))),
                quantiles={str(q): float(np.quantile(x, q)) for q in [0, .001, .01, .1, .5, .9, .99, .999, 1]})


def audit(train, ranking):
    y = train.TAXITIME_SEC_mvt
    report = {'rows': {'train': len(train), 'ranking': len(ranking)}, 'target': stats(y),
              'exact_target_identity': bool(((train.MVT_TIME_UTC_mvt - train.BLOCK_TIME_UTC_mvt).dt.total_seconds() == y).all()),
              'missing': {split: df.isna().mean().to_dict() for split, df in [('train', train), ('ranking', ranking)]},
              'prediction_residuals': {}, 'airports': {}, 'duplicates': {}, 'categorical_shift': {}}
    for col in TIMES:
        pred = train['dt_' + col]
        report['prediction_residuals'][col] = {'raw': stats(pred-y), 'clipped_60_5400': stats(pred.clip(60, 5400)-y)}
    for airport, df in train.groupby('ADEP_mvt', observed=True):
        yp = df.TAXITIME_SEC_mvt
        pred = df.dt_AOBT_3_flt
        rank = ranking[ranking.ADEP_mvt == airport]
        report['airports'][airport] = {'count': len(df), 'target': stats(yp),
            'aobt_residual': stats(pred-yp), 'aobt_clipped_residual': stats(pred.clip(60,5400)-yp),
            'aobt_abs_error_above_300_fraction': float(((pred-yp).abs()>300).mean()),
            'aobt_missing_train': float(pred.isna().mean()), 'aobt_missing_ranking': float(rank.dt_AOBT_3_flt.isna().mean()),
            'target_seconds_mod60': yp.mod(60).value_counts(normalize=True).head(5).to_dict()}
    for split, df in [('train', train), ('ranking', ranking)]:
        fid = df.FLIGHT_ID_mvt.dropna()
        report['duplicates'][split] = {'movement_ids': int(df.MVT_ID_mvt.duplicated().sum()),
            'nonnull_flight_ids': int(fid.duplicated().sum()),
            'target_negative': int((df.TAXITIME_SEC_mvt < 0).sum())}
    for col in ['ADEP_mvt','ADES_mvt','RUNWAY_mvt','STAND_mvt','AIRCRAFT_TYPE_mvt','AIRCRAFT_OPERATOR_flt','FLIGHT_mvt']:
        valid = ranking[col].notna()
        known = set(train[col].dropna())
        report['categorical_shift'][col] = {'train_unique': train[col].nunique(), 'ranking_unique': ranking[col].nunique(),
            'ranking_unseen_fraction': float((~ranking.loc[valid,col].isin(known)).mean())}
    assert report['rows']['train'] > 2_000_000 and report['rows']['ranking'] == 344841
    return report


if __name__ == '__main__':
    train = pd.concat([read(p) for p in sorted(DATA.glob('training*.parquet'))], ignore_index=True)
    ranking = read(DATA / 'ranking.parquet')
    result = audit(train, ranking)
    out = ROOT / 'docs/signal_audit.json'
    out.write_text(json.dumps(result, indent=2, default=lambda x: x.item()), encoding='utf-8')
    print(out)
    print(json.dumps({k: result[k] for k in ['target','exact_target_identity','prediction_residuals','airports','duplicates','categorical_shift']}, indent=2))
