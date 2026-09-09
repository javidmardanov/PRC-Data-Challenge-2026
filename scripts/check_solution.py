"""Small checks for hidden-label isolation and the submission contract."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from train import RAW, TARGET, TIME, ID, features


def check_features():
    d = pd.read_parquet(next(RAW.glob('training*.parquet'))).head(3000)
    a, _ = features(d)
    dep = d.PHASE_mvt.eq('DEP')
    d.loc[dep,TARGET] = np.nan
    d.loc[dep,'BLOCK_TIME_UTC_mvt'] = pd.NaT
    b, _ = features(d)
    pd.testing.assert_frame_equal(a,b)
    assert all(TARGET not in c and 'BLOCK_TIME_UTC_mvt' not in c for c in a)
    deps=d.loc[dep].reset_index(drop=True)
    ap=deps.iloc[0].ADEP_mvt; t=deps.iloc[0][TIME]
    expected=((deps.ADEP_mvt==ap)&(deps[TIME]>=t-pd.Timedelta(seconds=900))&(deps[TIME]<t)).sum()
    assert a.loc[0,'DEP_count_-1_900']==expected
    assert a.loc[0,'DEP_count_-1_900'] < 100
    print('PASS: features invariant to held-out departure target/block timestamps')


def check_submission(path):
    expected = pd.read_parquet(RAW/'submitting.parquet')
    actual = pd.read_parquet(path)
    if actual.columns.tolist() != expected.columns.tolist() or len(actual) != 344841:
        raise ValueError('Submission must preserve the original columns and all 344,841 rows')
    if not actual[ID].equals(expected[ID]) or not actual[ID].is_unique:
        raise ValueError('Submission IDs must exactly match the original template order')
    if actual[TARGET].dtype.kind not in 'fi' or not np.isfinite(actual[TARGET]).all():
        raise ValueError('Submission predictions must be finite numeric values')
    print('PASS: exactly 344,841 finite predictions; original IDs, order and columns')
    print(actual[TARGET].describe(percentiles=[.01,.5,.99,.999]).to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('submission', nargs='?',type=Path)
    args=p.parse_args()
    if args.submission: check_submission(args.submission)
    else: check_features()
