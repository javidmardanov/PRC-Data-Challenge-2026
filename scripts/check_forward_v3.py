"""Regression-check forward_v3 arithmetic against the retained frozen V3 cache."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import forward_v3

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; ID=forward_v3.ID
MAP={'tfm':'tfm_queue_d8_predictions.parquet','plain':'no_tfm_full_d8_predictions.parquet',
 'lgbm':'lgbm_tfm_validation.parquet','d10':'tfm_queue_d10_v2_predictions.parquet',
 'v2tail':'catboost_tail_d6_s600_predictions.parquet','tail':'catboost_tail_d8_full_v3_predictions.parquet',
 'arrival':'tfm_arrival_d8_v3_predictions.parquet','missing':'missing_validation.parquet',
 'rome':'lirf_identity_expert_validation.parquet','missing_v2':'moe_normal_cap12000_raw_missing_validation.parquet',
 'missing_arrival':'moe_missing_arrival_v3_raw_validation.parquet'}

def main():
    expected=pd.read_parquet(ART/'v3_validation.parquet').set_index(ID).prediction
    report={}
    for month in ('2025-07','2025-11'):
        ids=set(pd.read_parquet(ART/'rows.parquet',columns=[ID,forward_v3.TIME]).loc[
            lambda d:d[forward_v3.TIME].dt.strftime('%Y-%m').eq(month),ID])
        prefix='check_v3_'+month
        for key,src in MAP.items():
            suffix='_predictions' if key in {'tfm','plain','lgbm','d10','v2tail','tail','arrival'} else ''
            path=ART/f'{prefix}_{key}{suffix}.parquet'
            if not path.exists():
                d=pd.read_parquet(ART/src); d.loc[d[ID].isin(ids)].to_parquet(path,index=False)
        output=ART/f'{prefix}_complete.parquet'
        stats=forward_v3.reconstruct(month,prefix,[ART/f'{prefix}_{x}_predictions.parquet' for x in ('tfm','plain','lgbm','d10')],output)
        actual=pd.read_parquet(output).set_index(ID).prediction
        diff=np.abs(actual-expected.reindex(actual.index)); maximum=float(diff.max())
        if maximum>=1e-8: raise AssertionError(f'{month}: max difference {maximum}')
        report[month]={'rows':len(actual),'max_abs_difference':maximum,'rmse':stats['rmse']}
    (ROOT/'docs/forward_v3_reconstruction_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
