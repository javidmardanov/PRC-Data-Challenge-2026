"""Leakage-conscious calibration checks for the frozen missing-v2 MoE predictions."""
import json

import numpy as np
import pandas as pd

from ensemble import ID, TARGET, TIME
from missing_v2 import OUT, load_validation, rmse


def fit_alpha(base, raw, y, mask):
    d = raw[mask] - base[mask]
    return float(np.clip(np.dot(d, y[mask] - base[mask]) / max(np.dot(d, d), 1), 0, 1))


def routed(base, raw, valid, alphas):
    pred = base.copy()
    rome = valid.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & valid['mvt_minus_SCHED_TIME_UTC_mvt'].ge(24000).to_numpy()
    for key, mask in [('Rome', rome & ~tail), ('other', ~rome)]:
        pred[mask] += alphas[key] * (raw[mask] - pred[mask])
    pred[tail] = base[tail]
    return pred


def scores(full, valid, pred):
    full_pred = full.prediction.to_numpy().copy()
    pos = pd.Index(full[ID]).get_indexer(valid[ID])
    full_pred[pos] = pred
    return {
        str(m): rmse(full.loc[full[TIME].dt.month.eq(m), TARGET],
                     full_pred[full[TIME].dt.month.eq(m)])
        for m in [7, 11]
    } | {'combined': rmse(full[TARGET], full_pred), 'missing': rmse(valid[TARGET], pred)}


def complete_prediction(full, valid, pred):
    out = full[[ID]].copy()
    out['prediction'] = full.prediction.to_numpy()
    pos = pd.Index(full[ID]).get_indexer(valid[ID])
    out.loc[pos, 'prediction'] = pred
    assert len(out) == 353045 and out[ID].is_unique and np.isfinite(out.prediction).all()
    return out


def main():
    full, valid, _, _ = load_validation()
    raw_df = pd.read_parquet(OUT/'missing_v2_mixture_validation.parquet').set_index(ID)
    raw = raw_df.loc[valid[ID], 'prediction'].to_numpy()
    base, y = valid.prediction.to_numpy(), valid[TARGET].to_numpy()
    month = valid[TIME].dt.month.to_numpy()
    rome = valid.ADEP_mvt.eq('LIRF').to_numpy()
    tail = rome & valid['mvt_minus_SCHED_TIME_UTC_mvt'].ge(24000).to_numpy()
    region_masks = {'Rome': rome & ~tail, 'other': ~rome}
    trials = {}
    for calibration_month, check_month in [(7, 11), (11, 7)]:
        alphas = {k: fit_alpha(base, raw, y, mask & (month == calibration_month))
                  for k, mask in region_masks.items()}
        pred = routed(base, raw, valid, alphas)
        trials[f'fit_{calibration_month}_check_{check_month}'] = {
            'alphas': alphas, 'scores': scores(full, valid, pred),
            'check_missing_rmse': rmse(y[month == check_month], pred[month == check_month])}
    day = valid[TIME].dt.day.to_numpy()
    for calibration_parity in [0, 1]:
        fit = day % 2 == calibration_parity
        alphas = {k: fit_alpha(base, raw, y, mask & fit) for k, mask in region_masks.items()}
        pred = routed(base, raw, valid, alphas)
        check = ~fit
        trials[f'fit_day_parity_{calibration_parity}'] = {
            'alphas': alphas, 'scores': scores(full, valid, pred),
            'check_missing_rmse': rmse(y[check], pred[check])}
    pooled = {k: fit_alpha(base, raw, y, mask) for k, mask in region_masks.items()}
    pred = routed(base, raw, valid, pooled)
    output = {'december_read': False, 'pooled': {'alphas': pooled, 'scores': scores(full, valid, pred)},
              'transfer_trials': trials,
              'routing': 'Apply after complete v1; LIRF schedule>=24000 remains byte-for-byte v1.'}
    (OUT/'moe_calibration_validation.parquet').write_bytes(
        valid[[ID]].assign(prediction=pred).to_parquet(index=False))
    complete_prediction(full, valid, pred).to_parquet(OUT/'moe_selected_complete_validation.parquet', index=False)
    (OUT.parent/'docs'/'research_moe.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
