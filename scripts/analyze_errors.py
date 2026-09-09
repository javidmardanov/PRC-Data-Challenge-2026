"""Aggregate July/November residual diagnostics; never read December target labels."""
import json

import numpy as np
import pandas as pd

from audit_signals import ROOT

OUT = ROOT/'artifacts'
ID, TARGET, TIME = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'MVT_TIME_UTC_mvt'
COLUMNS = ['month', 'mvt_minus_AOBT_3_flt', 'mvt_minus_LOBT_flt',
           'mvt_minus_IOBT_flt', 'mvt_minus_EOBT_1_flt', 'mvt_minus_SCHED_TIME_UTC_mvt',
           'LOBT_flt_minus_AOBT_3_flt', 'EOBT_1_flt_minus_AOBT_3_flt',
           'nm_adep_match', 'nm_ades_match', 'nm_aircraft_match', 'diverted',
           'RUNWAY_mvt', 'STAND_mvt', 'AIRCRAFT_TYPE_mvt', 'flight_prefix',
           'MVT_TIME_UTC_mvt_hour', 'arr_taxi_mean_7200', 'DEP_count_-1_3600']


def metrics(d, pred='prediction'):
    e = d[pred].to_numpy()-d[TARGET].to_numpy()
    return dict(rows=len(d), rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()),
                sse=float(np.sum(e**2)))


def main():
    meta = pd.read_parquet(OUT/'rows.parquet', columns=[ID, TIME, 'source'])
    monthly = meta[TIME].dt.month.isin([7, 11])
    meta = meta.loc[monthly].reset_index(drop=True)
    x = pd.read_parquet(OUT/'features.parquet', columns=COLUMNS, filters=[('month', 'in', [7, 11])])
    assert len(x) == len(meta)
    x[ID] = meta[ID].to_numpy()
    x = x.loc[meta.source.eq('training')]
    d = pd.read_parquet(OUT/'residual_d8_predictions.parquet').merge(x, on=ID, validate='one_to_one')
    assert len(d) == 353045 and d[TIME].dt.month.isin([7, 11]).all()
    specialist = pd.read_parquet(OUT/'missing_validation.parquet')[[ID, 'prediction']].rename(columns={'prediction': 'specialist'})
    d = d.merge(specialist, how='left', on=ID, validate='one_to_one')
    alpha = json.loads((OUT/'baseline_mix.json').read_text())['specialist_alpha']
    d['blended'] = d.prediction
    missing = d.specialist.notna()
    d.loc[missing, 'blended'] += alpha*(d.loc[missing, 'specialist']-d.loc[missing, 'prediction'])
    d['blended'] = d.blended.clip(lower=0)
    d['aobt_missing'] = d.mvt_minus_AOBT_3_flt.lt(-100000)
    report = {'baseline': metrics(d), 'specialist_blend': metrics(d, 'blended')}
    report['airport_month_availability'] = {f'{a}/{int(m)}/{"missing" if miss else "known"}': metrics(g, 'blended')
            for (a, m, miss), g in d.groupby(['ADEP_mvt', 'month', 'aobt_missing'])}
    report['availability'] = {str(miss): metrics(g, 'blended') for miss, g in d.groupby('aobt_missing')}
    d['sq_error'] = (d.blended-d[TARGET])**2
    report['concentration'] = {str(n): float(d.nlargest(n, 'sq_error').sq_error.sum()/d.sq_error.sum())
                               for n in [1, 5, 10, 25, 100, 1000]}
    known = d.loc[~d.aobt_missing].copy()
    for name, bins, values in [
            ('aobt_proxy', [-np.inf, 0, 300, 600, 1200, 2400, 3600, 7200, 86400, np.inf], known.mvt_minus_AOBT_3_flt),
            ('target', [-np.inf, 300, 600, 1200, 2400, 3600, 7200, np.inf], known[TARGET]),
            ('lobt_aobt_disagreement', [-np.inf, -3600, -1800, -600, 0, 600, 1800, 3600, np.inf], known.LOBT_flt_minus_AOBT_3_flt)]:
        report[name] = {str(k): metrics(g, 'blended') for k, g in known.groupby(pd.cut(values, bins), observed=True)}
    report['match_flags'] = {f'{col}/{k}': metrics(g, 'blended') for col in ['nm_adep_match', 'nm_ades_match', 'nm_aircraft_match', 'diverted']
                             for k, g in known.groupby(col)}
    lobt_known = d.mvt_minus_LOBT_flt.gt(-100000)
    residual = d.loc[lobt_known, 'mvt_minus_LOBT_flt'] - d.loc[lobt_known, TARGET]
    report['lobt_target_bound'] = dict(min=float(residual.min()), max=float(residual.max()),
        violations_above_3606=int(residual.abs().gt(3606).sum()), rows=len(residual))
    d['bounded'] = d.blended
    lo = d.loc[lobt_known, 'mvt_minus_LOBT_flt']-3606
    hi = d.loc[lobt_known, 'mvt_minus_LOBT_flt']+3606
    d.loc[lobt_known, 'bounded'] = np.clip(d.loc[lobt_known, 'blended'], lo, hi)
    d.bounded = d.bounded.clip(lower=0)
    report['lobt_bound_correction'] = {'changed_rows': int(d.bounded.ne(d.blended).sum()),
        'pooled': metrics(d, 'bounded'), 'months': {str(int(m)): metrics(g, 'bounded') for m, g in d.groupby('month')},
        'airports': {a: metrics(g, 'bounded') for a, g in d.groupby('ADEP_mvt')}}
    report['worst_100_patterns'] = {'airport_counts': d.nlargest(100, 'sq_error').ADEP_mvt.value_counts().to_dict(),
        'aobt_missing': int(d.nlargest(100, 'sq_error').aobt_missing.sum()),
        'median_schedule_delta': float(d.nlargest(100, 'sq_error').mvt_minus_SCHED_TIME_UTC_mvt.median())}
    report['conclusions'] = [
        'Project NM-known predictions into the empirical MVT-LOBT +/-3606-second interval: no July/November target violates it; both months improve.',
        'Missing-AOBT rows are 1.50% of validation but 29.28% of squared error. LIRF accounts for about 39% of all squared error.',
        'Known targets above 2400 seconds comprise 1.08% of known rows but 21.17% of total squared error; prioritize long-taxi source ambiguity.',
        'Try LOBT as the residual reference: its training target is bounded while the AOBT residual can be very large. Validate before replacing the existing model.',
        'Test airport/runway queue counts from supplied positive AOBT-to-MVT intervals at each flight AOBT and MVT. Existing movement-window counts measure traffic flow, not the number of aircraft taxiing.',
        'Aircraft/destination mismatch flags explain little total squared error; optimizing these flags alone is lower priority.'
    ]
    (ROOT/'docs/error_analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['baseline', 'specialist_blend', 'availability', 'concentration',
        'lobt_target_bound', 'lobt_bound_correction', 'worst_100_patterns']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
