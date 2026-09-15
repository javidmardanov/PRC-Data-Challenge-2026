"""Independently assemble attempts 2/3 from frozen raw models and original V3."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET, PRED, align
from check_solution import check_submission
from review_candidate import review


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, choices=[5, 6], required=True)
    args = parser.parse_args()
    art = ROOT/'artifacts'
    baseline = pd.read_parquet(ROOT/'submissions/elegant-alligator_v3.parquet')
    ids = baseline[ID]
    raw = align(pd.read_parquet(art/'known_lobt_d10_a2_final_predictions.parquet'), ids, 'final LOBT')
    rows = pd.read_parquet(art/'rows.parquet', columns=[ID, TIME, 'ADEP_mvt'])
    features = pd.read_parquet(art/'features.parquet', columns=['mvt_minus_AOBT_3_flt', 'mvt_minus_LOBT_flt', 'mvt_minus_SCHED_TIME_UTC_mvt'])
    rows[['aobt', 'lobt', 'schedule']] = features.to_numpy()
    meta = rows.set_index(ID).loc[ids]
    gate = (meta.aobt.to_numpy() > -100000) & (meta.lobt.to_numpy() > -100000)
    bounds = pd.read_parquet(ROOT/'data/processed/timing_bounds.parquet').set_index(ID).loc[ids]
    coefficient = np.full(len(ids), .630797588656986)
    if args.version == 6:
        config = json.loads((ROOT/'docs/autoresearch_lobt_regional.json').read_text())
        coeff = config['trials'][config['selected_index']]['coefficients']
        coefficient = meta.ADEP_mvt.map(coeff).fillna(.630797588656986).to_numpy(float)
    base = baseline[TARGET].to_numpy(float)
    result = base.copy()
    result[gate] += coefficient[gate]*(raw[PRED].to_numpy()[gate]-base[gate])
    result[gate] = np.clip(result[gate], np.maximum(bounds.lower_bound.fillna(-np.inf).to_numpy()[gate], 0),
                            bounds.upper_bound.fillna(np.inf).to_numpy()[gate])
    winter_gate = np.zeros(len(ids), dtype=bool)
    if args.version == 6:
        winter_gate = (meta.aobt.to_numpy() < -100000) & meta.ADEP_mvt.eq('LIRF').to_numpy() & meta[TIME].dt.month.eq(1).to_numpy() & (meta.schedule.to_numpy() < 24000)
        identity = pd.read_parquet(art/'autoresearch_flight_identity_final_raw.parquet').set_index(ID)
        positions = np.flatnonzero(winter_gate)
        candidate = identity.loc[ids.iloc[positions], PRED].to_numpy(float)
        result[positions] += .8788581367641393*(candidate-base[positions])
    if np.any(gate & winter_gate) or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('Overlapping corrections or nonfinite result')
    np.testing.assert_array_equal(result[~(gate | winter_gate)], base[~(gate | winter_gate)])
    output = ROOT/'submissions'/f'elegant-alligator_v{args.version}.parquet'
    if output.exists(): raise FileExistsError(output)
    out = baseline.copy(); out[TARGET] = result; out.to_parquet(output, index=False)
    check_submission(output)
    if args.version == 5:
        validation = pd.read_parquet(art/'known_lobt_d10_a2_gated_validation.parquet')
    else:
        validation = pd.read_parquet(art/'autoresearch_lobt_regional_validation.parquet')
        v3 = align(pd.read_parquet(art/'v3_validation.parquet'), validation[ID], 'local V3')
        winter = align(pd.read_parquet(art/'autoresearch_identity_winter_validation.parquet'), validation[ID], 'local winter')
        delta = winter[PRED].to_numpy()-v3[PRED].to_numpy()
        if np.any((delta != 0) & (validation[PRED].to_numpy() != v3[PRED].to_numpy())):
            raise ValueError('Validation corrections overlap')
        validation[PRED] += delta
    validation[PRED] = np.maximum(validation[PRED].to_numpy(float), 0)
    valpath = art/f'autoresearch_v{args.version}_validation.parquet'
    validation.to_parquet(valpath, index=False)
    report = review(valpath, art/'v3_validation.parquet')
    report.update(version=args.version, official_attempt=args.version-3,
                  ranking_known_gate_rows=int(gate.sum()), ranking_winter_gate_rows=int(winter_gate.sum()),
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  baseline='elegant-alligator_v3.parquet; rejected V4 correction excluded')
    report['nonnegative_projection'] = True
    (ROOT/'docs'/f'autoresearch_v{args.version}_review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
