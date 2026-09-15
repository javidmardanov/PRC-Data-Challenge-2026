"""Independent reconstruction of the frozen first-attempt day-length correction."""
import hashlib
import json
import numpy as np
import pandas as pd
from ensemble import ROOT, ID, TIME, TARGET
from missing_specialist import read_missing
from check_solution import check_submission
from review_candidate import review


def main():
    recipe = json.loads((ROOT/'docs/autoresearch_ensemble_lirf_tail_policy.json').read_text())
    raw = ROOT/'data/raw/prc-2026-datasets'
    training = pd.concat([read_missing(p) for p in sorted(raw.glob('training*.parquet'))], ignore_index=True)
    s = 'dt_SCHED_TIME_UTC_mvt'
    par = recipe['final_parameters']
    mask = training.ADEP_mvt.eq('LIRF') & training[s].between(54000, 86400)
    level = 86400 + training.loc[training.ADEP_mvt.eq('LIRF') & training[TARGET].lt(4000), TARGET].median()
    delta = level-training.loc[mask, s].to_numpy(float)
    residual = training.loc[mask, TARGET].to_numpy(float)-training.loc[mask, s].to_numpy(float)
    weight = float(np.clip(np.dot(delta, residual)/np.dot(delta, delta), 0, 1))
    np.testing.assert_allclose([level, weight], [par['day_level'], par['weight']], rtol=0, atol=1e-10)
    baseline = pd.read_parquet(ROOT/'submissions/elegant-alligator_v3.parquet')
    ranking = read_missing(raw/'ranking.parquet')
    gate = ranking.ADEP_mvt.eq('LIRF') & ranking[s].between(54000, 86400)
    pos = pd.Index(baseline[ID]).get_indexer(ranking.loc[gate, ID])
    if (pos < 0).any(): raise ValueError('Ranking ID absent from template')
    out = baseline.copy()
    schedule = ranking.loc[gate, s].to_numpy(float)
    out.loc[pos, TARGET] = schedule + weight*(level-schedule)
    proposed = pd.read_parquet(ROOT/recipe['ranking_output'])
    np.testing.assert_array_equal(out[ID], proposed[ID])
    np.testing.assert_allclose(out[TARGET], proposed[TARGET], rtol=0, atol=1e-8)
    output = ROOT/'submissions/elegant-alligator_v4.parquet'
    if output.exists(): raise FileExistsError(output)
    out.to_parquet(output, index=False)
    check_submission(output)
    changed = out[TARGET].ne(baseline[TARGET]).to_numpy()
    if changed.sum() != gate.sum() or set(np.flatnonzero(changed)) != set(pos):
        raise ValueError('Unexpected changed rows')
    # Independently reconstruct development predictions using the frozen training-only coefficient.
    dev = recipe['trials'][recipe['selected_index']]['spec']
    val = pd.read_parquet(ROOT/'artifacts/v3_validation.parquet')
    source = training.loc[training[ID].isin(val[ID])].set_index(ID)
    gated = source.ADEP_mvt.eq('LIRF') & source[s].between(54000, 86400)
    vp = pd.Index(val[ID]).get_indexer(source.index[gated])
    vs = source.loc[gated, s].to_numpy(float)
    val.loc[vp, 'prediction'] = vs+dev['weight']*(dev['day_level']-vs)
    valpath = ROOT/'artifacts/autoresearch_v4_validation.parquet'
    val.to_parquet(valpath, index=False)
    report = review(valpath, ROOT/'artifacts/v3_validation.parquet')
    report.update(ranking_changed_rows=int(changed.sum()), final_training_support=int(mask.sum()),
                  final_weight=weight, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  official_attempt=1, maximum_official_attempts=3)
    (ROOT/'docs/autoresearch_v4_review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
