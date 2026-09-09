"""Departure features from supplied arrival observations at the same airport.

Same-month arrival summaries are available in this post-operations challenge.
Departure labels and block timestamps are never loaded to build these features.
"""
import argparse
import json

import numpy as np
import pandas as pd

from train import ROOT, RAW, ID, TIME, TARGET


DEP_COLUMNS = [ID, TIME, 'ADEP_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt']
ARR_COLUMNS = [TIME, 'ADES_mvt', 'STAND_mvt', 'RUNWAY_mvt', 'AIRCRAFT_TYPE_mvt',
               'BLOCK_TIME_UTC_mvt', TARGET]


def features(departures, arrivals):
    d, a = departures.reset_index(drop=True).copy(), arrivals.copy()
    d['airport'] = d.ADEP_mvt
    a['airport'] = a.ADES_mvt
    d['period'] = d[TIME].dt.strftime('%Y-%m')
    a['period'] = a[TIME].dt.strftime('%Y-%m')
    a['arrival_duration'] = a[TARGET].clip(0, 7200)
    out = d[[ID]].copy()
    for key, label in [('STAND_mvt', 'stand'), ('RUNWAY_mvt', 'runway')]:
        keys = ['airport', 'period', key]
        stats = a.dropna(subset=[key]).groupby(keys).arrival_duration.agg(['mean', 'median', 'count'])
        joined = d[keys].join(stats, on=keys)
        for statistic in stats.columns:
            out[f'arr_{label}_{statistic}'] = joined[statistic].to_numpy(dtype='float32')

    # An earlier arrival at this stand is a useful turnaround clue; it is not
    # assumed to be the same aircraft. Missing stands receive no match.
    previous = ['age', 'taxi', 'aircraft_match']
    for name in previous:
        out[f'arr_stand_previous_{name}'] = np.nan
    arrival_groups = a.dropna(subset=['STAND_mvt', 'BLOCK_TIME_UTC_mvt']).groupby(['airport', 'period', 'STAND_mvt'])
    for key, indexes in d.dropna(subset=['STAND_mvt']).groupby(['airport', 'period', 'STAND_mvt']).groups.items():
        if key not in arrival_groups.groups:
            continue
        group = arrival_groups.get_group(key).sort_values('BLOCK_TIME_UTC_mvt')
        times = group.BLOCK_TIME_UTC_mvt.dt.as_unit('ns').astype('int64').to_numpy()
        takeoff = d.loc[indexes, TIME].dt.as_unit('ns').astype('int64').to_numpy()
        positions = np.searchsorted(times, takeoff, side='right') - 1
        safe = np.maximum(positions, 0)
        age = (takeoff - times[safe]) / 1e9
        good = (positions >= 0) & (age <= 86400)
        indexes = np.asarray(indexes)[good]
        selected = group.iloc[safe[good]]
        out.loc[indexes, 'arr_stand_previous_age'] = age[good]
        out.loc[indexes, 'arr_stand_previous_taxi'] = selected.arrival_duration.to_numpy()
        match = selected.AIRCRAFT_TYPE_mvt.to_numpy() == d.loc[indexes, 'AIRCRAFT_TYPE_mvt'].to_numpy()
        out.loc[indexes, 'arr_stand_previous_aircraft_match'] = match.astype(float)
    for column in out.columns.difference([ID]):
        out[column] = out[column].astype('float32')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        frame = pd.read_parquet(next(RAW.glob('training*.parquet'))).head(6000)
        dep = frame.PHASE_mvt.eq('DEP')
        before = features(frame.loc[dep], frame.loc[~dep])
        frame.loc[dep, TARGET] = -123456
        frame.loc[dep, 'BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2000-01-01', tz='UTC')
        after = features(frame.loc[dep], frame.loc[~dep])
        pd.testing.assert_frame_equal(before, after)
        if before.arr_stand_previous_age.notna().sum() == 0:
            raise ValueError('Mutation check must exercise a real previous-arrival match')
        print('PASS: arrival context unchanged after hidden departure label/block mutation')
        return
    results = []
    for path in [*sorted(RAW.glob('training*.parquet')), RAW/'ranking.parquet']:
        d = pd.read_parquet(path, columns=DEP_COLUMNS, filters=[('PHASE_mvt', '==', 'DEP')])
        a = pd.read_parquet(path, columns=ARR_COLUMNS, filters=[('PHASE_mvt', '==', 'ARR')])
        results.append(features(d, a))
        print(path.name, len(d), flush=True)
    out = pd.concat(results, ignore_index=True)
    if not out[ID].is_unique:
        raise ValueError('Arrival context must have unique departure IDs')
    destination = ROOT/'data/processed/arrival_context_v3.parquet'
    out.to_parquet(destination, index=False)
    combined = ROOT/'data/processed/queue_arrival_context_v3.parquet'
    queue = pd.read_parquet(ROOT/'data/processed/queue_features.parquet')
    if set(queue[ID]) != set(out[ID]):
        raise ValueError('Queue and arrival features must cover the same departure IDs')
    queue.merge(out, on=ID, validate='one_to_one').to_parquet(combined, index=False)
    report = {'rows': len(out), 'path': str(destination.relative_to(ROOT)),
              'queue_combined_path': str(combined.relative_to(ROOT)),
              'columns': out.columns.tolist(), 'departure_target_or_block_read': False,
              'source': 'Official supplied arrival observations; group by airport and calendar month',
              'arrival_duration_clip_seconds': [0, 7200], 'previous_arrival_max_age_seconds': 86400,
              'coverage': out.notna().mean().to_dict()}
    (ROOT/'docs/arrival_context_v3.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
