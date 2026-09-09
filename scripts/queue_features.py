"""Ranking-visible taxi queues/proxy neighbors, streamed one supplied month at a time."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from audit_signals import DATA, ROOT

OUT = ROOT/'data/processed'
ID, TIME = 'MVT_ID_mvt', 'MVT_TIME_UTC_mvt'
COLS = [ID, 'PHASE_mvt', 'ADEP_mvt', 'ADES_mvt', TIME, 'AOBT_3_flt',
        'RUNWAY_mvt', 'SCHED_TIME_UTC_mvt', 'LOBT_flt']


def seconds(series):
    values = series.dt.as_unit('ns').astype('int64').to_numpy(dtype=float)/1e9
    values[series.isna().to_numpy()] = np.nan
    return values


def active(starts, ends, query):
    count = np.searchsorted(np.sort(starts), query, side='right') - np.searchsorted(np.sort(ends), query, side='right')
    return np.where(np.isfinite(query), count, np.nan).astype(float)


def neighbors(times, values, query, own, width):
    order = np.argsort(times)
    times, values = times[order], values[order]
    cumulative = np.r_[0., np.cumsum(values)]
    left = np.searchsorted(times, query-width, side='left')
    right = np.searchsorted(times, query+width, side='right')
    own_valid = np.isfinite(own)
    count = right-left-own_valid.astype(int)
    total = cumulative[right]-cumulative[left]-np.where(own_valid, own, 0)
    mean = np.divide(total, count, out=np.full(len(query), np.nan), where=count>0)
    return mean, count.astype(float)


def self_check():
    starts, ends = np.array([0., 5., 10.]), np.array([10., 15., 20.])
    np.testing.assert_array_equal(active(starts, ends, starts)-1, [0, 1, 1])
    np.testing.assert_array_equal(active(starts, ends, ends), [2, 1, 0])
    mean, count = neighbors(ends, np.array([10., 20., 30.]), ends, np.array([10., 20., 30.]), 5)
    np.testing.assert_array_equal(count, [1, 2, 1])
    np.testing.assert_array_equal(mean, [20, 20, 20])
    mean, count = neighbors(np.array([10.]), np.array([10.]), np.array([10.]), np.array([10.]), 5)
    assert count[0] == 0 and np.isnan(mean[0])


def features(d):
    start, end = seconds(d.AOBT_3_flt), seconds(d[TIME])
    duration = end-start
    good = np.isfinite(start) & np.isfinite(end) & (duration>0) & (duration<=21600)
    out = pd.DataFrame({ID: d[ID].to_numpy()})
    for prefix, keys in [('airport', ['ADEP_mvt']), ('runway', ['ADEP_mvt', 'RUNWAY_mvt'])]:
        columns = [f'{prefix}_queue_at_aobt', f'{prefix}_queue_at_takeoff']
        columns += [f'{prefix}_proxy_taxi_{kind}_{width}' for width in [1800, 7200] for kind in ['mean', 'count']]
        values = np.full((len(d), len(columns)), np.nan, dtype='float32')
        for _, index in d.groupby(keys, sort=False, dropna=False).groups.items():
            index = np.asarray(index)
            valid = index[good[index]]
            values[index, 0] = active(start[valid], end[valid], start[index])-good[index]
            # The interval ends at takeoff, so this count already excludes itself.
            values[index, 1] = active(start[valid], end[valid], end[index])
            own = np.where(good[index], duration[index], np.nan)
            for i, width in enumerate([1800, 7200]):
                mean, count = neighbors(end[valid], duration[valid], end[index], own, width)
                values[index, 2+2*i], values[index, 3+2*i] = mean, count
        out[columns] = values
    delta = end-seconds(d.LOBT_flt)
    bounds = pd.DataFrame({ID: d[ID].to_numpy(), 'lower_bound': delta-3606, 'upper_bound': delta+3606})
    assert (out.filter(like='queue_').fillna(0).to_numpy() >= 0).all()
    assert (out.filter(like='_count_').to_numpy() >= 0).all()
    return out, bounds, int(good.sum())


def verify_development_bound():
    """Separate label check; these labels never enter either generated feature file."""
    count, minimum, maximum = 0, np.inf, -np.inf
    for path in sorted(DATA.glob('training*.parquet')):
        if int(path.name.split('_')[1][5:7]) in [7, 11, 12]:
            continue
        d = pd.read_parquet(path, columns=[TIME, 'LOBT_flt', 'TAXITIME_SEC_mvt'], filters=[('PHASE_mvt', '==', 'DEP')])
        residual = (d[TIME]-d.LOBT_flt).dt.total_seconds()-d.TAXITIME_SEC_mvt
        residual = residual.dropna().to_numpy()
        assert np.abs(residual).max() <= 3606
        count += len(residual)
        minimum, maximum = min(minimum, float(residual.min())), max(maximum, float(residual.max()))
    return dict(rows=count, min=minimum, max=maximum, violations=0, excluded_months=[7, 11, 12])


def main():
    self_check()
    OUT.mkdir(exist_ok=True)
    queue_temp, bounds_temp = OUT/'queue_features.partial.parquet', OUT/'timing_bounds.partial.parquet'
    queue_writer = bounds_writer = None
    ids, total_valid, ranking_rows = [], 0, 0
    try:
        for path in sorted(DATA.glob('training*.parquet')) + [DATA/'ranking.parquet']:
            d = pd.read_parquet(path, columns=COLS, filters=[('PHASE_mvt', '==', 'DEP')]).reset_index(drop=True)
            assert d[ID].is_unique
            queue, bounds, valid = features(d)
            qt, bt = pa.Table.from_pandas(queue, preserve_index=False), pa.Table.from_pandas(bounds, preserve_index=False)
            if queue_writer is None:
                queue_writer = pq.ParquetWriter(queue_temp, qt.schema, compression='zstd')
                bounds_writer = pq.ParquetWriter(bounds_temp, bt.schema, compression='zstd')
            queue_writer.write_table(qt)
            bounds_writer.write_table(bt)
            ids.append(d[ID].to_numpy())
            total_valid += valid
            if path.name == 'ranking.parquet':
                ranking_rows = len(d)
            print(path.name, len(d), 'departures', flush=True)
    finally:
        if queue_writer is not None:
            queue_writer.close()
            bounds_writer.close()
    all_ids = np.concatenate(ids)
    assert len(all_ids) == len(np.unique(all_ids)) == 2429888 and ranking_rows == 344841
    queue_temp.replace(OUT/'queue_features.parquet')
    bounds_temp.replace(OUT/'timing_bounds.parquet')
    report = dict(rows=len(all_ids), ranking_departures=ranking_rows, features=12, valid_source_intervals=total_valid,
        source_columns=COLS, source_interval_seconds=[0, 21600], source_interval_lower_exclusive=True,
        self_exclusion='Own valid interval excluded at AOBT; own takeoff is already outside the half-open interval; neighbor means/counts exclude own duration.',
        temporal_scope='Each supplied file is processed independently; ranking contains two disconnected months. Symmetric proxy windows use provided post-operations times.',
        missing_queries='Queue at missing AOBT is NaN; empty neighbor means are NaN with count zero.',
        development_bound_check=verify_development_bound())
    (ROOT/'docs/queue_features.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
