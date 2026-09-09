"""Reproducible flight-level RMSE optimization; all validation departure labels stay hidden."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/prc-2026-datasets'
OUT = ROOT / 'artifacts'
TARGET = 'TAXITIME_SEC_mvt'
TIME = 'MVT_TIME_UTC_mvt'
ID = 'MVT_ID_mvt'


def features(frame):
    """Only ranking-visible fields; arrival labels may be used, departure labels never."""
    dep = frame.PHASE_mvt.eq('DEP')
    d = frame.loc[dep].reset_index(drop=True)
    x = pd.DataFrame(index=d.index)
    cats = [c for c in d if d[c].dtype == object and c not in ['PHASE_mvt']]
    for c in cats:
        x[c] = d[c].fillna('__NA__').astype(str)
    x['airport_runway'] = x.ADEP_mvt + '_' + x.RUNWAY_mvt
    x['airport_stand'] = x.ADEP_mvt + '_' + x.STAND_mvt
    x['stand_prefix'] = x.STAND_mvt.str.extract(r'^([A-Za-z]+)', expand=False).fillna('__NA__')
    x['stand_number'] = pd.to_numeric(x.STAND_mvt.str.extract(r'(\d+)', expand=False), errors='coerce')
    x['flight_prefix'] = x.FLIGHT_mvt.str.extract(r'^([A-Za-z]+)', expand=False).fillna('__NA__')
    x['flight_number'] = pd.to_numeric(x.FLIGHT_mvt.str.extract(r'(\d+)', expand=False), errors='coerce')
    x['callsign_prefix'] = x.CALLSIGN_flt.str[:3]
    dates = [c for c in d if isinstance(d[c].dtype, pd.DatetimeTZDtype) and c != 'BLOCK_TIME_UTC_mvt']
    for c in dates:
        s = d[c]
        x[c + '_hour'] = s.dt.hour + s.dt.minute / 60
        x[c + '_minute'] = s.dt.minute
        x[c + '_second'] = s.dt.second
        if c != TIME:
            x['mvt_minus_' + c] = (d[TIME] - s).dt.total_seconds().clip(-172800, 172800)
    for a, b in [('LOBT_flt','AOBT_3_flt'), ('IOBT_flt','AOBT_3_flt'),
                 ('EOBT_1_flt','AOBT_3_flt'), ('SCHED_TIME_UTC_mvt','AOBT_3_flt'),
                 ('ARVT_3_flt','ARVT_1_flt'), ('LOBT_flt','IOBT_flt')]:
        x[a + '_minus_' + b] = (d[a] - d[b]).dt.total_seconds().clip(-172800,172800)
    x['month'] = d[TIME].dt.month
    x['dow'] = d[TIME].dt.dayofweek
    x['day'] = d[TIME].dt.day
    x['dayofyear'] = d[TIME].dt.dayofyear
    x['week'] = d[TIME].dt.isocalendar().week.astype(float)
    x['nm_missing'] = d.FLIGHT_ID_mvt.isna().astype(int)
    x['nm_adep_match'] = d.ADEP_mvt.eq(d.ADEP_flt).astype(int)
    x['nm_ades_match'] = d.ADES_mvt.eq(d.ADES_flt).astype(int)
    x['nm_aircraft_match'] = d.AIRCRAFT_TYPE_mvt.eq(d.AIRCRAFT_TYPE_flt).astype(int)
    x['diverted'] = d.ADES_flt.ne(d.ADES_FILED_flt).astype(int)

    # Post-operations task: movement times and arrival taxi labels are supplied in ranking.
    airport = frame.ADEP_mvt.where(frame.PHASE_mvt.eq('DEP'), frame.ADES_mvt)
    seconds = frame[TIME].dt.as_unit('ns').astype('int64').to_numpy() / 1e9
    ds = d[TIME].dt.as_unit('ns').astype('int64').to_numpy() / 1e9
    for ap, idx in d.groupby('ADEP_mvt').groups.items():
        ix = np.asarray(idx)
        for phase in ['DEP', 'ARR']:
            mask = airport.eq(ap) & frame.PHASE_mvt.eq(phase)
            st = np.sort(seconds[mask])
            for width in [900, 1800, 3600, 7200]:
                for direction in [-1, 1]:
                    lo = ds[ix] - width if direction == -1 else ds[ix]
                    hi = ds[ix] if direction == -1 else ds[ix] + width
                    # Excludes current departure from its own preceding interval.
                    count = np.searchsorted(st, hi, side='left') - np.searchsorted(st, lo, side='left')
                    x.loc[ix, f'{phase}_count_{direction}_{width}'] = count
            if phase == 'ARR':
                ar = frame.loc[mask, [TIME, TARGET]].sort_values(TIME)
                at = ar[TIME].dt.as_unit('ns').astype('int64').to_numpy() / 1e9
                val = ar[TARGET].to_numpy(dtype=float)
                good = np.isfinite(val)
                cs = np.r_[0., np.cumsum(np.where(good, val, 0.))]
                cn = np.r_[0, np.cumsum(good)]
                for width in [1800, 7200, 21600]:
                    l = np.searchsorted(at, ds[ix]-width)
                    r = np.searchsorted(at, ds[ix]+width)
                    x.loc[ix, f'arr_taxi_mean_{width}'] = (cs[r]-cs[l])/np.maximum(1, cn[r]-cn[l])
        for rw, subidx in d.loc[ix].groupby('RUNWAY_mvt').groups.items():
            j = np.asarray(subidx)
            st = np.sort(ds[j])
            pos = np.searchsorted(st, ds[j])
            prev = st[np.maximum(pos-1, 0)]
            nex = st[np.minimum(pos+1, len(st)-1)]
            x.loc[j, 'runway_prev_gap'] = ds[j] - prev
            x.loc[j, 'runway_next_gap'] = nex - ds[j]
            x.loc[j, 'runway_count_30min'] = np.searchsorted(st, ds[j]+900)-np.searchsorted(st,ds[j]-900)
    for c in x:
        if x[c].dtype != object:
            x[c] = pd.to_numeric(x[c], errors='coerce').astype('float32').fillna(-999999)
    assert 'BLOCK_TIME_UTC_mvt' not in x and TARGET not in x and ID not in x
    return x, d[[ID, TIME, 'ADEP_mvt', TARGET]].copy()


def prepare():
    OUT.mkdir(exist_ok=True)
    xs, metas = [], []
    for p in sorted(RAW.glob('training*.parquet')) + [RAW/'ranking.parquet']:
        print('features', p.name, flush=True)
        f = pd.read_parquet(p)
        x, meta = features(f)
        meta['source'] = 'ranking' if p.name == 'ranking.parquet' else 'training'
        xs.append(x); metas.append(meta)
        del f; gc.collect()
    x = pd.concat(xs, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    x.to_parquet(OUT/'features.parquet', index=False)
    meta.to_parquet(OUT/'rows.parquet', index=False)
    print('prepared', x.shape, flush=True)


def rmse(y,p):
    return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))


def run(args):
    if args.boost_priority:
        import os
        if os.name=='nt':
            import ctypes
            assert ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1),0x8000)
    print('Loading feature cache',flush=True)
    x = pd.read_parquet(OUT/'features.parquet')
    rows = pd.read_parquet(OUT/'rows.parquet')
    known = rows.source.eq('training') & rows[TARGET].notna()
    month = rows[TIME].dt.month
    train = known & ~month.isin([7,11,12])
    valid = known & month.isin([7,11])
    if args.forward:
        train = known & month.lt(7)
        valid = known & month.eq(7)
    if args.final:
        train = known
    has_proxy=x['mvt_minus_AOBT_3_flt'].gt(-100000)
    if args.known_only:
        train = train & has_proxy
    if args.sample and not args.final:
        idx=np.flatnonzero(train)
        selected=np.random.default_rng(args.seed).choice(idx,min(args.sample,len(idx)),replace=False)
        train[:]=False
        train.iloc[selected]=True
    for feature_path in [args.tfm,args.weather,args.extra]:
        if not feature_path:
            continue
        tfm = pd.read_parquet(feature_path)
        if ID in tfm:
            add = rows[[ID]].merge(tfm, on=ID, how='left', validate='one_to_one').drop(columns=ID)
        else:
            keys=rows[['ADEP_mvt']].assign(hour=rows[TIME].dt.floor('h'))
            add=keys.merge(tfm,on=['ADEP_mvt','hour'],how='left',validate='many_to_one').drop(columns=['ADEP_mvt','hour'])
        for c in add:
            x[c] = add[c].fillna(-999999).astype('float32')
    if args.interactions:
        x['stand_runway']=x['airport_stand'].astype(str)+'_'+x['RUNWAY_mvt'].astype(str)
        x['airport_operator']=x['ADEP_mvt'].astype(str)+'_'+x['AIRCRAFT_OPERATOR_flt'].astype(str)
        x['runway_aircraft']=x['airport_runway'].astype(str)+'_'+x['AIRCRAFT_TYPE_mvt'].astype(str)
        if 'wx_wind_north_knots' in x:
            heading=pd.to_numeric(x.RUNWAY_mvt.str.extract(r'(\d+)',expand=False),errors='coerce')*np.pi/18
            north=x.wx_wind_north_knots.replace(-999999,np.nan)
            east=x.wx_wind_east_knots.replace(-999999,np.nan)
            x['headwind']=(north*np.cos(heading)+east*np.sin(heading)).fillna(-999999).astype('float32')
            x['crosswind']=(east*np.cos(heading)-north*np.sin(heading)).abs().fillna(-999999).astype('float32')
    cats = x.select_dtypes(['object','category']).columns.tolist()
    for c in cats:
        x[c] = x[c].astype('category')
    y = rows[TARGET].to_numpy(float)
    base = np.zeros(len(x))
    if args.residual:
        base = x['mvt_minus_AOBT_3_flt'].to_numpy().copy()
        missing = base < -100000
        base[(base < 0) | (base > 172800)] = 900
        use_schedule = missing & x.ADEP_mvt.eq('LIRF').to_numpy()
        base[use_schedule] = x.loc[use_schedule, 'mvt_minus_SCHED_TIME_UTC_mvt'].to_numpy()
        base[(base < 0) | (base > 172800)] = 900
    start = time.time()
    if args.load:
        model=CatBoostRegressor().load_model(args.load)
    else:
        print('Building pools',int(train.sum()),'train rows',flush=True)
        tr = Pool(x.loc[train], y[train]-base[train], cat_features=cats)
        eval_mask=valid & has_proxy if args.known_only else valid
        va = None if args.final else Pool(x.loc[eval_mask], y[eval_mask]-base[eval_mask], cat_features=cats)
        model = CatBoostRegressor(iterations=args.iterations, depth=args.depth,
            learning_rate=args.rate, l2_leaf_reg=args.l2, loss_function='RMSE',
            task_type=args.device, devices='0' if args.device=='GPU' else None,
            random_seed=args.seed, thread_count=8, border_count=128,
            one_hot_max_size=20, max_ctr_complexity=1, gpu_ram_part=0.55,
            allow_writing_files=True,train_dir=str(OUT/(args.name+'_logs')))
        print('Fitting',flush=True)
        model.fit(tr, eval_set=None if args.final else va,
            early_stopping_rounds=None if args.final else 200, verbose=100)
        model.save_model(str(OUT/(args.name+'.cbm')))
    sel = rows.source.eq('ranking') if args.final else (known & month.eq(12) if args.audit else valid)
    pred = model.predict(x.loc[sel], thread_count=8) + base[sel]
    result = rows.loc[sel, [ID,TIME,'ADEP_mvt',TARGET]].copy()
    result['prediction'] = pred
    result.to_parquet(OUT/(args.name+'_predictions.parquet'), index=False)
    info = {'args':vars(args), 'seconds':time.time()-start, 'trees':model.tree_count_}
    if not args.load:
        info['importance'] = dict(sorted(zip(x.columns, model.feature_importances_), key=lambda t:-t[1]))
    if not args.final and not args.audit:
        result['month'] = result[TIME].dt.month
        info['rmse'] = {str(m):rmse(g[TARGET],g.prediction) for m,g in result.groupby('month')}
        info['airport_rmse'] = {str(a):rmse(g[TARGET],g.prediction) for a,g in result.groupby('ADEP_mvt')}
        info['tune_rmse'] = rmse(result[TARGET],result.prediction)
        info['known_rmse'] = rmse(result.loc[has_proxy[sel].to_numpy(),TARGET],result.loc[has_proxy[sel].to_numpy(),'prediction'])
    (OUT/(args.name+'.json')).write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k!='importance'},indent=2), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--name', default='baseline')
    p.add_argument('--iterations', type=int, default=1800)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--rate', type=float, default=.07)
    p.add_argument('--l2', type=float, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='GPU')
    p.add_argument('--residual', action='store_true')
    p.add_argument('--forward', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--tfm')
    p.add_argument('--weather')
    p.add_argument('--extra')
    p.add_argument('--interactions',action='store_true')
    p.add_argument('--boost-priority',action='store_true')
    p.add_argument('--sample',type=int,default=0)
    p.add_argument('--known-only',action='store_true')
    p.add_argument('--load')
    args=p.parse_args()
    if args.prepare: prepare()
    else: run(args)
