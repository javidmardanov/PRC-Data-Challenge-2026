"""Bayesian search of non-differentiable tree parameters on fixed hidden months."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import optuna

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=int,default=4)
    p.add_argument('--name',default='tfm_search')
    p.add_argument('--tfm',default='data/processed/timesfm_hourly_dev.parquet')
    args=p.parse_args()
    study=optuna.create_study(study_name=args.name,direction='minimize',
        storage='sqlite:///'+str(ROOT/'artifacts/search.sqlite').replace('\\','/'),
        sampler=optuna.samplers.TPESampler(seed=2026,n_startup_trials=2),load_if_exists=True)
    if not study.trials:
        for depth,rate,l2,weather in [(8,.06,10.,False),(6,.08,30.,True),(10,.05,30.,True)]:
            study.enqueue_trial(dict(depth=depth,rate=rate,l2=l2,weather=weather))
    def objective(trial):
        params=dict(depth=trial.suggest_categorical('depth',[6,8,10]),
            rate=trial.suggest_float('rate',.025,.12,log=True),
            l2=trial.suggest_float('l2',3,80,log=True),
            weather=trial.suggest_categorical('weather',[False,True]))
        name=f'{args.name}_{trial.number}'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/train.py'),'--name',name,
            '--residual','--known-only','--iterations','1400','--sample','600000','--tfm',args.tfm]
        for k,v in params.items():
            if k!='weather': cmd.extend(['--'+k,str(v)])
        if params['weather']:
            cmd.extend(['--weather','data/processed/weather_hourly.parquet'])
        subprocess.run(cmd,cwd=ROOT,check=True,
            creationflags=subprocess.ABOVE_NORMAL_PRIORITY_CLASS if sys.platform=='win32' else 0)
        result=json.loads((ROOT/f'artifacts/{name}.json').read_text())
        trial.set_user_attr('model',name)
        trial.set_user_attr('trees',result['trees'])
        trial.set_user_attr('monthly_rmse',result['rmse'])
        # With the missing-record specialist fixed, minimizing known-row MSE
        # also minimizes the final full-population MSE.
        return result['known_rmse']
    done=sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective,n_trials=max(0,args.trials-done))
    best=dict(value=study.best_value,params=study.best_params,**study.best_trial.user_attrs)
    (ROOT/f'artifacts/{args.name}_best.json').write_text(json.dumps(best,indent=2)+'\n')
    print(json.dumps(best,indent=2))


if __name__=='__main__': main()
