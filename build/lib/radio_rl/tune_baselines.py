"""Fit on training trajectories and select settings using validation only."""
import argparse
import json
from pathlib import Path
import numpy as np
from .agents.baselines import Baseline, make_baseline
from .config import EnvConfig
from .env import RadioEnv
from .evaluate import rollout, score
from .scenarios import load_manifest, sample_config


def fit_predictor(length, episodes):
    rng=np.random.default_rng(80001)
    total={}
    for episode in range(episodes):
        env=RadioEnv(sample_config(rng))
        obs,_=env.reset(seed=800000+episode)
        behavior=Baseline(('random','round_robin','stay_switch')[episode%3],seed=episode)
        predictor=Baseline('myopic',context_length=length)
        behavior.reset(obs)
        predictor.reset(obs)
        while True:
            action=behavior.act(obs)
            predictor.last_context=predictor.context()
            nxt,r,done,tr,_=env.step(action)
            behavior.observe(obs,action,r,nxt,done,tr)
            predictor.observe(obs,action,r,nxt,done,tr)
            obs=nxt
            if done: break
        for key,counts in predictor.counts.items():
            total.setdefault(key,np.zeros((8,2)))
            total[key]+=counts
    return {k:v.tolist() for k,v in total.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--validation',default='artifacts/scenarios/validation.json')
    p.add_argument('--output',default='artifacts/baseline_settings.json')
    p.add_argument('--fit-episodes',type=int,default=60)
    args=p.parse_args()
    path=Path(args.output)
    if path.exists(): raise FileExistsError(path)
    suite=load_manifest(args.validation)
    if suite['split']!='validation': raise ValueError('baseline tuning requires validation, not test')
    settings={}
    trials=[]
    def assess(name,options,priors=None):
        rows=[]
        selected={name:options, 'predictor_priors':priors or {}}
        for r in suite['records']:
            a=make_baseline(name,r['seed']+900000,selected)
            rows.append(rollout(a,r)[0])
        value,energy=score(rows)
        trials.append(dict(policy=name,options=options,delivery=value,energy=energy))
        print(json.dumps(trials[-1]),flush=True)
        return (value,-energy)
    candidates=[dict(fixed_channel=c) for c in range(8)]
    settings['fixed']=max(candidates,key=lambda x:assess('fixed',x))
    candidates=[dict(window=w,bonus=b) for w in (50,200,500) for b in (.25,.5,1.)]
    settings['sw_ucb']=max(candidates,key=lambda x:assess('sw_ucb',x))
    prior_sets={length:fit_predictor(length,args.fit_episodes) for length in (1,2)}
    length=max((1,2),key=lambda n:assess('myopic',dict(context_length=n),prior_sets[n]))
    settings['myopic']=dict(context_length=length)
    settings['predictor_priors']=prior_sets[length]
    settings['metadata']=dict(validation_sha256=suite['sha256'],fit_episodes=args.fit_episodes,
                              fit_seed=80001,behavior=['random','round_robin','stay_switch'],trials=trials)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(settings,indent=2))


if __name__=='__main__': main()

