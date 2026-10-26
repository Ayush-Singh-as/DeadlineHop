import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .agents.baselines import POLICIES, load_settings, make_baseline
from .checkpoint import load_model, source_hash
from .config import EnvConfig
from .env import RadioEnv
from .scenarios import load_manifest, FAMILIES


def rollout(agent, record, trace=False):
    env = RadioEnv(EnvConfig(**record['environment']), debug=trace)
    obs,_ = env.reset(seed=record['seed'])
    agent.reset(obs)
    frames=[]
    while True:
        action = agent.act(obs)
        nxt, reward, terminal, truncated, info = env.step(action)
        agent.observe(obs,action,reward,nxt,terminal,truncated)
        if trace: frames.append(info['trace'])
        obs = nxt
        if terminal or truncated:
            break
    return dict(scenario_id=record['id'], scenario_seed=record['seed'], family=record['group'], **env.metrics()), frames


def evaluate(agent, records):
    return [rollout(agent, record)[0] for record in records]


def score(rows):
    groups = sorted(set(r['family'] for r in rows))
    means=[]
    energies=[]
    for group in groups:
        selected=[r for r in rows if r['family']==group]
        means.append(np.mean([r['delivery_fraction'] for r in selected if r['delivery_fraction'] is not None]))
        energies.extend(r['energy_per_delivery'] for r in selected if r['energy_per_delivery'] is not None)
    return float(np.mean(means)), float(np.mean(energies)) if energies else float('inf')


def write_rows(rows, output):
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'refusing to overwrite result {path}')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--policy',choices=POLICIES)
    p.add_argument('--settings', help='Training/validation-fitted baseline JSON')
    p.add_argument('--suite',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu',choices=['auto','cpu','cuda'])
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.limit is not None and args.limit<1: p.error('--limit must be positive')
    if Path(args.output).exists(): raise FileExistsError(args.output)
    suite=load_manifest(args.suite)
    records=suite['records'][:args.limit] if args.limit else suite['records']
    rows=[]
    if args.checkpoint:
        agent,payload=load_model(args.checkpoint,args.device)
        policy=f"ddqn_h{agent.config.history}"
        seed=payload['metadata']['seed']
        ck_hash=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        for record in records:
            if record['environment']['max_deadline'] != payload['config']['environment']['max_deadline']:
                raise ValueError('checkpoint deadline normalization differs from scenario')
        rows=evaluate(agent,records)
    else:
        policy,seed,ck_hash=args.policy,0,''
        settings=load_settings(args.settings)
        for record in records:
            cfg=record['environment']
            agent=make_baseline(policy,record['seed']+900000,settings,cfg['max_deadline'],cfg['retune_slots'])
            rows.append(rollout(agent,record)[0])
    settings_hash=hashlib.sha256(Path(args.settings).read_bytes()).hexdigest() if args.settings else ''
    evaluator_source_hash=source_hash()
    for row in rows:
        row.update(policy=policy,training_seed=seed,checkpoint_sha256=ck_hash,suite_sha256=suite['sha256'],
                   source_sha256=evaluator_source_hash,settings_sha256=settings_hash)
    write_rows(rows,args.output)
    structured=[r for r in rows if r['family'] in FAMILIES]
    print(json.dumps(dict(policy=policy,episodes=len(rows),structured_delivery=score(structured or rows)[0],output=args.output)))


if __name__=='__main__': main()

