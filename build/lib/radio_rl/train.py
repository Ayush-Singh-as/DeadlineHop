import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from .agents.ddqn import DDQN
from .agents.replay import Replay
from .checkpoint import atomic_save, provenance, save_model
from .config import load_config, config_dict, digest
from .env import RadioEnv
from .evaluate import evaluate, score
from .scenarios import load_manifest, sample_config


def train(args):
    base,cfg=load_config(args.config)
    if args.steps is not None: cfg=replace(cfg,steps=args.steps)
    root=Path(args.output)
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError(f'output exists: {root}; choose a new run or --resume')
    root.mkdir(parents=True,exist_ok=True)
    suite=load_manifest(args.validation)
    if suite['split']!='validation': raise ValueError('training must never select on the test suite')
    records=suite['records']
    agent=DDQN(cfg,args.seed,args.device)
    replay=Replay(cfg.replay_capacity,cfg.history,np.random.SeedSequence([args.seed,2]))
    scenario_rng=np.random.default_rng(np.random.SeedSequence([args.seed,3]))
    episode=0
    episode_step=0
    step0=0
    best=(-float('inf'),-float('inf'))
    env=None
    obs=None
    if args.resume:
        # Resume files contain our own simulator/replay objects. Never load an
        # untrusted resume file; released inference checkpoints use weights_only.
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        previous=saved['config']
        current=config_dict(base,cfg)
        previous['training']['steps']=current['training']['steps']
        if previous != current or saved['seed']!=args.seed or saved['validation_sha256']!=suite['sha256']:
            raise ValueError('resume configuration/seed/validation mismatch')
        if saved.get('fixed_family',False) != args.fixed_family:
            raise ValueError('resume training-distribution mode mismatch')
        if saved['source_sha256'] != provenance()['source_sha256']:
            raise ValueError('resume source differs; exact continuation requires the original source')
        agent.online.load_state_dict(saved['online'])
        agent.target.load_state_dict(saved['target'])
        agent.optimizer.load_state_dict(saved['optimizer'])
        agent.updates=saved['updates']
        agent.rng.bit_generator.state=saved['policy_rng']
        agent.history.values=saved['history']
        replay=saved['replay']
        scenario_rng.bit_generator.state=saved['scenario_rng']
        env,obs=saved['env'],saved['obs']
        episode,episode_step,step0=saved['episode'],saved['episode_step'],saved['step']
        best=tuple(saved['best'])
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    run_info=dict(seed=args.seed,device=str(agent.device),config=config_dict(base,cfg),
                  validation_sha256=suite['sha256'],provenance=provenance(),resumed_from=args.resume)
    (root/'run.json').write_text(json.dumps(run_info,indent=2))
    start=time.perf_counter()
    latest={}
    def metadata(step,value):
        return dict(seed=args.seed,step=step,validation_delivery=value,device=str(agent.device),
                    elapsed_seconds=time.perf_counter()-start,config_sha256=digest(config_dict(base,cfg)),
                    validation_sha256=suite['sha256'],provenance=run_info['provenance'])
    def validate(step):
        nonlocal best
        evaluator=copy.deepcopy(agent)
        value,energy=score(evaluate(evaluator,records))
        record=dict(step=step,delivery=value,energy_per_delivery=energy,elapsed_seconds=time.perf_counter()-start)
        with (root/'validation.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        if (value,-energy)>best:
            best=(value,-energy)
            save_model(agent,base,root/'best.pt',metadata(step,value))
        print(json.dumps(dict(event='validation',**record)),flush=True)
        return value
    def resume_save(step):
        payload=dict(config=config_dict(base,cfg),seed=args.seed,validation_sha256=suite['sha256'],fixed_family=args.fixed_family,
                     source_sha256=run_info['provenance']['source_sha256'],
                     online=agent.online.state_dict(),target=agent.target.state_dict(),optimizer=agent.optimizer.state_dict(),
                     updates=agent.updates,policy_rng=agent.rng.bit_generator.state,history=agent.history.values.copy(),
                     replay=replay,scenario_rng=scenario_rng.bit_generator.state,env=env,obs=obs,episode=episode,
                     episode_step=episode_step,step=step,best=best,torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if agent.device.type=='cuda' else None)
        last=root/'resume.pt'
        if last.exists(): last.replace(root/'resume_previous.pt')
        atomic_save(payload,last)
    if not args.resume: validate(0)
    with (root/'training.jsonl').open('a') as log:
        for step in range(step0+1,cfg.steps+1):
            if env is None:
                env_cfg=base if args.fixed_family else sample_config(scenario_rng,base)
                env=RadioEnv(env_cfg)
                obs,_=env.reset(seed=int(scenario_rng.integers(0,2**32)))
                agent.reset(obs)
                episode_step=0
            epsilon=1. if step<=cfg.warmup else max(cfg.epsilon_final,1-(1-cfg.epsilon_final)*step/cfg.epsilon_steps)
            action=agent.act(obs,epsilon)
            nxt,reward,terminal,truncated,info=env.step(action)
            replay.add(obs,action,reward,nxt,terminal,episode,episode_step)
            agent.observe(obs,action,reward,nxt,terminal,truncated)
            obs=nxt
            episode_step+=1
            if step>=cfg.warmup and step%cfg.update_every==0:
                latest=agent.update(replay.sample(cfg.batch_size))
            if terminal or truncated:
                record=dict(step=step,episode=episode,family=env.cfg.family,epsilon=epsilon,
                            elapsed_seconds=time.perf_counter()-start,updates=agent.updates,**latest,**env.metrics())
                log.write(json.dumps(record)+'\n')
                log.flush()
                episode+=1
                env=None
            if step%cfg.validation_every==0 or step==cfg.steps:
                current_value=validate(step)
                save_model(agent,base,root/'last.pt',metadata(step,current_value))
                resume_save(step)
    elapsed=time.perf_counter()-start
    summary=dict(steps=cfg.steps,executed_steps=cfg.steps-step0,seconds=elapsed,
                 steps_per_second=(cfg.steps-step0)/elapsed,best_delivery=best[0],device=str(agent.device))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--steps',type=int)
    p.add_argument('--validation',default='artifacts/scenarios/validation.json')
    p.add_argument('--output',required=True)
    p.add_argument('--resume')
    p.add_argument('--fixed-family',action='store_true',help='Diagnostic pilot only: do not randomize training family')
    train(p.parse_args())


if __name__=='__main__': main()
