from argparse import Namespace
from dataclasses import asdict
import json
import numpy as np
import torch
from radio_rl.agents.ddqn import DDQN
from radio_rl.checkpoint import save_model,load_model
from radio_rl.config import EnvConfig,TrainConfig,digest
from radio_rl.env import RadioEnv
from radio_rl.scenarios import make_manifest,load_manifest
from radio_rl.train import train


def test_portable_checkpoint_and_schema(tmp_path):
    agent=DDQN(TrainConfig(history=1),seed=8)
    path=tmp_path/'agent.pt'
    save_model(agent,EnvConfig(),path,dict(seed=8))
    loaded,_=load_model(path)
    x=torch.ones(3,48)
    torch.testing.assert_close(agent.online(x),loaded.online(x),rtol=0,atol=0)
    payload=torch.load(path,weights_only=True)
    payload['schema']='incorrect'
    torch.save(payload,path)
    import pytest
    with pytest.raises(ValueError):load_model(path)


def test_manifest_disjoint_and_integrity(tmp_path):
    a,b=make_manifest('validation',2),make_manifest('test',2,2,1)
    assert not set(r['seed'] for r in a['records']) & set(r['seed'] for r in b['records'])
    path=tmp_path/'suite.json';path.write_text(json.dumps(a))
    assert load_manifest(path)['sha256']==a['sha256']
    for record in b['records']:
        cfg=EnvConfig(**record['environment'])
        assert cfg.horizon==512 and cfg.max_deadline==12


def test_full_resume_matches_uninterrupted(tmp_path):
    cfg=tmp_path/'config.yaml'
    cfg.write_text('environment:\n  arrival_slots: 6\n  family: none\ntraining:\n  history: 2\n  warmup: 4\n  batch_size: 4\n  replay_capacity: 100\n  validation_every: 12\n  epsilon_steps: 100\n  target_every: 2\n')
    record=dict(id='val-0',seed=100000,group='none',environment=asdict(EnvConfig(arrival_slots=6,family='none')))
    suite=tmp_path/'validation.json'
    suite.write_text(json.dumps(dict(split='validation',records=[record],sha256=digest([record]))))
    def args(output,steps,resume=None):
        return Namespace(config=str(cfg),seed=3,device='cpu',steps=steps,validation=str(suite),output=str(output),resume=resume,fixed_family=True)
    train(args(tmp_path/'whole',24))
    train(args(tmp_path/'split',12))
    train(args(tmp_path/'split',24,str(tmp_path/'split'/'resume.pt')))
    whole,_=load_model(tmp_path/'whole'/'last.pt')
    split,_=load_model(tmp_path/'split'/'last.pt')
    for a,b in zip(whole.online.parameters(),split.online.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_iid_attempt_probability_matches_analytic_value():
    from radio_rl.agents.baselines import Baseline
    from radio_rl.evaluate import rollout
    delivered=attempts=0
    for seed in range(40):
        cfg=EnvConfig(family='iid',width=4,arrival_probability=1,deadlines=(12,))
        record=dict(id=str(seed),seed=seed,group='iid',environment=asdict(cfg))
        row,_=rollout(Baseline('fixed'),record)
        delivered+=row['successes'];attempts+=row['transmissions']
    expected=.98*(.5+.02*.5)
    sigma=(expected*(1-expected)/attempts)**.5
    assert abs(delivered/attempts-expected)<5*sigma
