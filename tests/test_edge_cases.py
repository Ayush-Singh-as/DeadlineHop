from dataclasses import replace
import numpy as np
import pytest
from radio_rl.config import EnvConfig
from radio_rl.env import RadioEnv
from radio_rl.interference import Interference
from radio_rl.scenarios import sample_config


@pytest.mark.parametrize('family',['none','iid','sweep','burst','follower','switching'])
def test_all_families_preserve_bounds_and_counts(family):
    rng=np.random.default_rng(18)
    for seed in range(3):
        env=RadioEnv(sample_config(rng,family=family))
        obs,_=env.reset(seed=seed)
        for _ in range(env.cfg.horizon):
            obs,reward,done,_,_=env.step(int(rng.integers(9)))
            assert np.isfinite(reward) and env.observation_space.contains(obs)
            assert len(env.queue)<=8
        assert done and env.counts['generated']==sum(env.counts[k] for k in ('delivered','expired','overflow'))


def test_zero_retune_duration_can_deliver_and_charges_switch():
    cfg=EnvConfig(family='none',clean_success=1,arrival_probability=1,arrival_slots=1,deadlines=(1,),retune_slots=0)
    env=RadioEnv(cfg)
    obs,_=env.reset(seed=1)
    nxt,r,_,_,info=env.step((env.channel+1)%8)
    assert info['components']['attempt']==info['components']['retuned']==info['components']['delivered']==1
    assert r==pytest.approx(1-.02-.012)
    assert nxt[43]==1


def test_follower_detection_delay_and_replacement():
    cfg=EnvConfig(family='follower',detection_probability=.5,response_delay=3,response_dwell=2)
    jam=Interference(cfg,0)
    jam.observe_emission(1,.9)
    assert not jam.pending
    jam.observe_emission(2,.1)
    jam.advance()
    jam.observe_emission(6,.1)
    jam.advance()
    assert not jam.mask().any()
    jam.advance()
    assert np.flatnonzero(jam.mask()).tolist()==[2]
    jam.advance()
    assert np.flatnonzero(jam.mask()).tolist()==[6]
    jam.advance()
    assert jam.mask()[6]
    jam.advance()
    assert not jam.mask().any()


def test_completely_blocked_channel_never_claims_delivery():
    env=RadioEnv(EnvConfig(family='iid',width=8,interference_multiplier=0,arrival_probability=1,arrival_slots=20))
    obs,_=env.reset(seed=1)
    for _ in range(env.cfg.horizon):env.step(env.channel)
    assert env.counts['delivered']==0
    assert env.counts['expired']+env.counts['overflow']==20


def test_reward_components_reconcile():
    env=RadioEnv(EnvConfig(family='switching'))
    env.reset(seed=123)
    rng=np.random.default_rng(4)
    totals=dict(delivered=0,expired=0,overflow=0,retuned=0,attempt=0,energy=0.)
    total=0
    for _ in range(env.cfg.horizon):
        _,r,_,_,info=env.step(int(rng.integers(9)))
        total+=r
        for k,v in info['components'].items():totals[k]+=v
    expected=totals['delivered']-2*(totals['expired']+totals['overflow'])-.02*totals['retuned']-.01*totals['energy']
    assert total==pytest.approx(expected)
    assert totals['energy']==pytest.approx(totals['attempt']+.2*totals['retuned'])
