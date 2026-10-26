import numpy as np
import pytest
import torch
from radio_rl.agents.baselines import Baseline
from radio_rl.agents.ddqn import DDQN, targets
from radio_rl.agents.replay import Replay
from radio_rl.config import EnvConfig, TrainConfig
from radio_rl.env import RadioEnv


def test_double_selection_and_terminal_target():
    class Fixed(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.register_buffer('values', torch.tensor(values, dtype=torch.float32))
        def forward(self, x): return self.values.expand(len(x), -1)
    y = targets(Fixed([5,4]), Fixed([1,10]), torch.zeros(2,1), torch.tensor([2.,3.]), torch.tensor([False,True]), .9)
    torch.testing.assert_close(y, torch.tensor([2.9,3.]))
    assert not y.requires_grad


def test_replay_boundaries_overwrite_and_copy():
    replay = Replay(6, 3, 1)
    obs = np.zeros(48, np.float32)
    for i in range(9):
        obs.fill(i)
        replay.add(obs, 0, i, obs+1, i in (2,8), i//3, i%3)
    obs.fill(99)
    assert replay.valid([3,4,5,6,7,8]).all()
    x, _, _, nxt, _ = replay.batch([3,4,6])
    assert np.all(x[0].reshape(3,48)[:,0] == [0,0,3])
    assert np.all(x[1].reshape(3,48)[:,0] == [0,3,4])
    assert np.all(nxt[0].reshape(3,48)[:,0] == [0,3,4])
    replay = Replay(5, 3)
    for i in range(8): replay.add(np.full(48,i,np.float32), 0, 0, np.full(48,i+1,np.float32), False, 0, i)
    assert replay.valid([3,4,5,6,7]).tolist() == [False,False,True,True,True]
    with pytest.raises(ValueError): replay.batch([4])
    assert replay.sample(64)[0].shape == (64,144)


def test_truncation_bootstraps_with_final_observation():
    replay = Replay(10,2)
    replay.add(np.ones(48,np.float32),0,1,np.full(48,7,np.float32),False,0,0)
    replay.add(np.full(48,2,np.float32),0,0,np.full(48,3,np.float32),False,1,0)
    x, _, _, nxt, term = replay.batch([0,1])
    assert not term.any()
    assert np.all(nxt[0].reshape(2,48)[1] == 7)
    assert not x[1].reshape(2,48)[0].any()


def test_optimizer_and_target_sync():
    config = TrainConfig(history=1,target_every=2,batch_size=4)
    agent = DDQN(config)
    initial = [p.clone() for p in agent.target.parameters()]
    batch = (np.ones((4,48),np.float32), np.zeros(4,np.int64), np.ones(4,np.float32),
             np.zeros((4,48),np.float32), np.ones(4,bool))
    result = agent.update(batch)
    assert np.isfinite(list(result.values())).all()
    assert all(p.grad is None for p in agent.target.parameters())
    assert all(torch.equal(a,b) for a,b in zip(initial,agent.target.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(initial,agent.online.parameters()))
    agent.update(batch)
    assert all(torch.equal(a,b) for a,b in zip(agent.online.parameters(),agent.target.parameters()))


@pytest.mark.parametrize('name',['random','round_robin','stay_switch','sw_ucb','myopic'])
def test_baseline_completes_retune_before_new_target(name):
    env = RadioEnv(EnvConfig(family='none',arrival_probability=1,clean_success=1,deadlines=(12,)))
    obs,_ = env.reset(seed=3)
    agent = Baseline(name)
    agent.reset(obs)
    agent.target = (env.channel+1)%8
    a = agent.act(obs)
    nxt,r,t,tr,_ = env.step(a)
    agent.observe(obs,a,r,nxt,t,tr)
    assert agent.act(nxt) == a
    _,_,_,_,info = env.step(agent.act(nxt))
    assert info['components']['attempt'] == 1
