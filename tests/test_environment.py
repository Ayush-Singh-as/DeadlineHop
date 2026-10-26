from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from radio_rl.config import EnvConfig, TrainConfig
from radio_rl.env import RadioEnv
from radio_rl.interference import Interference
from radio_rl.observations import History
from radio_rl.packets import Packet


def test_gym_contract_and_terminal_accounting():
    env = RadioEnv(EnvConfig(arrival_slots=20, family="none"))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(9)
    for _ in range(env.cfg.horizon):
        obs, _, done, truncated, _ = env.step(int(rng.integers(9)))
        assert env.observation_space.contains(obs)
        assert not truncated
    assert done and not env.queue
    c = env.counts
    assert c['generated'] == c['delivered'] + c['expired'] + c['overflow']
    with pytest.raises(RuntimeError):
        env.step(0)


def test_deadline_one_and_perfect_service():
    cfg = EnvConfig(arrival_slots=10, deadlines=(1,), arrival_probability=1, clean_success=1, family="none")
    env = RadioEnv(cfg)
    obs, _ = env.reset(seed=1)
    channel = int(obs[:8].argmax())
    total = 0
    for _ in range(cfg.horizon):
        _, r, _, _, _ = env.step(channel)
        total += r
    assert env.counts['delivered'] == 10
    assert env.counts['latency_sum'] == 10
    assert total == pytest.approx(9.9)


def test_expiry_inclusive_and_wait():
    env = RadioEnv(EnvConfig(arrival_slots=1, deadlines=(2,), arrival_probability=1, clean_success=1, family="none"))
    obs, _ = env.reset(seed=2)
    channel = int(obs[:8].argmax())
    env.step(8)
    assert len(env.queue) == 1
    _, reward, _, _, _ = env.step(channel)
    assert reward == pytest.approx(.99)
    assert env.counts['expired'] == 0
    assert env.counts['latency_sum'] == 2


def test_retune_consumes_deadline_and_never_emits():
    env = RadioEnv(EnvConfig(arrival_slots=1, deadlines=(1,), arrival_probability=1, family="follower", detection_probability=1))
    obs, _ = env.reset(seed=3)
    _, reward, _, _, info = env.step((int(obs[:8].argmax()) + 1) % 8)
    assert reward == pytest.approx(-2.022)
    assert info['components']['attempt'] == 0
    assert env.interference.pending == []
    assert env.counts['expired'] == 1


def test_edf_and_overflow_conservation():
    env = RadioEnv(EnvConfig(arrival_slots=20, deadlines=(12,), arrival_probability=1, family="none"))
    env.reset(seed=5)
    for _ in range(env.cfg.horizon):
        env.step(8)
    assert env.counts['overflow'] > 0
    assert env.counts['generated'] == env.counts['expired'] + env.counts['overflow']
    env = RadioEnv(EnvConfig(arrival_slots=1, arrival_probability=0, family="none", clean_success=1))
    env.reset(seed=5)
    env.queue = [Packet(0, 0, 2), Packet(1, 0, 4)]
    env.counts['generated'] = 2
    env.step(env.channel)
    assert [p.id for p in env.queue] == [1]


def test_follower_causality_dwell_and_mode_reset():
    cfg = EnvConfig(family="follower", detection_probability=1, response_delay=1, response_dwell=2)
    jam = Interference(cfg, 3)
    jam.observe_emission(4, 0)
    assert not jam.mask().any()
    jam.advance()
    assert jam.mask()[4]
    jam.advance()
    assert jam.mask()[4]
    jam.advance()
    assert not jam.mask().any()
    jam = Interference(replace(cfg, family="switching", switch_times=(1,), modes=("follower","burst")), 3)
    jam.observe_emission(4, 0)
    jam.advance()
    assert jam.mode == 'burst' and not jam.mask().any() and not jam.pending


def test_sweep_wrap_and_burst_fixture():
    jam = Interference(EnvConfig(family='sweep', width=2, move_probability=1), 1)
    jam.phase = 7
    assert np.flatnonzero(jam.mask()).tolist() == [0,7]
    jam.advance()
    assert np.flatnonzero(jam.mask()).tolist() == [0,1]
    jam = Interference(EnvConfig(family='burst', burst_entry=1, burst_persistence=0, width=3), 1)
    assert not jam.mask().any()
    jam.advance()
    assert jam.mask().sum() == 3
    jam.advance()
    assert not jam.mask().any()


def test_same_seed_actions_and_exogenous_pairing():
    a, b = RadioEnv(), RadioEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    np.testing.assert_array_equal(oa, ob)
    for act in [0,0,8,5,5,2,2]:
        ta, tb = a.step(act), b.step(act)
        np.testing.assert_array_equal(ta[0], tb[0])
        assert ta[1:] == tb[1:]
    a.step(8)
    b.step(b.channel)
    np.testing.assert_array_equal(a.arrivals, b.arrivals)
    np.testing.assert_array_equal(a.success_draws, b.success_draws)


def test_hidden_mask_does_not_enter_observation():
    a, b = RadioEnv(EnvConfig(family='sweep')), RadioEnv(EnvConfig(family='sweep'))
    oa, _ = a.reset(seed=1)
    ob, _ = b.reset(seed=1)
    b.interference.phase = (a.interference.phase + 4) % 8
    np.testing.assert_array_equal(oa, ob)
    assert not np.array_equal(a.interference.mask(), b.interference.mask())
    history = History(16)
    x = history.reset(oa).reshape(16,48)
    assert not x[:-1].any()
    assert x[-1,47] == 0


def test_zero_arrival_ratios_are_missing():
    env = RadioEnv(EnvConfig(arrival_probability=0))
    env.reset(seed=1)
    env.step(8)
    assert env.metrics()['delivery_fraction'] is None
    assert env.metrics()['energy_per_delivery'] is None
    assert env.counts['return_sum'] == 0


def test_invalid_config_and_action():
    with pytest.raises(ValueError): EnvConfig(response_delay=0)
    with pytest.raises(ValueError): EnvConfig(deadlines=(13,))
    with pytest.raises(ValueError): TrainConfig(history=0)
    env = RadioEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError): env.step(-1)

