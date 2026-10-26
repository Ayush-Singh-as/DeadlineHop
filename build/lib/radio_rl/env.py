from dataclasses import asdict
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import EnvConfig
from .interference import Interference
from .observations import observation, WAIT
from .packets import Packet, edf_key


class RadioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, debug=False):
        self.cfg = config or EnvConfig()
        self.debug = debug
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(0, 1, (48,), np.float32)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32))
        streams = np.random.SeedSequence(seed).spawn(6)
        self.arrivals = np.random.default_rng(streams[0]).random(self.cfg.arrival_slots) < self.cfg.arrival_probability
        self.deadlines = np.random.default_rng(streams[1]).choice(self.cfg.deadlines, self.cfg.arrival_slots)
        self.success_draws = np.random.default_rng(streams[2]).random((self.cfg.horizon, 8))
        self.detection_draws = np.random.default_rng(streams[3]).random(self.cfg.horizon)
        self.interference = Interference(self.cfg, streams[4])
        self.channel = int(np.random.default_rng(streams[5]).integers(8))
        self.t = 0
        self.queue = []
        self.done = False
        self.counts = dict(generated=0, delivered=0, expired=0, overflow=0, transmissions=0,
                           successes=0, retunes=0, energy=0., latency_sum=0, return_sum=0., jammed_slots=0)
        self._arrival()
        return observation(self.queue, self.channel, self.t, self.cfg), {}

    def _arrival(self):
        if self.t >= self.cfg.arrival_slots or not self.arrivals[self.t]:
            return 0
        packet = Packet(self.counts["generated"], self.t, int(self.deadlines[self.t]))
        self.counts["generated"] += 1
        if len(self.queue) >= self.cfg.capacity:
            self.counts["overflow"] += 1
            return 1
        self.queue.append(packet)
        self.queue.sort(key=edf_key)
        return 0

    def step(self, action):
        if self.done:
            raise RuntimeError("reset is required before stepping a completed episode")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        action = int(action)
        slot = self.t
        mask = self.interference.mask()
        mode = self.interference.mode
        retuned = int(action != WAIT and action != self.channel)
        if retuned:
            self.channel = action
        attempt = int(action != WAIT and bool(self.queue) and (not retuned or self.cfg.retune_slots == 0))
        delivered = 0
        outcome = 2 if retuned else 3
        latency = 0
        if attempt:
            probability = self.cfg.clean_success * (self.cfg.interference_multiplier if mask[self.channel] else 1.)
            delivered = int(self.success_draws[slot, self.channel] < probability)
            outcome = 0 if delivered else 1
            if delivered:
                packet = self.queue.pop(0)
                latency = slot - packet.arrival + 1
            self.interference.observe_emission(self.channel, self.detection_draws[slot])
        expired = sum(packet.deadline <= slot for packet in self.queue)
        self.queue = [packet for packet in self.queue if packet.deadline > slot]
        energy = attempt + .2 * retuned
        for key, value in dict(delivered=delivered, successes=delivered, expired=expired,
                               transmissions=attempt, retunes=retuned, energy=energy,
                               latency_sum=latency, jammed_slots=int(mask.sum())).items():
            self.counts[key] += value
        self.t += 1
        self.interference.advance()
        overflow = self._arrival()
        reward = delivered - 2 * (expired + overflow) - .02 * retuned - .01 * energy
        self.counts["return_sum"] += reward
        self.done = self.t >= self.cfg.horizon
        assert self.counts["generated"] == self.counts["delivered"] + self.counts["expired"] + self.counts["overflow"] + len(self.queue)
        if self.done:
            assert not self.queue, "terminal packets must have an explicit outcome"
        info = {"components": {"delivered": delivered, "expired": expired, "overflow": overflow,
                               "retuned": retuned, "attempt": attempt, "energy": energy}}
        if self.done:
            info["episode"] = self.metrics()
        if self.debug:
            info["trace"] = dict(slot=slot, mode=mode, mask=mask.astype(int).tolist(), action=action,
                                  channel=self.channel, outcome=outcome, queue=len(self.queue),
                                  **info["components"])
        obs = observation(self.queue, self.channel, self.t, self.cfg, action, outcome)
        return obs, float(reward), self.done, False, info

    def metrics(self):
        c = self.counts
        def ratio(a, b):
            return a / b if b else None
        return dict(c, slots=self.t, delivery_fraction=ratio(c["delivered"], c["generated"]),
                    expiry_fraction=ratio(c["expired"], c["generated"]), overflow_fraction=ratio(c["overflow"], c["generated"]),
                    goodput=ratio(c["delivered"], self.t), latency=ratio(c["latency_sum"], c["delivered"]),
                    energy_per_delivery=ratio(c["energy"], c["delivered"]),
                    retunes_per_100_slots=ratio(100*c["retunes"], self.t),
                    interference_occupancy=ratio(c["jammed_slots"], 8*self.t))
