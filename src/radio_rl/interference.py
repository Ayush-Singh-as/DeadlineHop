import numpy as np
from .config import EnvConfig


class Interference:
    """Causal interferer; policy-dependent emissions only affect future slots."""

    def __init__(self, config: EnvConfig, seed):
        self.cfg = config
        rng = np.random.default_rng(seed)
        # Fixed per-slot draws keep exogenous randomness paired across policies.
        self.draws = rng.random((config.horizon + 1, 4))
        self.subsets = np.argsort(rng.random((config.horizon + 1, config.channels)), axis=1)
        self.t = 0
        self.mode = config.modes[0] if config.family == "switching" else config.family
        self._initialize()

    def _initialize(self):
        self.phase = int(self.draws[self.t, 0] * self.cfg.channels)
        self.on = False
        self.group = self.subsets[self.t, :self.cfg.width].copy()
        self.target = None
        self.until = -1
        self.pending = []

    def mask(self):
        mask = np.zeros(self.cfg.channels, dtype=bool)
        if self.mode == "iid":
            mask[self.subsets[self.t, :self.cfg.width]] = True
        elif self.mode == "sweep":
            mask[(self.phase + np.arange(self.cfg.width)) % self.cfg.channels] = True
        elif self.mode == "burst" and self.on:
            mask[self.group] = True
        elif self.mode == "follower" and self.target is not None and self.t < self.until:
            mask[self.target] = True
        return mask

    def observe_emission(self, channel, detection_draw):
        if self.mode == "follower" and detection_draw < self.cfg.detection_probability:
            self.pending.append((self.t + self.cfg.response_delay, self.t, int(channel)))

    def advance(self):
        self.t += 1
        if self.cfg.family == "switching" and self.t in self.cfg.switch_times:
            self.mode = self.cfg.modes[self.cfg.switch_times.index(self.t) + 1]
            self._initialize()
            return
        if self.mode == "sweep" and self.draws[self.t, 1] < self.cfg.move_probability:
            self.phase = (self.phase + self.cfg.direction) % self.cfg.channels
        elif self.mode == "burst":
            if self.on:
                self.on = self.draws[self.t, 1] < self.cfg.burst_persistence
            elif self.draws[self.t, 1] < self.cfg.burst_entry:
                self.on = True
                self.group = self.subsets[self.t, :self.cfg.width].copy()
        elif self.mode == "follower":
            due = [event for event in self.pending if event[0] <= self.t]
            self.pending = [event for event in self.pending if event[0] > self.t]
            if due:
                _, _, self.target = max(due, key=lambda event: event[1])
                self.until = self.t + self.cfg.response_dwell
            elif self.t >= self.until:
                self.target = None

