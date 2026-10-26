from collections import deque
import json
from pathlib import Path
import numpy as np

POLICIES = ("fixed", "random", "round_robin", "stay_switch", "sw_ucb", "myopic")


class Baseline:
    def __init__(self, name, seed=0, fixed_channel=0, window=200, bonus=.5,
                 context_length=2, priors=None, max_deadline=12, retune_slots=1):
        if name not in POLICIES and name != "wait":
            raise ValueError(f"unknown baseline {name}")
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.fixed_channel = fixed_channel
        self.window = window
        self.bonus = bonus
        self.context_length = context_length
        self.priors = priors or {}
        self.max_deadline = max_deadline
        self.retune_slots = retune_slots

    def reset(self, obs):
        self.t = 0
        self.attempts = deque()
        self.events = deque(maxlen=2)
        self.counts = {k: np.array(v, dtype=float) for k, v in self.priors.items()}
        self.target = None
        self.next_target = int(obs[:8].argmax())
        self.last_context = "root"

    def context(self, extra=0):
        events = list(self.events)[-self.context_length:]
        return "|".join(f"{c}:{s}:{min(self.t + extra - t, 8)}" for t,c,s in events) or "root"

    def estimates(self, key):
        table = self.counts.get(key)
        global_table = self.counts.get("global", np.zeros((8,2)))
        prior = (global_table[:,0] + 1) / (global_table[:,1] + 2)
        if table is None:
            return prior
        return (table[:,0] + 2*prior) / (table[:,1] + 2)

    def act(self, obs):
        channel = int(obs[:8].argmax())
        self.last_context = self.context()
        if self.name == "wait" or not obs[8]:
            return 8
        if self.name == "fixed":
            return self.fixed_channel
        if self.target is not None:
            return self.target
        if self.name in ("random", "round_robin", "stay_switch"):
            self.target = self.next_target
            return self.target
        cost_slots = np.ones(8) * (1 + self.retune_slots)
        cost_slots[channel] = 1
        if self.name == "sw_ucb":
            while self.attempts and self.attempts[0][0] < self.t - self.window:
                self.attempts.popleft()
            successes, counts = np.zeros(8), np.zeros(8)
            for _, c, s in self.attempts:
                successes[c] += s
                counts[c] += 1
            estimate = (successes + 1) / (counts + 2)
            estimate += self.bonus * np.sqrt(np.log(2 + counts.sum()) / (1 + counts))
            estimate = np.minimum(estimate, 1)
        else:
            estimate = self.estimates(self.context(extra=self.retune_slots)).copy()
            estimate[channel] = self.estimates(self.context())[channel]
        # Give a retune followed by service its actual two-slot cost. If the
        # first packet would expire, consider any later EDF packet that survives.
        remaining = obs[8:32].reshape(8,3)
        can_serve = (remaining[:,0,None] * (remaining[:,2,None]*self.max_deadline >= cost_slots[None,:] - 1e-5)).any(axis=0)
        score = estimate/cost_slots - .01/cost_slots - .022*(np.arange(8) != channel)/cost_slots
        score[~can_serve] = -np.inf
        # Prefer the tuned channel on exact ties, avoiding arbitrary extra cost.
        best = int(np.argmax(score))
        if score[channel] >= score[best] - 1e-12:
            best = channel
        self.target = best
        return best

    def observe(self, obs, action, reward, next_obs, terminated, truncated=False):
        outcome = int(next_obs[43:47].argmax())
        if outcome in (0,1):
            channel = int(next_obs[:8].argmax())
            success = int(outcome == 0)
            self.attempts.append((self.t, channel, success))
            for key in set(("global", self.last_context)):
                if key not in self.counts:
                    self.counts[key] = np.zeros((8,2))
                self.counts[key][channel] += (success, 1)
            self.events.append((self.t, channel, success))
            self.target = None
            if self.name == "random":
                self.next_target = int(self.rng.integers(8))
            elif self.name == "round_robin" or (self.name == "stay_switch" and not success):
                self.next_target = (channel + 1) % 8
            else:
                self.next_target = channel
        self.t += 1


def load_settings(path=None):
    return json.loads(Path(path).read_text()) if path else {}


def make_baseline(name, seed, settings=None, max_deadline=12, retune_slots=1):
    settings = settings or {}
    options = settings.get(name, {}).copy()
    if name == "myopic":
        options["priors"] = settings.get("predictor_priors", {})
    return Baseline(name, seed, max_deadline=max_deadline, retune_slots=retune_slots, **options)

