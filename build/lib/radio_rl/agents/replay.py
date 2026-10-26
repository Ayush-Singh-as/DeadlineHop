import numpy as np


class Replay:
    def __init__(self, capacity, history, seed=0):
        self.capacity, self.history = capacity, history
        self.obs = np.empty((capacity, 48), np.float32)
        self.next_obs = np.empty((capacity, 48), np.float32)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, bool)
        self.steps = np.empty(capacity, np.int64)
        self.episode = np.empty(capacity, np.int64)
        self.total = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated, episode, step):
        i = self.total % self.capacity
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.steps[i], self.episode[i] = step, episode
        self.total += 1

    def valid(self, ids):
        ids = np.asarray(ids)
        oldest = max(0, self.total - self.capacity)
        return (ids >= oldest) & (ids < self.total) & (ids - np.minimum(self.steps[ids % self.capacity], self.history-1) >= oldest)

    def batch(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if not self.valid(ids).all():
            raise ValueError("history predecessor evicted or invalid transition")
        slots = ids % self.capacity
        offsets = np.arange(1-self.history, 1)
        gather = (ids[:,None] + offsets) % self.capacity
        present = self.steps[slots,None] + offsets >= 0
        if np.any(present & (self.episode[gather] != self.episode[slots,None])):
            raise ValueError("cross-episode history")
        histories = self.obs[gather].copy()
        histories[~present] = 0
        following = np.concatenate((histories[:,1:], self.next_obs[slots,None,:]), axis=1)
        return (histories.reshape(len(ids), -1), self.actions[slots].copy(), self.rewards[slots].copy(),
                following.reshape(len(ids), -1), self.terminated[slots].copy())

    def sample(self, size):
        if self.total == 0:
            raise ValueError("empty replay")
        chosen = []
        while sum(len(x) for x in chosen) < size:
            ids = self.rng.integers(max(0, self.total-self.capacity), self.total, size*2)
            chosen.append(ids[self.valid(ids)])
        return self.batch(np.concatenate(chosen)[:size])

