import numpy as np

OBS_DIM = 48
ACTION_COUNT = 9
WAIT = 8
PREVIOUS_ACTION = slice(34, 43)
PREVIOUS_OUTCOME = slice(43, 47)


def observation(queue, channel, t, config, previous_action=None, outcome=None):
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[channel] = 1
    for i, packet in enumerate(queue):
        obs[8 + 3*i:11 + 3*i] = (1, (t - packet.arrival) / config.max_deadline,
                                  (packet.deadline - t + 1) / config.max_deadline)
    obs[32:34] = ((config.horizon - t) / config.horizon,
                  max(config.arrival_slots - t, 0) / config.arrival_slots)
    if previous_action is not None:
        obs[34 + previous_action] = 1
        obs[43 + outcome] = 1
        obs[47] = 1
    return obs


class History:
    def __init__(self, length):
        self.values = np.zeros((length, OBS_DIM), dtype=np.float32)

    def reset(self, obs):
        self.values.fill(0)
        self.values[-1] = obs
        return self.flat()

    def append(self, obs):
        self.values[:-1] = self.values[1:]
        self.values[-1] = obs
        return self.flat()

    def flat(self):
        return self.values.reshape(-1).copy()

