import copy
import numpy as np
import torch
from torch.nn import functional as F
from .network import QNetwork
from ..observations import History


def targets(online, target, next_states, rewards, terminated, gamma):
    with torch.no_grad():
        actions = online(next_states).argmax(dim=1, keepdim=True)
        values = target(next_states).gather(1, actions).squeeze(1)
        return rewards + gamma * (~terminated).float() * values


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu or install the matching laptop CUDA build")
    return torch.device(name)


class DDQN:
    def __init__(self, config, seed=0, device="cpu"):
        self.config = config
        self.device = resolve_device(device)
        torch.set_num_threads(config.threads)
        torch.manual_seed(seed)
        self.online = QNetwork(config.history, config.hidden).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.history = History(config.history)
        self.updates = 0

    def reset(self, obs):
        self.history.reset(obs)

    def act(self, obs=None, epsilon=0.):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(9))
        with torch.inference_mode():
            x = torch.as_tensor(self.history.flat(), device=self.device).unsqueeze(0)
            return int(self.online(x).argmax(1).item())

    def observe(self, obs, action, reward, next_obs, terminated, truncated=False):
        self.history.append(next_obs)

    def update(self, batch):
        states, actions, rewards, next_states, terminal = [torch.as_tensor(x, device=self.device) for x in batch]
        y = targets(self.online, self.target, next_states, rewards, terminal, self.config.gamma)
        q = self.online(states).gather(1, actions[:,None]).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return dict(loss=float(loss.detach()), q_mean=float(q.detach().mean()), grad_norm=float(norm))
