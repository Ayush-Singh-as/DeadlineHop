from dataclasses import asdict, dataclass, fields
from pathlib import Path
import hashlib
import json
import yaml


@dataclass
class EnvConfig:
    channels: int = 8
    capacity: int = 8
    arrival_slots: int = 500
    max_deadline: int = 12
    deadlines: tuple[int, ...] = (4, 8, 12)
    arrival_probability: float = 0.6
    clean_success: float = 0.98
    interference_multiplier: float = 0.02
    retune_slots: int = 1
    family: str = "sweep"
    width: int = 3
    move_probability: float = 0.5
    direction: int = 1
    burst_entry: float = 0.1
    burst_persistence: float = 0.8
    detection_probability: float = 0.8
    response_delay: int = 2
    response_dwell: int = 3
    switch_times: tuple[int, ...] = (150, 350)
    modes: tuple[str, ...] = ("sweep", "burst", "follower")

    def __post_init__(self):
        for key in ("deadlines", "switch_times", "modes"):
            setattr(self, key, tuple(getattr(self, key)))
        if self.channels != 8 or self.capacity != 8:
            raise ValueError("radio48-v1 requires eight channels and eight queue entries")
        for key in ("arrival_slots", "max_deadline", "width", "response_delay", "response_dwell"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("arrival_probability", "clean_success", "interference_multiplier", "move_probability", "burst_entry", "burst_persistence", "detection_probability"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be within [0, 1]")
        if not self.deadlines or any(not isinstance(d, int) or not 1 <= d <= self.max_deadline for d in self.deadlines):
            raise ValueError("deadlines must be within the supported normalization range")
        if self.width > self.channels or self.direction not in (-1, 1) or self.retune_slots not in (0, 1):
            raise ValueError("invalid width, direction, or retune duration")
        if self.family not in ("none", "iid", "sweep", "burst", "follower", "switching"):
            raise ValueError("unknown interference family")
        if self.family == "switching":
            if len(self.modes) != len(self.switch_times) + 1 or any(m not in ("sweep", "burst", "follower") for m in self.modes):
                raise ValueError("switch schedule must specify one structured mode per segment")
            if tuple(sorted(set(self.switch_times))) != self.switch_times or any(not 0 < t < self.arrival_slots for t in self.switch_times):
                raise ValueError("switch times must increase within the arrival phase")

    @property
    def horizon(self):
        return self.arrival_slots + self.max_deadline


@dataclass
class TrainConfig:
    history: int = 16
    hidden: int = 128
    steps: int = 250_000
    gamma: float = 0.99
    learning_rate: float = 0.0003
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup: int = 5_000
    update_every: int = 4
    target_every: int = 1_000
    epsilon_steps: int = 100_000
    epsilon_final: float = 0.05
    validation_every: int = 25_000
    threads: int = 1

    def __post_init__(self):
        for f in fields(self):
            if f.name not in ("gamma", "learning_rate", "epsilon_final"):
                if not isinstance(getattr(self, f.name), int) or getattr(self, f.name) < 1:
                    raise ValueError(f"{f.name} must be a positive integer")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_final <= 1 or self.learning_rate <= 0:
            raise ValueError("invalid training probability/rate")
        if self.replay_capacity <= self.history or self.batch_size > self.replay_capacity - self.history:
            raise ValueError("replay capacity too small")


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if set(raw) - {"environment", "training"}:
        raise ValueError(f"unknown configuration sections: {set(raw) - {'environment', 'training'}}")
    return EnvConfig(**raw.get("environment", {})), TrainConfig(**raw.get("training", {}))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_dict(env, train):
    return {"environment": asdict(env), "training": asdict(train)}

