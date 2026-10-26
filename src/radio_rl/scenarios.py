"""Frozen, disjoint scenario manifests and training-only sampling."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from .config import EnvConfig, digest

FAMILIES = ("sweep", "burst", "follower", "switching")


def sample_config(rng, base=None, family=None):
    c = base or EnvConfig()
    family = family or str(rng.choice(FAMILIES))
    times = c.switch_times
    modes = c.modes
    if family == "switching":
        low, high = max(1, c.arrival_slots // 5), max(3, 4*c.arrival_slots // 5)
        if high - low < 2:
            raise ValueError("switching episodes need a longer arrival phase")
        gap = max(1, c.arrival_slots // 5)
        while True:
            times = tuple(sorted(int(t) for t in rng.choice(np.arange(low, high), 2, replace=False)))
            if times[1] - times[0] >= gap:
                break
        first = str(rng.choice(FAMILIES[:3]))
        second = str(rng.choice([m for m in FAMILIES[:3] if m != first]))
        third = str(rng.choice([m for m in FAMILIES[:3] if m != second]))
        modes = (first, second, third)
    return replace(c, family=family, arrival_probability=float(rng.uniform(.35, .65)),
                   width=int(rng.integers(2, 5)), direction=int(rng.choice([-1, 1])),
                   move_probability=float(rng.uniform(.25, .75)),
                   burst_entry=float(rng.uniform(.05, .2)), burst_persistence=float(rng.uniform(.6, .9)),
                   detection_probability=float(rng.uniform(.6, 1)),
                   response_delay=int(rng.integers(1, 4)), response_dwell=int(rng.integers(2, 6)),
                   switch_times=times, modes=modes)


def make_manifest(split, per_family=50, controls=50, stress=25):
    if split not in ("validation", "test"):
        raise ValueError("manifest split must be validation or test")
    start = 100_000 if split == "validation" else 200_000
    rng = np.random.default_rng(start)
    records = []
    def add(cfg, group, n):
        for _ in range(n):
            index = len(records)
            records.append(dict(id=f"{split}-{index:04d}", seed=start+index, group=group,
                                environment=asdict(cfg() if callable(cfg) else cfg)))
    for family in FAMILIES:
        add(lambda f=family: sample_config(rng, family=f), family, per_family)
    if split == "test":
        add(EnvConfig(family="none"), "none", controls)
        add(lambda: EnvConfig(family="iid", width=int(rng.choice([2,4]))), "iid", controls)
        add(EnvConfig(family="switching", arrival_probability=.8), "stress_load", stress)
        add(EnvConfig(family="switching", deadlines=(2,4,6)), "stress_deadline", stress)
        add(EnvConfig(family="follower", response_delay=4), "stress_delay", stress)
        add(EnvConfig(family="sweep", move_probability=.9), "stress_speed", stress)
    return dict(version=1, split=split, records=records, sha256=digest(records))


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if data["sha256"] != digest(data["records"]):
        raise ValueError("scenario manifest hash mismatch")
    ids = [r["id"] for r in data["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate scenario ID")
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/scenarios")
    p.add_argument("--test-per-family", type=int, default=50)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for split, count in (("validation", 20), ("test", args.test_per_family)):
        path = root / f"{split}.json"
        data = make_manifest(split, count)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen manifest {path}")
        path.write_text(json.dumps(data, indent=2))
        print(path, len(data["records"]), data["sha256"])


if __name__ == "__main__":
    main()

