from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import torch
from . import SCHEMA_VERSION
from .config import TrainConfig, config_dict, digest
from .agents.ddqn import DDQN


def source_hash():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[2]
    git = ['git', f'--git-dir={root / ".development-history"}', f'--work-tree={root}'] if (root/'.development-history').exists() else ['git']
    try:
        commit = subprocess.check_output(git+['rev-parse','HEAD'], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(git+['status','--porcelain','--untracked-files=no'], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(source_sha256=source_hash(), commit=commit, dirty=dirty, python=platform.python_version(),
                platform=platform.platform(), packages={n:importlib.metadata.version(n) for n in ('torch','numpy','gymnasium','matplotlib','PyYAML')})


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_model(agent, env_config, path, metadata):
    payload = dict(schema=SCHEMA_VERSION, config=config_dict(env_config, agent.config),
                   weights={k:v.detach().cpu().clone() for k,v in agent.online.state_dict().items()},
                   metadata=metadata)
    atomic_save(payload, path)
    sidecar = {k:v for k,v in payload.items() if k != 'weights'}
    sidecar['checkpoint_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    Path(path).with_suffix('.json').write_text(json.dumps(sidecar, indent=2))


def load_model(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != SCHEMA_VERSION:
        raise ValueError('checkpoint observation schema mismatch')
    cfg = TrainConfig(**payload['config']['training'])
    with torch.random.fork_rng():
        agent = DDQN(cfg, seed=payload['metadata'].get('seed',0), device=device)
    agent.online.load_state_dict(payload['weights'])
    agent.online.eval()
    return agent, payload

