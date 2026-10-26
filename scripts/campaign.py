"""Run the predefined seed matrix. Existing complete runs are checked and skipped."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    p.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    p.add_argument('--steps',type=int,default=250000)
    args=p.parse_args()
    for history,config in ((16,'history_ddqn'),(1,'single_observation_ddqn')):
        for seed in args.seeds:
            run=ROOT/'runs'/'campaign'/f'h{history}_seed{seed}'
            summary=run/'summary.json'
            if summary.exists():
                if json.loads(summary.read_text())['steps']!=args.steps:
                    raise ValueError(f'existing run has different budget: {run}')
            else:
                command=[sys.executable,'-m','radio_rl.train','--config',f'configs/{config}.yaml',
                         '--seed',str(seed),'--steps',str(args.steps),'--device',args.device,'--output',str(run)]
                if (run/'resume.pt').exists(): command+=['--resume',str(run/'resume.pt')]
                subprocess.run(command,cwd=ROOT,check=True)
            dest=ROOT/'artifacts'/'checkpoints';dest.mkdir(parents=True,exist_ok=True)
            for ext in ('pt','json'):
                shutil.copy2(run/f'best.{ext}',dest/f'h{history}_seed{seed}.{ext}')
            # Release compact learning logs so plots do not depend on ignored runs.
            log_dest=ROOT/'artifacts'/'training_logs'/run.name;log_dest.mkdir(parents=True,exist_ok=True)
            for name in ('validation.jsonl','training.jsonl','run.json','summary.json'):
                shutil.copy2(run/name,log_dest/name)


if __name__=='__main__': main()
