"""Evaluate released agents/baselines and regenerate figures without retraining."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='reproduced')
    p.add_argument('--limit',type=int,help='Functionality check only; not the complete experiment')
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'use an empty output directory: {out}')
    def run(*parts): subprocess.run([sys.executable,*parts],cwd=ROOT,check=True)
    common=['--suite','artifacts/scenarios/test.json']
    if args.limit: common+=['--limit',str(args.limit)]
    checkpoints=sorted((ROOT/'artifacts/checkpoints').glob('h*_seed*.pt'))
    if len(checkpoints)!=6: raise ValueError('expected six released core checkpoints; run campaign.py first')
    for path in checkpoints:
        run('-m','radio_rl.evaluate','--checkpoint',str(path),*common,'--output',str(out/'results'/f'{path.stem}.csv'))
    for policy in ('fixed','random','round_robin','stay_switch','sw_ucb','myopic'):
        run('-m','radio_rl.evaluate','--policy',policy,'--settings','artifacts/baseline_settings.json',*common,
            '--output',str(out/'results'/f'{policy}.csv'))
    run('-m','radio_rl.plots','--results',str(out/'results'),'--output',str(out/'figures'),'--runs','artifacts/training_logs')


if __name__=='__main__': main()
