import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .scenarios import FAMILIES

LABELS={'fixed':'Fixed','random':'Random','round_robin':'Cyclic','stay_switch':'Stay/switch',
        'sw_ucb':'Window UCB','myopic':'History predictor','ddqn_h1':'DDQN H=1','ddqn_h16':'DDQN H=16'}


def read_results(root):
    rows=[]
    for path in sorted(Path(root).glob('*.csv')):
        with path.open() as f:
            for row in csv.DictReader(f):
                for key in ('delivery_fraction','energy_per_delivery','expiry_fraction','overflow_fraction','goodput','latency'):
                    row[key]=float(row[key]) if row.get(key) else np.nan
                rows.append(row)
    if not rows: raise ValueError('no episode CSV files found')
    keys=[(r['policy'],r['training_seed'],r['scenario_id']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('duplicate policy/seed/scenario results')
    if len(set(r['suite_sha256'] for r in rows))!=1: raise ValueError('mixed test manifests')
    return rows


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['policy'],r['family'],r['training_seed'])].append(r)
    seed_rows=[]
    for (policy,family,seed),members in sorted(groups.items()):
        metrics={m:float(np.nanmean([r[m] for r in members])) for m in
                 ('delivery_fraction','energy_per_delivery','expiry_fraction','overflow_fraction','goodput','latency')}
        seed_rows.append(dict(policy=policy,family=family,training_seed=seed,episodes=len(members),**metrics))
    return seed_rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',default='artifacts/results')
    p.add_argument('--output',default='artifacts/figures')
    p.add_argument('--runs',default='runs/campaign')
    args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rows=read_results(args.results)
    summaries=aggregate(rows)
    (root/'seed_summary.json').write_text(json.dumps(summaries,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    policies=sorted(set(r['policy'] for r in rows),key=lambda p: (p.startswith('ddqn'),p))
    def bars(families,filename):
        fig,axes=plt.subplots(1,len(families),figsize=(max(8,3.2*len(families)),4),sharey=True,squeeze=False)
        for ax,fam in zip(axes[0],families):
            for x,policy in enumerate(policies):
                vals=[100*r['delivery_fraction'] for r in summaries if r['policy']==policy and r['family']==fam]
                if not vals: continue
                color='#176b87' if policy=='ddqn_h16' else '#72b3c4' if policy.startswith('ddqn') else '#a9afb9'
                ax.bar(x,np.mean(vals),color=color)
                ax.scatter(np.repeat(x,len(vals)),vals,s=10,color='#263238',zorder=3)
            ax.set_title(fam.replace('stress_','').replace('_',' ').title())
            ax.set_xticks(range(len(policies)),[LABELS.get(p,p) for p in policies],rotation=65,ha='right')
            ax.set_ylim(0,103)
        axes[0,0].set_ylabel('On-time delivery (%)')
        fig.suptitle('Dots: independently trained seeds; baselines: one paired evaluation stream',fontsize=10)
        fig.tight_layout()
        fig.savefig(root/f'{filename}.png');fig.savefig(root/f'{filename}.pdf');plt.close(fig)
    bars(FAMILIES,'delivery')
    bars(('none','iid'),'controls')
    bars(('stress_load','stress_deadline','stress_delay','stress_speed'),'stress')
    fig,ax=plt.subplots(figsize=(7,3.7))
    for path in sorted(Path(args.runs).glob('*/validation.jsonl')):
        records=[json.loads(line) for line in path.read_text().splitlines()]
        ax.plot([r['step'] for r in records],[100*r['delivery'] for r in records],label=path.parent.name,alpha=.8)
    if ax.lines:
        ax.set(xlabel='Environment steps',ylabel='Validation delivery (%)',title='Frozen validation scenarios; individual training seeds')
        ax.legend(fontsize=7,ncol=2);fig.tight_layout()
        fig.savefig(root/'learning.png');fig.savefig(root/'learning.pdf')
    plt.close(fig)
    # Macro-average each structured family within each trained seed; do not treat
    # hundreds of scenario episodes as independent training repetitions.
    overall=[]
    for policy in policies:
        seeds=sorted(set(r['training_seed'] for r in summaries if r['policy']==policy))
        vals=[np.mean([r['delivery_fraction'] for r in summaries if r['policy']==policy and r['training_seed']==s and r['family'] in FAMILIES]) for s in seeds]
        overall.append(dict(policy=policy,seeds=seeds,seed_delivery=vals,mean_delivery=float(np.mean(vals)),
                            minimum=float(np.min(vals)),maximum=float(np.max(vals))))
    (root/'overall.json').write_text(json.dumps(overall,indent=2))
    print(json.dumps(overall,indent=2))


if __name__=='__main__': main()

