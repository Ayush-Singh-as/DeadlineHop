import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from .checkpoint import load_model
from .evaluate import rollout
from .agents.baselines import make_baseline,load_settings
from .scenarios import load_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--scenario',default='artifacts/scenarios/validation.json')
    p.add_argument('--output',default='artifacts/demo')
    p.add_argument('--settings',default='artifacts/baseline_settings.json')
    args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    records=load_manifest(args.scenario)['records']
    record=next(r for r in records if r['group']=='switching')
    agent,_=load_model(args.checkpoint)
    baseline=make_baseline('sw_ucb',record['seed']+900000,load_settings(args.settings))
    results={}
    traces={}
    for name,policy in [('DDQN',agent),('Window UCB',baseline)]:
        results[name],traces[name]=rollout(policy,record,trace=True)
    (root/'trace.json').write_text(json.dumps(dict(scenario=record,metrics=results,traces=traces),indent=2))
    fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True)
    for ax,(name,trace) in zip(axes,traces.items()):
        masks=np.array([x['mask'] for x in trace]).T
        ax.imshow(masks,origin='lower',aspect='auto',cmap='Reds',vmin=0,vmax=2,extent=(-.5,len(trace)-.5,-.5,7.5))
        tx=[x for x in trace if x['attempt']]
        ax.scatter([x['slot'] for x in tx],[x['channel'] for x in tx],s=9,c=['#127c65' if x['delivered'] else '#313c4a' for x in tx])
        ax.set(ylabel='Channel',title=f"{name}: {100*results[name]['delivery_fraction']:.1f}% on-time delivery")
        for t in record['environment']['switch_times']: ax.axvline(t,color='#8b5b12',ls='--',lw=1)
    axes[-1].set_xlabel('Slot; red background = evaluator-only interference, green = success, dark = failure')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    # Short animation includes the first hidden regime change and uses the same
    # predetermined validation scenario as the complete static trace.
    start=max(0,record['environment']['switch_times'][0]-60)
    frames=[]
    trace=traces['DDQN']
    for stop in range(start+10,min(start+160,len(trace))+1,5):
        fig,ax=plt.subplots(figsize=(7,3))
        segment=trace[start:stop]
        ax.imshow(np.array([x['mask'] for x in segment]).T,origin='lower',aspect='auto',cmap='Reds',vmin=0,vmax=2,extent=(start-.5,stop-.5,-.5,7.5))
        ax.scatter([x['slot'] for x in segment],[x['channel'] for x in segment],s=12,c=['#127c65' if x['delivered'] else '#313c4a' for x in segment])
        ax.set(xlim=(start,start+160),ylim=(-.5,7.5),xlabel='Slot (red interference is evaluator-only)',ylabel='Channel',
               title=f"DDQN | slot {stop-1} | queue {segment[-1]['queue']}")
        fig.tight_layout();fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        plt.close(fig)
    frames[0].save(root/'rollout.gif',save_all=True,append_images=frames[1:],duration=160,loop=0)
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
