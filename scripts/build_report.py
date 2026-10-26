"""Generate a two-column, two-content-page report from measured artifacts."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
LABELS={'fixed':'Fixed channel','random':'Random hopping','round_robin':'Cyclic hopping','stay_switch':'Stay/switch',
        'sw_ucb':'Window UCB','myopic':'History predictor','ddqn_h1':'Double DQN H=1','ddqn_h16':'Double DQN H=16'}
FAMILIES=('sweep','burst','follower','switching')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--draft',action='store_true')
    args=p.parse_args()
    metadata=json.loads((ROOT/'report/metadata.json').read_text())
    if (not metadata['team'] or not metadata['repository_url']) and not args.draft:
        raise ValueError('Final report requires actual team names/IDs and repository URL in report/metadata.json; use --draft until supplied')
    summaries=json.loads((ROOT/'artifacts/figures/seed_summary.json').read_text())
    overall=json.loads((ROOT/'artifacts/figures/overall.json').read_text())
    by_policy={r['policy']:r for r in overall}
    if not all(p in by_policy for p in LABELS): raise ValueError('incomplete comparison matrix')
    for policy in ('ddqn_h1','ddqn_h16'):
        if len(by_policy[policy]['seeds'])!=3: raise ValueError('report requires all three trained seeds')
    def metric(policy, key='delivery_fraction', families=FAMILIES):
        return float(np.mean([r[key] for r in summaries if r['policy']==policy and r['family'] in families]))
    best_baseline=max((p for p in by_policy if not p.startswith('ddqn')),key=lambda p:by_policy[p]['mean_delivery'])
    h16=100*by_policy['ddqn_h16']['mean_delivery'];h1=100*by_policy['ddqn_h1']['mean_delivery']
    conventional=100*by_policy[best_baseline]['mean_delivery']
    figure_dir=ROOT/'artifacts/figures'
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.4,2.1),dpi=220)
    for path in sorted((ROOT/'artifacts/training_logs').glob('*/validation.jsonl')):
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        history='16' if path.parent.name.startswith('h16_') else '1'
        ax.plot([r['step']/1000 for r in rows],[100*r['delivery'] for r in rows],
                color='#176b87' if history=='16' else '#d28238',alpha=.65,lw=.9)
    ax.plot([],[],color='#176b87',label='H=16, three seeds');ax.plot([],[],color='#d28238',label='H=1, three seeds')
    ax.set(xlabel='Training steps (thousands)',ylabel='Validation delivery (%)')
    ax.legend(fontsize=7,loc='lower right');fig.tight_layout(pad=.7)
    fig.savefig(figure_dir/'report_learning.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.4,2.15),dpi=220)
    x=np.arange(4)
    for offset,(policy,color) in enumerate(((best_baseline,'#a0a8b3'),('ddqn_h1','#d28238'),('ddqn_h16','#176b87'))):
        vals=[100*metric(policy,families=(fam,)) for fam in FAMILIES]
        ax.bar(x+(offset-1)*.25,vals,width=.24,label=LABELS[policy],color=color)
    ax.set_xticks(x,['Sweep','Burst','Follower','Switch'])
    ax.set(ylabel='Test delivery (%)',ylim=(0,103))
    ax.legend(fontsize=6.8,loc='lower left');fig.tight_layout(pad=.7)
    fig.savefig(figure_dir/'report_delivery.png');plt.close(fig)
    report_dir=ROOT/'report';report_dir.mkdir(exist_ok=True)
    output=report_dir/('report_draft.pdf' if args.draft else 'report.pdf')
    c=canvas.Canvas(str(output),pagesize=(595.28,841.89))
    c.setTitle(metadata['title']);c.setAuthor('; '.join(metadata['team']) or 'Project team - details pending')
    width=249.64; left=40; right=305.64
    style=ParagraphStyle('body',fontName='Helvetica',fontSize=9.1,leading=12,spaceAfter=7,textColor=HexColor('#18232e'))
    caption=ParagraphStyle('caption',parent=style,fontSize=8,leading=10)
    heading=ParagraphStyle('heading',parent=style,fontName='Helvetica-Bold',fontSize=11,leading=14,spaceAfter=6)
    def text(x,y,body,kind=style):
        paragraph=Paragraph(body,kind)
        _,height=paragraph.wrap(width,1000)
        if y-height<42: raise ValueError(f'report column overflow near {body[:60]}')
        paragraph.drawOn(c,x,y-height)
        return y-height-kind.spaceAfter
    def image(x,y,path,height=154):
        c.drawImage(str(path),x,y-height,width=width,height=height,mask='auto')
        return y-height-7
    def footer(page):
        c.setStrokeColor(HexColor('#cbd2d9'));c.line(40,32,555.28,32)
        c.setFont('Helvetica',7);c.setFillColor(HexColor('#53616f'))
        c.drawString(40,21,'Simulation only | independently implemented Double DQN and environment')
        c.drawRightString(555.28,21,str(page))
    def header(page):
        c.setFillColor(HexColor('#153c53'));c.setFont('Helvetica-Bold',15)
        c.drawString(40,800,'Deadline-Aware Wireless Channel Selection')
        c.setFont('Helvetica',10);c.drawString(40,784,'Partial feedback, packet deadlines, and changing interference')
        team='; '.join(metadata['team']) if metadata['team'] else 'DRAFT: team names / IDs awaiting confirmation'
        repo=metadata['repository_url'] or 'Repository: awaiting team repository URL'
        # Header metadata wraps across both columns and must not overflow.
        hs=ParagraphStyle('header',fontName='Helvetica',fontSize=8.5,leading=10)
        ph=Paragraph(escape(team)+'<br/>'+escape(repo),hs)
        _,h=ph.wrap(515,100)
        if h>42: raise ValueError('team/repository header too long; shorten identifiers')
        ph.drawOn(c,40,773-h)
        c.setStrokeColor(HexColor('#cbd2d9'));c.line(40,724,555.28,724)
        return 709
    y=header(1)
    y=text(left,y,'1. Problem and research grounding',heading)
    y=text(left,y,'A wireless sensor link must deliver queued packets before their deadlines while interference changes. Staying on a poor channel wastes service opportunities; hopping consumes time; transmissions can reveal the channel to a delayed follower. These coupled future effects motivate reinforcement learning under partial observation, without implying that conventional policies cannot work.')
    y=text(left,y,'Dynamic multichannel access with selected-channel feedback motivates history-based DQN [1]. Radar anti-jamming research motivates temporal context [2], while hidden anti-jamming motivates action-dependent interference [3]. Their observation and reward models differ from ours. We implement an original simulator and Double DQN [4], rather than reproducing their scores or copying their code.')
    y=text(left,y,'2. Environment and POMDP',heading)
    y=text(left,y,'One coordinated transmitter/receiver pair uses eight channels and an eight-packet queue. Packets arrive in 500 slots with Bernoulli probability sampled from 0.35-0.65; deadlines are 4, 8, or 12 slots. A 12-slot drain completes a 512-slot mission. Earliest-deadline-first service is fixed across every policy. Overflow rejects new packets; expired packets are counted explicitly.')
    y=text(left,y,'The hidden state contains interference mode, masks, phase, timers, and pending follower responses. The 48-value observation contains tuned channel (8), queue presence/age/slack (24), clocks (2), previous action (9), previous outcome (4), and validity (1). H=16 stacks observations into 768 inputs; H=1 uses 48 and still sees the previous event. No mask, mode, or future arrival is provided to the policy.')
    y=text(left,y,'Actions 0-7 choose a channel; action 8 waits. Changing channel occupies one slot without transmitting. Staying attempts one queued packet. Clean success probability is 0.98; interference multiplies it by 0.02. A packet arriving at a with deadline D can be served through a+D-1. Every transition checks generated = delivered + expired + overflow + queued.')
    y=709
    y=text(right,y,'3. Dynamics and learning objective',heading)
    y=text(right,y,'Training equally mixes a correlated sweep, ON/OFF bursts, a delayed follower, and episodes switching between these families. Sweeps block 2-4 adjacent channels and move with probability 0.25-0.75. Bursts block 2-4 channels. The follower detects emissions with probability 0.6-1, responds after 1-3 slots, and occupies its target for 2-5 slots. Current interference is fixed before the action; mode changes reset old pending responses.')
    y=text(right,y,'For delivery s, expiry x, overflow b, retune h, and attempt u, the reward is:<br/><b>r = s - 2(x+b) - 0.02h - 0.01(u+0.2h).</b><br/>Gamma is 0.99. WAIT has no bonus. Costs use normalized units; physical watts or milliseconds are not claimed. The terminal mission clock is observed, and terminal targets do not bootstrap.')
    y=text(right,y,'Two 128-unit ReLU layers output nine Q values. Double DQN selects the next action using the online network and evaluates it using the target network. Training uses Adam (0.0003), Huber loss, batch 64, replay 50,000, 5,000-step warm-up, one update per four steps, and target copies every 1,000 updates. Epsilon reaches 0.05 at 100,000 steps. Histories never cross episode or overwritten replay boundaries.')
    y=image(right,y,figure_dir/'report_learning.png',154)
    y=text(right,y,'Figure 1. Validation delivery for all six independently trained runs. Each history variant receives 250,000 steps per seed. These curves are not final test performance.',caption)
    footer(1);c.showPage()
    y=header(2)
    y=text(left,y,'4. Evaluation protocol',heading)
    y=text(left,y,'We train seeds 11, 22, and 33 for each history length. Eighty fixed validation episodes select each seed\'s checkpoint by family-macro-average on-time delivery, with energy as a tie-breaker. The test manifest is frozen before training: 200 structured episodes, 100 no-interference/IID controls, and 100 stress episodes. Network weights stay frozen in evaluation; adaptive baselines update observed counts within each episode.')
    y=text(left,y,'Baselines are fixed channel, random/cyclic hopping, stay-on-success/switch-on-failure, sliding-window UCB, and a history-conditioned myopic predictor. Hopping commits through retuning to a transmission. Fixed channel and UCB settings use validation only. The predictor fits smoothed context counts on 60 training episodes per candidate context length and selects length on validation. All policies use identical ACK feedback and EDF scheduling.')
    y=text(left,y,'Paired scenarios share arrivals and pre-indexed random draws; reactive interference still responds to each policy\'s own actions. Delivery includes every generated packet, including overflow. We average episodes within families, then families equally within each trained seed. The table reports mean and seed range; three seeds do not support precise claims of statistical significance.')
    table_data=[['Policy','Delivery %','Seed range']]
    for policy in ('fixed','random','round_robin','stay_switch','sw_ucb','myopic','ddqn_h1','ddqn_h16'):
        r=by_policy[policy]
        interval=f"{100*r['minimum']:.1f}-{100*r['maximum']:.1f}" if len(r['seeds'])>1 else '-'
        table_data.append([LABELS[policy],f"{100*r['mean_delivery']:.2f}",interval])
    table=Table(table_data,colWidths=[120,63,66.64])
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTNAME',(0,1),(-1,-1),'Helvetica'),
                              ('FONTSIZE',(0,0),(-1,-1),8),('LEADING',(0,0),(-1,-1),10),
                              ('BACKGROUND',(0,0),(-1,0),HexColor('#e6eef3')),('BOTTOMPADDING',(0,0),(-1,-1),5),
                              ('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,HexColor('#b0bec5')),
                              ('ALIGN',(1,1),(-1,-1),'RIGHT')]))
    _,h=table.wrap(width,1000);table.drawOn(c,left,y-h);y-=h+8
    y=text(left,y,'Table 1. Held-out structured delivery; baseline dashes mean one scenario-indexed evaluation stream, not zero uncertainty. All seeds are retained.',caption)
    y=709
    y=text(right,y,'5. Results and limitations',heading)
    delta=h16-conventional
    verdict='exceeds' if delta>0 else 'trails'
    y=text(right,y,f"H=16 achieves <b>{h16:.2f}%</b> structured delivery and H=1 achieves <b>{h1:.2f}%</b>. The strongest conventional policy is {LABELS[best_baseline]} at <b>{conventional:.2f}%</b>. H=16 {verdict} it by {abs(delta):.2f} percentage points. The H=16 minus H=1 difference is {h16-h1:+.2f} points. These measured results do not establish that longer history is necessary or that RL is the only effective approach.")
    y=image(right,y,figure_dir/'report_delivery.png',158)
    y=text(right,y,'Figure 2. Mean held-out delivery by structured family; full baseline, control, stress, and seed tables are in the repository.',caption)
    y=text(right,y,f"Normalized energy per delivery is {metric('ddqn_h16','energy_per_delivery'):.2f} for H=16 and {metric(best_baseline,'energy_per_delivery'):.2f} for {LABELS[best_baseline]}. H=16 delivery under no interference is {100*metric('ddqn_h16',families=('none',)):.1f}% and under IID interference is {100*metric('ddqn_h16',families=('iid',)):.1f}%. IID checks validate the per-attempt analytical mean; they do not require all controllers to have equal packet delivery under retuning costs.")
    y=text(right,y,f"Stress tests separately increase load to 0.80, shorten deadlines to {{2,4,6}}, extend follower delay to 4, or accelerate sweeps to 0.9. Under shorter deadlines H=1 delivers {100*metric('ddqn_h1',families=('stress_deadline',)):.2f}% versus {100*metric(best_baseline,families=('stress_deadline',)):.2f}% for the predictor. Simulation omits ACK loss, waveform effects, imperfect synchronization, multiple links, and real RF traces. Three seeds limit uncertainty estimates; H=1 and H=16 also differ in parameter count. Results apply to these modeled settings only.")
    y=text(right,y,'6. Reproducibility and conclusion',heading)
    y=text(right,y,'The repository includes original source, tested dependencies, six portable checkpoints, hashed scenarios, per-episode results, training logs, tests, plots, and a fixed-scenario demo. CPU inference and exact same-platform continuation are tested. Conventional predictors remain serious alternatives; the contribution is a reproducible RL formulation and measured comparison under delayed, partially observed control. Codex assisted implementation; the repository records provenance.')
    footer(2);c.showPage()
    c.setFillColor(HexColor('#153c53'));c.setFont('Helvetica-Bold',15);c.drawString(40,800,'References')
    refs=[
      '[1] S. Wang, H. Liu, P. H. Gomes, and B. Krishnamachari. Deep Reinforcement Learning for Dynamic Multichannel Access in Wireless Networks. arXiv:1802.06958, 2018. https://arxiv.org/abs/1802.06958',
      '[2] S. Ak and S. Bruggenwirth. Avoiding Jammers: A Reinforcement Learning Approach. arXiv:1911.08874, 2019. https://arxiv.org/abs/1911.08874',
      '[3] Y. Wang, X. Liu, M. Wang, and Y. Yu. A Hidden Anti-jamming Method Based on Deep Reinforcement Learning. arXiv:2012.12448, 2020. https://arxiv.org/abs/2012.12448',
      '[4] H. van Hasselt, A. Guez, and D. Silver. Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461, 2015. https://arxiv.org/abs/1509.06461']
    y=771
    for ref in refs:y=text(left,y,escape(ref))
    footer(3);c.save()
    assert len(PdfReader(output).pages)==3
    (report_dir/'results_used.json').write_text(json.dumps(dict(overall=overall,best_baseline=best_baseline,
                                                              history_difference_pp=h16-h1,baseline_difference_pp=delta),indent=2))
    print(output)


if __name__=='__main__':main()
