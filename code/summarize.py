"""Create a tidy endpoint table and the held-out comparison figure using finished fits."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
BG, INK, MUTED = '#FAFBF8','#172F38','#475D65'
GRAY, BLUE, GREEN, RED = '#626D73','#166985','#377657','#AE4526'


def endpoint_table():
    baselines=json.loads((RESULTS/'baselines.json').read_text())
    fits={r['group']:r for file in sorted(RESULTS.glob('pairwise-*.json')) for r in [json.loads(file.read_text())]}
    expected=json.loads((ROOT/'protocol.json').read_text())['pairwise_groups_one_based']
    if set(fits)!=set(expected):raise ValueError('The five frozen fits must be complete.')
    rows=[]
    for b in baselines:
        row={'group':b['group'],'test_windows':b['test_windows'],'observed_events':b['test_events'],
             'observed_per_1000':b['observed']*1000,'observed_ci_low_per_1000':b['observed_ci95'][0]*1000,
             'observed_ci_high_per_1000':b['observed_ci95'][1]*1000,'independent_per_1000':b['independent']*1000,
             'movie_conditioned_per_1000':b['movie_conditioned']*1000,'observed_over_independent':b['obs_over_independent'],
             'observed_over_movie_conditioned':b['obs_over_movie_conditioned'],'pairwise_per_1000':None,
             'pairwise_mc_se_per_1000':None,'pairwise_over_observed':None,'pairwise_abs_log_error':None,
             'independent_abs_log_error':abs(math.log(b['independent']/b['observed'])),
             'movie_conditioned_abs_log_error':abs(math.log(b['movie_conditioned']/b['observed']))}
        if b['group'] in fits:
            f=fits[b['group']];health=f['health']
            # Report an endpoint only after checking the defining training constraints.
            if health['train_single_rms']>.001 or health['train_pair_rms']>.00030 or abs(health['mean_sim_minus_train_k'])>.035:
                raise ValueError(f'Group {b["group"]} needs further moment matching before interpretation.')
            row.update(pairwise_per_1000=f['pairwise']*1000,pairwise_mc_se_per_1000=f['mc_se']*1000,
                       pairwise_over_observed=f['pairwise']/b['observed'],pairwise_abs_log_error=abs(math.log(f['pairwise']/b['observed'])))
        rows.append(row)
    with (RESULTS/'endpoint.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return rows


def plot_primary(row):
    W,H,DPI=1200,948,120
    fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI,facecolor=BG)
    def font(px,style='Regular'):
        return FontProperties(fname=ROOT/'code'/'fonts'/f'SourceSans3-{style}.ttf',size=px*72/DPI)
    def text(x,y,s,size,style='Regular',color=INK,align='left'):
        return fig.text(x/W,1-y/H,s,fontproperties=font(size,style),color=color,ha=align,va='top')
    fig.subplots_adjust(0,0,1,1)
    total=row['test_windows']; count=row['observed_events']
    text(68,53,'What predicted the 57 bursts?',72,'Bold')
    text(70,152,'Exactly 10 of 40 cells fired in each burst. The first group had',36,color=MUTED)
    text(70,198,f'{total:,} held-out windows from 99 repeats of the same movie.',36,color=MUTED)
    text(70,296,'Prediction or observation',33,'Semibold')
    text(1130,296,'Number of windows',33,'Semibold',align='right')
    ax=fig.add_axes([.31,.255,.545,.36],facecolor=BG)
    values=[row['independent_per_1000']*total/1000,row['pairwise_per_1000']*total/1000,
            row['movie_conditioned_per_1000']*total/1000,count]
    colors=[GRAY,BLUE,GREEN,RED]; labels=['Independent','Pairwise','Movie-timed¹','Observed retina']; tops=[0,1,2,3]
    maximum=math.ceil(max(values)/50)*50
    ax.set_xlim(0,maximum);ax.set_ylim(3.5,-.5);ax.set_axisbelow(True)
    ax.grid(axis='x',color='#D8E0DE',lw=.9);ax.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    for i,(v,c,label) in enumerate(zip(values,colors,labels)):
        ax.barh(i,v,color=c,height=.27,align='center',zorder=3)
        if v<maximum*.01:ax.scatter([v],[i],s=26,color=c,zorder=4,clip_on=False)
        y=H*(1-(.255+.36*(1-(i+.5)/4)))-23
        text(70,y,label,36,'Semibold',color=c)
        value=f'{v:.2f}' if i==0 else (f'{v:.1f}' if i==2 else f'{v:,.0f}')
        text(1128,y-3,value,40,'Bold',color=c,align='right')
    ax.spines[:].set_visible(False);ax.set_yticks([]);ax.tick_params(axis='x',length=0,pad=13)
    for tick in ax.get_xticklabels():tick.set_fontproperties(font(29));tick.set_color(MUTED)
    text(70,795,'¹ Each cell has its own rate at each moment of the movie.',30,color=MUTED)
    text(70,850,'Reanalysis of Marre, Tkačik, Amodei et al.  ·  20 ms windows',27,color=MUTED)
    text(70,890,'Data  doi.org/10.15479/AT:ISTA:61',26,'Semibold',color=MUTED)
    out=ROOT/'figures/held-out-bursts.png'
    fig.savefig(out,dpi=DPI,facecolor=BG,metadata={'Title':'What predicted the 57 bursts?','Source':'https://research-explorer.ista.ac.at/record/5562',
        'Description':'Held-out redrawn analysis. Each bar measures expected or recorded windows with exactly ten active cells. Numerical predictions come from training repeats only.'})
    plt.close(fig)

if __name__=='__main__':plot_primary(endpoint_table()[0])
