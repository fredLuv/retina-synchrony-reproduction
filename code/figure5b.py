"""Redraw the published Figure 5B from the supplied, attributed vector readings."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np


W, H, DPI = 1200, 1500, 120
BG = '#FAFBF8'
INK = '#172F38'
MUTED = '#475D65'
FAINT = '#D8E0DE'
INDEP = '#626D73'
MODEL = '#166985'
OBS = '#AE4526'
BAND = '#E8F0EB'

readings = np.genfromtxt(BASE/'results/figure5b-digitized.csv',delimiter=',',names=True)
meta = json.loads((BASE/'results/figure5b-readout.json').read_text())
pair = readings['pairwise_per_1000']
retina = readings['observed_per_1000']
independent = readings['independent_per_1000'][:10]
edge_x = meta['independent_floor_crossing_x']

fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI,facecolor=BG)

def fp(px,weight=400):
    return FontProperties(fname=str(BASE / 'code' / 'fonts' / ('SourceSans3-' + {400:'Regular',600:'Semibold',700:'Bold'}[weight] + '.ttf')),size=px*72/DPI)

def text(x,y,s,px,weight=400,color=INK,ha='left',va='top',**kwargs):
    return fig.text(x/W,1-y/H,s,fontproperties=fp(px,weight),color=color,ha=ha,va=va,**kwargs)

def line(x0,y0,x1,y1,color=FAINT,lw=1.5,ls='-'):
    fig.add_artist(Line2D([x0/W,x1/W],[1-y0/H,1-y1/H],transform=fig.transFigure,
                          color=color,lw=lw*72/DPI,ls=ls,solid_capstyle='round'))

def legend_sample(x,y,color,marker=None,dashed=False):
    line(x,y,x+50,y,color=color,lw=4.6,ls=(0,(3,2)) if dashed else '-')
    if marker:
        fig.add_artist(Line2D([(x+25)/W],[1-y/H],transform=fig.transFigure,marker=marker,
                              markersize=9*72/DPI,markerfacecolor=BG if marker=='D' else color,
                              markeredgecolor=color,markeredgewidth=2.3*72/DPI,linestyle='none'))

text(73,52,'How often do 10 of 40',88,700)
text(73,138,'neurons fire together?',88,700)
text(76,250,'Salamander retinal activity in 20-millisecond recording windows.',36,400,MUTED)
line(76,325,1124,325)

text(76,345,'Windows per 1,000 with exactly 10 neurons firing',37,600)
cols=[(76,'Independent model','<0.001',INDEP,None,True),
      (448,'Pairwise model',f'≈{pair[10]:.1f}',MODEL,'D',False),
      (820,'Observed retina',f'≈{retina[10]:.1f}',OBS,'o',False)]
for x,label,value,color,marker,dashed in cols:
    legend_sample(x,437,color,marker,dashed)
    text(x+64,416,label,31,600,color)
    text(x,463,value,80,600,color)
line(76,572,1124,572)

text(76,599,'Windows per 1,000',33,600)
text(1121,604,'Log scale',29,400,MUTED,ha='right')

xleft,xright,ytop,ybottom=176,1119,677,1148
ax=fig.add_axes([xleft/W,1-ybottom/H,(xright-xleft)/W,(ybottom-ytop)/H],facecolor=BG)
ax.set_xlim(-.12,14.12);ax.set_yscale('log');ax.set_ylim(.001,1000)
ax.axvspan(9.56,10.44,color=BAND,ec='none',zorder=0)
ax.axvline(10,color='#93AAA0',lw=1,ls=(0,(2,3)),zorder=1)
ticks=[.001,.01,.1,1,10,100,1000]
ax.set_yticks(ticks,labels=['0.001','0.01','0.1','1','10','100','1,000'])
ax.yaxis.set_minor_locator(NullLocator());ax.set_xticks(np.arange(0,15,2))
ax.grid(axis='y',color=FAINT,linewidth=.8,zorder=0)
for spine in ax.spines.values():spine.set_visible(False)
ax.spines['bottom'].set_visible(True);ax.spines['bottom'].set_color('#9EABA9')
ax.spines['bottom'].set_linewidth(.85)
ax.tick_params(axis='x',length=0,pad=9);ax.tick_params(axis='y',length=0,pad=11)
for t in ax.get_yticklabels():t.set_fontproperties(fp(30));t.set_color(MUTED)
for val,t in zip(ax.get_xticks(),ax.get_xticklabels()):
    t.set_fontproperties(fp(31,700 if val==10 else 400));t.set_color(INK if val==10 else MUTED)

ind_x=list(range(10))+[edge_x]
ind_y=list(independent)+[.001]
ax.plot(ind_x,ind_y,ls=(0,(5,3)),c=INDEP,lw=2.8,dash_capstyle='round',zorder=3)
ax.plot(edge_x,.001,marker='v',ms=7.8,markerfacecolor=INDEP,markeredgecolor=BG,
        markeredgewidth=.9,zorder=5,clip_on=False)
ks=np.arange(15)
ax.plot(ks,pair[:15],c=MODEL,lw=3,marker='D',ms=4.6,mfc=BG,mec=MODEL,mew=1.1,zorder=4)
ax.plot(ks,retina[:15],c=OBS,lw=3,marker='o',ms=4.9,mfc=OBS,mec=BG,mew=.7,zorder=4)
ax.plot(10,pair[10],marker='D',ms=9,mfc=BG,mec=MODEL,mew=2,zorder=6)
ax.plot(10,retina[10],marker='o',ms=9.5,mfc=OBS,mec=BG,mew=1.15,zorder=7)

text((xleft+xright)/2,1214,'Number of neurons firing in the same window',35,600,ha='center')
line(76,1290,1124,1290)
text(76,1310,'The gray curve falls off-scale before 10. The paper reports about 10,000×.',30,400,MUTED)
text(76,1346,'Whether the authors measured that ratio for the group shown is unknown.',30,400,MUTED)
text(76,1413,'Central curves adapted from Tkačik, Marre, Amodei et al. (2014), Fig. 5B · CC BY 4.0',25,400,MUTED)
text(76,1446,'doi.org/10.1371/journal.pcbi.1003408.g005',25,600,MUTED)

out=BASE/'figures/figure-5b-linkedin.png'
fig.savefig(out,dpi=DPI,facecolor=BG,metadata={
    'Title':'How often do 10 of 40 neurons fire together?',
    'Description':'Central estimates redrawn from Figure 5B for 0–14 of 40 salamander retinal neurons firing in a 20-ms window. At exactly 10 active neurons: independent prediction below 0.001 per 1,000 windows, pairwise prediction about 2.2, and recorded frequency about 1.3.',
    'Source':'Tkačik G, Marre O, Amodei D, Schneidman E, Bialek W, Berry MJ II (2014). Searching for Collective Behavior in a Large Network of Sensory Neurons. Figure 5B, https://doi.org/10.1371/journal.pcbi.1003408.g005',
    'License':'Figure adapted from the original under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Colors, units, text, and layout changed. Values are approximate readings of the vector curves.'})
print(out, 'at 10', pair[10], retina[10], 'gray plot edge',edge_x)
