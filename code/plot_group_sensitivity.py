"""Show how the ten-cell synchrony ratio varies with the selection of 40 neurons."""
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import NullLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def font(points, face='Regular'):
    return FontProperties(fname=str(ROOT/'code'/'fonts'/f'SourceSans3-{face}.ttf'), size=points)


def main():
    data = ROOT/'results'
    with gzip.open(data/'group-sensitivity.csv.gz', 'rt', newline='') as stream:
        values = np.asarray([float(row['fold_full_k10']) for row in csv.DictReader(stream)])
    result = json.loads((data/'group-sensitivity.json').read_text())
    median = np.median(values)
    threshold = 10_000
    tail = np.count_nonzero(values >= threshold)
    assert len(values) == result['groups'] and tail == result['full_groups_at_least_10000']
    np.testing.assert_allclose(median, result['full_exact']['10']['median'])

    bg, dark, muted, main_color, gold = '#FAFBF8', '#19343A', '#4B6066', '#38786F', '#995320'
    fig = plt.figure(figsize=(10, 7.15), dpi=140, facecolor=bg)
    def label(x, y, text, size, face='Regular', color=dark):
        fig.text(x, y, text, va='top', fontproperties=font(size, face), color=color)

    label(.07, .95, 'How much does the choice of neurons matter?', 28, 'Bold')
    label(.07, .868, 'For each group of 40, count how often exactly ten cells fired together.', 16, color=muted)
    label(.07, .82, 'Then divide that rate by what independent firing predicts.', 16, color=muted)
    ax = fig.add_axes([.11, .215, .84, .458], facecolor=bg)
    ax.set_xscale('log')
    edges = 10**np.arange(1.90, 4.651, .05)
    # Make the threshold an exact bin edge so every colored bin belongs on one side.
    edges[np.isclose(edges, threshold)] = threshold
    counts, bins = np.histogram(values, bins=edges)
    assert counts.sum() == len(values) and counts[bins[:-1] >= threshold].sum() == tail
    colors = [gold if x >= threshold else main_color for x in bins[:-1]]
    ax.bar(bins[:-1], counts, np.diff(bins), align='edge', color=colors, edgecolor=bg, linewidth=.45, zorder=2)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, max(counts)*1.13)
    ax.set_xticks([100, 300, 1000, 3000, 10000, 30000], ['100×','300×','1,000×','3,000×','10,000×','30,000×'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(np.arange(0, max(counts)+1, 200))
    ax.grid(axis='y', color='#D9E1DD', linewidth=.8, zorder=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for t in ax.get_xticklabels()+ax.get_yticklabels():
        t.set_fontproperties(font(13)); t.set_color(muted)
    ax.tick_params(which='both', length=0, pad=7)
    ax.axvline(median, color=dark, linestyle=(0,(4,3)), lw=1.6, zorder=3)
    ax.axvline(threshold, color=gold, linestyle=(0,(4,3)), lw=1.6, zorder=3)
    ax.text(median, 1.035, f'Median {median:,.0f}×', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', color=dark, fontproperties=font(15,'Semibold'))
    ax.text(threshold, 1.035, f'{tail:,} groups at 10,000× or higher', transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', color=gold, fontproperties=font(14,'Semibold'))
    ax.text(-.044, 1.035, 'Groups', transform=ax.transAxes, color=muted, fontproperties=font(13), va='bottom')
    ax.set_xlabel('Recorded rate ÷ independent prediction  (log scale)', fontproperties=font(16,'Semibold'), labelpad=12, color=dark)
    label(.07, .089, '10,000 random, overlapping groups from one salamander retina. Each time window lasts 20 ms.', 13, color=muted)
    label(.07, .050, 'Data: Marre, Tkačik, Amodei et al.  ·  doi.org/10.15479/AT:ISTA:61', 12.5, color=muted)
    fig.savefig(ROOT/'figures'/'group-sensitivity.png', facecolor=bg,
                metadata={'Title':'Ten-cell synchrony across 10,000 groups of 40 neurons',
                          'Source':'Original analysis of the recording at https://doi.org/10.15479/AT:ISTA:61'})
    plt.close(fig)


if __name__ == '__main__':
    main()
