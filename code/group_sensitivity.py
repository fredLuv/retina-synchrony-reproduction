"""Quantify cell-selection effects in the released retina without selecting a matching group."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from reproduce import poisson_binomial


ROOT = Path(__file__).resolve().parents[1]


def pack_cells(rows):
    raw = np.packbits(rows, axis=1, bitorder='little')
    pad = (-raw.shape[1]) % 8
    return np.pad(raw, ((0, 0), (0, pad))).view('<u8')


def summarize(values):
    probabilities = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    return dict(zip(['min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max'],
                    (float(x) for x in np.quantile(values, probabilities))))


def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    design = json.loads((ROOT/'group-sensitivity-plan.json').read_text())
    mat = next((ROOT/'data').glob('*.mat'))
    assert hashlib.sha256(mat.read_bytes()).hexdigest() == protocol['sha256']
    cube = loadmat(mat, variable_names=['bint'])['bint'].astype(np.uint8)
    repeats, cells, bins = cube.shape
    is_test = np.arange(repeats) % 3 == 2
    window_test = np.repeat(is_test, bins)
    windows = np.ascontiguousarray(cube.transpose(0, 2, 1)).reshape(-1, cells)
    viable = windows.sum(1) >= 9  # Omitted windows can contribute to none of the audited counts.
    bit_columns = np.ascontiguousarray(pack_cells(windows[viable]).T)
    test_viable = window_test[viable]
    rng = np.random.default_rng(design['seed'])
    groups = np.stack([np.sort(rng.choice(cells, size=design['group_size'], replace=False))
                       for _ in range(design['n_groups'])])
    selected = np.zeros((len(groups), cells), dtype=np.uint8)
    selected[np.arange(len(groups))[:, None], groups] = 1
    masks = pack_cells(selected)
    all_rates = windows.mean(0)
    training_rates = cube[~is_test].mean(axis=(0, 2))
    independent = poisson_binomial(all_rates[groups]); independent_test = poisson_binomial(training_rates[groups])[:,10]
    full_count = np.zeros((len(groups), 4), dtype=np.int64)
    full_at_least = np.empty(len(groups), dtype=np.int64)
    test_count = np.empty(len(groups), dtype=np.int64)
    for group, mask in enumerate(masks):
        active = (np.bitwise_count(bit_columns[0] & mask[0]) + np.bitwise_count(bit_columns[1] & mask[1])
                  + np.bitwise_count(bit_columns[2] & mask[2]))
        hist = np.bincount(active, minlength=41)
        full_count[group] = hist[9:13]; full_at_least[group] = hist[10:].sum()
        test_count[group] = np.count_nonzero((active == 10) & test_viable)
    # Independent, direct counts and direct polynomial convolution check the bit-packing conventions.
    for j in (0, len(groups)//2, len(groups)-1):
        k = windows[:, groups[j]].sum(axis=1)
        assert np.array_equal(np.bincount(k.astype(int),minlength=41)[9:13],full_count[j])
        assert np.count_nonzero(k >= 10) == full_at_least[j]
        assert np.count_nonzero((k == 10) & window_test) == test_count[j]
        exact = np.ones(1)
        for p in all_rates[groups[j]]: exact = np.convolve(exact, [1-p,p])
        np.testing.assert_allclose(exact,independent[j],atol=0,rtol=1e-12)
    ratios = full_count / len(windows) / independent[:,9:13]
    tail_ratios = full_at_least / len(windows) / independent[:,10:].sum(axis=1)
    held_ratios = test_count / window_test.sum() / independent_test
    group_mean = all_rates[groups].mean(axis=1)
    primary = ratios[:, 1]
    out = ROOT/'results'
    with (out/'group-sensitivity.csv.gz').open('wb') as dest, gzip.GzipFile(fileobj=dest,mode='wb',mtime=0) as compressed, io.TextIOWrapper(compressed, encoding='utf-8', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['group','cell_ids_zero_based','mean_cell_probability','observed_count_full_k10','independent_probability_full_k10','fold_full_k10','observed_count_test_k10','independent_probability_test_k10','fold_test_k10','fold_full_k9','fold_full_k11','fold_full_k12','fold_full_at_least_10'])
        for i,ids in enumerate(groups):
            writer.writerow([i+1,' '.join(str(int(x)) for x in ids),group_mean[i],full_count[i,1],independent[i,10],primary[i],test_count[i],independent_test[i],held_ratios[i],ratios[i,0],ratios[i,2],ratios[i,3],tail_ratios[i]])
    summary={'seed':design['seed'],'groups':len(groups),'windows_full':len(windows),'windows_test':int(window_test.sum()),
             'full_exact':{str(k):summarize(ratios[:,k-9]) for k in range(9,13)},
             'full_at_least_10':summarize(tail_ratios),'test_exact_10':summarize(held_ratios),
             'full_groups_at_least_5000':int(np.count_nonzero(primary>=5000)),
             'full_groups_at_least_10000':int(np.count_nonzero(primary>=10000)),
             'test_groups_at_least_10000':int(np.count_nonzero(held_ratios>=10000)),
             'mean_rate_across_cells':summarize(group_mean),
             'frozen_30_full_median':None}
    held_primary=json.loads((out/'baselines.json').read_text())
    full_ratios=[]
    for i,ids in enumerate(protocol['groups_zero_based']):
        k=windows[:, ids].sum(axis=1)
        full_ratios.append(np.mean(k==10)/poisson_binomial(all_rates[ids])[10])
        test_ratio=np.mean(k[window_test]==10)/poisson_binomial(training_rates[ids])[10]
        assert np.isclose(test_ratio,held_primary[i]['obs_over_independent'],rtol=1e-12,atol=0)
    summary['frozen_30_full_median']=float(np.median(full_ratios))
    (out/'group-sensitivity.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Full-recording exact 10:',summary['full_exact']['10'],flush=True)
    print('Groups with ratios >= 10,000:',summary['full_groups_at_least_10000'],'/',len(groups),flush=True)


if __name__=='__main__': main()
