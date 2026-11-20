"""Post-hoc check of the paper's 277/20 split size with the original 30 frozen groups."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from reproduce import poisson_binomial


ROOT = Path(__file__).resolve().parents[1]


def check_splits():
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    mat = ROOT / 'data' / 'IST-2017-61-v1+1_bint_fishmovie32_100.mat'
    if hashlib.sha256(mat.read_bytes()).hexdigest() != protocol['sha256']:
        raise ValueError('Source file differs from the published analysis bundle.')
    cube = loadmat(mat, variable_names=['bint'])['bint'].astype(np.uint8)
    repeats, cells, bins = cube.shape
    rng = np.random.default_rng(np.random.SeedSequence([protocol['seed'], 277, 20]))
    # Split selection is independent of outcomes. Use the same test repeats for every
    # cell group within a split. Across splits and groups the observations overlap.
    test_ids = np.array([np.sort(rng.choice(repeats, 20, replace=False)) for _ in range(100)])
    cell_counts = cube.sum(axis=2, dtype=np.int64)
    train_rates = (cell_counts.sum(axis=0) - cell_counts[test_ids].sum(axis=1)) / ((repeats-20)*bins)
    ratio = np.empty((len(test_ids), len(protocol['groups_zero_based'])))
    for number, group in enumerate(protocol['groups_zero_based']):
        burst_count_by_repeat = (cube[:, group, :].sum(axis=1) == 10).sum(axis=1)
        held_rate = burst_count_by_repeat[test_ids].sum(axis=1) / (20*bins)
        predicted = poisson_binomial(train_rates[:, group])[:, 10]
        ratio[:, number] = held_rate / predicted
    medians = np.median(ratio, axis=1)
    return {
        'scope': 'Post-hoc stationary independent model sensitivity check. Fixed 30 groups, 100 new random partitions of whole repeats, fit on 277 and evaluate on 20. No original figure IDs or partition are claimed. No pairwise model is refitted in this check.',
        'split_seed': [protocol['seed'], 277, 20],
        'shape': list(cube.shape), 'occupied_cell_bins': int(cube.sum(dtype=np.int64)),
        'proportion_occupied': float(cube.mean()), 'total_cell_bins': int(repeats*cells*bins),
        'first_split_median': float(medians[0]),
        'median_of_split_medians': float(np.median(medians)),
        'range_of_split_medians': [float(medians.min()), float(medians.max())],
        'split_medians': medians.tolist(), 'first_split_test_ids_zero_based': test_ids[0].tolist(),
    }


if __name__ == '__main__':
    result = check_splits()
    output = ROOT / 'results' / 'split-check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('100 split medians:', result['median_of_split_medians'], 'range:', result['range_of_split_medians'])
