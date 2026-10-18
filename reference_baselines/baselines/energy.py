"""Compare JEPA neighbours with energy-only and energy-gap-matched controls.

This is a descriptive known-crystal check. It does not evaluate candidate stability.
"""

from pathlib import Path
import argparse
import json
import platform

import numpy as np
import pandas as pd
import torch

from reference_baselines.core.io import sha256
from reference_baselines.core.retrieval import KS
from reference_baselines.paths import ROOT, RESULTS_DIR


def energy_knn(energy, k=50, batch_size=512):
    """Exact absolute energy-gap search; exclude self and break ties by row ID."""
    values = torch.tensor(energy, dtype=torch.float64)
    indices = np.empty((len(energy), k), dtype=np.int32)
    gaps = np.empty((len(energy), k), dtype=np.float64)
    for start in range(0, len(energy), batch_size):
        stop = min(start + batch_size, len(energy))
        distance = torch.abs(values[start:stop, None] - values[None, :])
        distance[torch.arange(stop - start), torch.arange(start, stop)] = torch.inf
        nearest, ids = distance.topk(k + 1, largest=False, sorted=True)
        nearest, ids = nearest.numpy(), ids.numpy()
        for j in range(stop - start):
            if nearest[j, k - 1] == nearest[j, k]:
                chosen = torch.where(distance[j] <= nearest[j, k - 1])[0].numpy()
                d = distance[j, chosen].numpy()
            else:
                chosen, d = ids[j, :k], nearest[j, :k]
            order = np.lexsort((chosen, d))[:k]
            indices[start + j], gaps[start + j] = chosen[order], d[order]
        if start == 0 or stop == len(energy) or (start // batch_size) % 25 == 0:
            print(f'Energy-only search: {stop:,}/{len(energy):,}', flush=True)
    return indices, gaps


def matched_controls(energy, jepa_ids, bin_width, draws, seed):
    """Uniform controls in the same absolute-energy-gap bin, for each query/edge.

    Both higher- and lower-energy references are eligible. Self is excluded.
    Sampling is with replacement; controls may include an actual JEPA neighbour.
    """
    n, k = jepa_ids.shape
    order = np.lexsort((np.arange(n), energy))
    ordered = energy[order]
    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = np.arange(n)
    rng = np.random.default_rng(seed)
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        query = energy[start:stop, None]
        gap = np.abs(energy[jepa_ids[start:stop]] - query)
        bins = np.floor(gap / bin_width).astype(np.int64)
        low, high = bins * bin_width, (bins + 1) * bin_width
        left_start = np.searchsorted(ordered, query - high, side='right')
        left_stop = np.searchsorted(ordered, query - low, side='right')
        zero = bins == 0
        # Assign exact-energy ties to the upper interval only.
        left_stop = np.where(zero, np.searchsorted(ordered, query, side='left'), left_stop)
        right_start = np.searchsorted(ordered, query + low, side='left')
        right_stop = np.searchsorted(ordered, query + high, side='left')
        left_count = left_stop - left_start
        pool_count = left_count + right_stop - right_start - zero
        if (pool_count < 1).any():
            raise ValueError('Empty energy-matched pool; inspect bin boundaries.')
        offsets = rng.integers(0, pool_count[..., None], size=(stop - start, k, draws))
        choose_left = offsets < left_count[..., None]
        positions_right = right_start[..., None] + offsets - left_count[..., None]
        self_pos = inverse[start:stop, None, None]
        positions_right += zero[..., None] & (positions_right >= self_pos)
        positions = np.where(choose_left, left_start[..., None] + offsets, positions_right)
        controls = order[positions]
        if (controls == np.arange(start, stop)[:, None, None]).any():
            raise AssertionError('Self selected as a control.')
        actual = np.abs(energy[controls] - query[..., None])
        if ((actual < low[..., None] - 1e-12) | (actual > high[..., None] + 1e-12)).any():
            raise AssertionError('Control outside requested energy-gap bin.')
        yield start, stop, controls, pool_count


def gap_stats(values):
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95))}


def bootstrap_mean_ci(values, seed=2026, repeats=500):
    """Paired bootstrap over query rows, not individual graph edges."""
    rng = np.random.default_rng(seed)
    estimates = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(repeats)]
    return np.quantile(estimates, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--bin-width', type=float, default=.01, help='Energy-gap bin width in eV/atom')
    parser.add_argument('--control-draws', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    if args.threads < 1 or args.bin_width <= 0 or args.control_draws < 1:
        parser.error('threads, bin-width, and control-draws must be positive')
    torch.set_num_threads(args.threads)
    out = args.out_dir
    meta_path = args.data_dir / 'metadata.csv'
    rm = pd.read_csv(meta_path)
    saved = pd.read_csv(out / 'reference_rows.csv')
    pd.testing.assert_frame_equal(saved[rm.columns], rm, check_dtype=False)
    jepa_path = out / 'reference_knn_indices.npy'
    ji = np.load(jepa_path)
    energy = rm.formation_energy_per_atom.to_numpy(dtype=np.float64)
    if not np.isfinite(energy).all() or ji.shape != (len(rm), max(KS)):
        raise ValueError('Invalid energies or JEPA-neighbour shape.')
    if ((ji < 0) | (ji >= len(rm))).any() or (ji == np.arange(len(rm))[:, None]).any():
        raise ValueError('Invalid JEPA neighbour IDs or self edges.')
    ei, eg = energy_knn(energy, max(KS))
    assert not (ei == np.arange(len(rm))[:, None]).any()
    assert (np.diff(eg, axis=1) >= 0).all()
    assert (np.diff(np.sort(ei, axis=1), axis=1) > 0).all()
    np.testing.assert_array_equal(eg, np.abs(energy[ei] - energy[:, None]))
    np.save(out / 'energy_knn_indices.npy', ei)
    np.save(out / 'energy_knn_absolute_gaps.npy', eg)
    je = np.abs(energy[ji] - energy[:, None])
    systems = rm.chemical_system.to_numpy()
    sg = rm['spacegroup.number'].to_numpy()
    band_gap = rm.band_gap.to_numpy(dtype=np.float64)
    if not np.isfinite(band_gap).all() or pd.isna(sg).any():
        raise ValueError('Missing band gaps or space group labels.')
    jchem = systems[ji] == systems[:, None]
    echem = systems[ei] == systems[:, None]
    jsg = sg[ji] == sg[:, None]
    esg = sg[ei] == sg[:, None]
    jbg = np.abs(band_gap[ji] - band_gap[:, None])
    ebg = np.abs(band_gap[ei] - band_gap[:, None])
    control_means = {name: np.empty_like(je) for name in ['energy_gap', 'same_system', 'same_spacegroup', 'band_gap_gap']}
    pool_sizes = np.empty_like(ji)
    for start, stop, ids, counts in matched_controls(energy, ji, args.bin_width, args.control_draws, args.seed):
        pool_sizes[start:stop] = counts
        control_means['energy_gap'][start:stop] = np.abs(energy[ids] - energy[start:stop, None, None]).mean(2)
        control_means['same_system'][start:stop] = (systems[ids] == systems[start:stop, None, None]).mean(2)
        control_means['same_spacegroup'][start:stop] = (sg[ids] == sg[start:stop, None, None]).mean(2)
        control_means['band_gap_gap'][start:stop] = np.abs(band_gap[ids] - band_gap[start:stop, None, None]).mean(2)
    detail = rm[['material_id', 'chemical_system', 'formation_energy_per_atom']].rename_axis('reference_row').reset_index()
    summaries = []
    for k in KS:
        overlap = np.array([len(set(a[:k]) & set(b[:k])) for a, b in zip(ji, ei)])
        detail[f'jepa_energy_overlap_count_k{k}'] = overlap
        detail[f'jepa_mean_energy_gap_k{k}'] = je[:, :k].mean(1)
        detail[f'energy_only_mean_energy_gap_k{k}'] = eg[:, :k].mean(1)
        for name, values in control_means.items():
            detail[f'energy_matched_{name}_k{k}'] = values[:, :k].mean(1)
        summaries.append({
            'k': k, 'mean_neighbour_set_overlap_fraction': float(overlap.mean() / k),
            'fraction_queries_with_no_overlap': float((overlap == 0).mean()),
            'jepa_energy_gap_eV_per_atom': gap_stats(je[:, :k]),
            'energy_only_gap_eV_per_atom': gap_stats(eg[:, :k]),
            'matched_control_mean_energy_gap_eV_per_atom': float(control_means['energy_gap'][:, :k].mean()),
            'matched_control_mean_pool_size': float(pool_sizes[:, :k].mean()),
            'matched_control_fraction_edges_with_pool_size_one': float((pool_sizes[:, :k] == 1).mean()),
            'same_chemical_system_fraction': {'jepa': float(jchem[:, :k].mean()), 'energy_only': float(echem[:, :k].mean()),
                                             'energy_gap_matched_random': float(control_means['same_system'][:, :k].mean())},
            'same_spacegroup_fraction': {'jepa': float(jsg[:, :k].mean()), 'energy_only': float(esg[:, :k].mean()),
                                         'energy_gap_matched_random': float(control_means['same_spacegroup'][:, :k].mean())},
            'mean_band_gap_difference_eV': {'jepa': float(jbg[:, :k].mean()), 'energy_only': float(ebg[:, :k].mean()),
                                           'energy_gap_matched_random': float(control_means['band_gap_gap'][:, :k].mean())},
        })
    # Prespecified main comparison at k=10. Positive differences mean more label agreement.
    main_differences = {}
    for name, values in [('same_system', jchem), ('same_spacegroup', jsg), ('band_gap_gap', jbg)]:
        difference = values[:, :10].mean(1) - control_means[name][:, :10].mean(1)
        main_differences[name] = {'jepa_minus_control': float(difference.mean()),
                                 'query_bootstrap_95_percent_interval': bootstrap_mean_ci(difference)}
    # Check exact energy retrieval using a separate full-sort implementation.
    rng = np.random.default_rng(2026)
    for row in rng.choice(len(rm), 128, replace=False):
        direct = np.abs(energy - energy[row]); direct[row] = np.inf
        expected = np.lexsort((np.arange(len(rm)), direct))[:max(KS)]
        np.testing.assert_array_equal(ei[row], expected)
        np.testing.assert_array_equal(eg[row], direct[expected])
    toy = np.array([0., 0., 1., -1., 1e-12])
    ti, tg = energy_knn(toy, 3, batch_size=2)
    for row in range(len(toy)):
        direct = np.abs(toy - toy[row]); direct[row] = np.inf
        expected = np.lexsort((np.arange(len(toy)), direct))[:3]
        np.testing.assert_array_equal(ti[row], expected)
        np.testing.assert_array_equal(tg[row], direct[expected])
    detail.to_csv(out / 'energy_comparison_per_reference.csv', index=False)
    report = {
        'reference_count': len(rm), 'comparisons': summaries, 'main_k10_paired_differences': main_differences,
        'energy_matched_control': {'gap_bin_width_eV_per_atom': args.bin_width, 'draws_per_edge': args.control_draws,
                                  'seed': args.seed, 'sampling': 'uniform within each query and absolute energy-gap bin, with replacement; self excluded'},
        'bootstrap': '500 resamples of query rows; descriptive within-dataset intervals, not independent-training or chemistry-held-out evidence',
        'verification': {'passed': True, 'exhaustive_full_sort_checks': 128, 'tie_fixture_rows': len(toy),
                         'all_controls_checked_for_self_exclusion_and_gap_bins': True},
        'limits': ['Only known crystals are compared; candidate energies are unavailable.',
                   'Space group agreement is a weak structural clue, not proof of the same atomic arrangement.',
                   'Energy-gap bins are approximate matches, not exact energy conditioning.',
                   'Differences beyond the energy controls do not establish structural relevance or stability.',
                   'This is a descriptive check of a frozen encoder; these crystals may have been in its pretraining data.'],
        'provenance': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
                       'sha256': {str(p.resolve()): sha256(p) for p in [meta_path, jepa_path, out / 'reference_rows.csv', Path(__file__).resolve()]}},
    }
    (out / 'energy_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'comparisons': summaries, 'main_k10_paired_differences': main_differences, 'verification': report['verification']}, indent=2))


if __name__ == '__main__':
    main()
