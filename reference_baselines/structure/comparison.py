"""Check JEPA neighbour geometry against energy-only and matched random controls."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from reference_baselines.baselines.energy import bootstrap_mean_ci, matched_controls
from reference_baselines.core.io import sha256
from reference_baselines.core.retrieval import KS
from reference_baselines.paths import ROOT, PACKAGE_DIR, RESULTS_DIR


def distances(features, query_rows, neighbour_rows):
    return np.clip(1 - np.einsum('bd,bkd->bk', features[query_rows], features[neighbour_rows]), 0, 2)


def row_average(values, mask):
    count = mask.sum(1)
    return np.divide(np.where(mask, values, 0).sum(1), count,
                     out=np.full(len(count), np.nan), where=count > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    out = args.out_dir
    rm = pd.read_csv(args.data_dir / 'metadata.csv')
    saved = pd.read_csv(out / 'reference_rows.csv')
    pd.testing.assert_frame_equal(saved[rm.columns], rm, check_dtype=False)
    ji = np.load(out / 'reference_knn_indices.npy')
    ei = np.load(out / 'energy_knn_indices.npy')
    energy = rm.formation_energy_per_atom.to_numpy(dtype=np.float64)
    systems = rm.chemical_system.to_numpy()
    profiles = ['physical', 'density_normalized']
    features = {name: np.load(out / f'soap_{name}.npy') for name in profiles}
    for values in features.values():
        assert values.shape[0] == len(rm)
        valid = np.isfinite(values).all(1)
        np.testing.assert_allclose(np.linalg.norm(values[valid], axis=1), 1, atol=1e-10)
    n, max_k = ji.shape
    assert ji.shape == ei.shape == (len(rm), max(KS))
    jdist = {name: np.empty((n, max_k)) for name in profiles}
    edist = {name: np.empty((n, max_k)) for name in profiles}
    sums = {name: np.zeros((n, max_k)) for name in profiles}
    counts = {name: np.zeros((n, max_k), dtype=int) for name in profiles}
    outside_sums = {name: np.zeros((n, max_k)) for name in profiles}
    outside_counts = {name: np.zeros((n, max_k), dtype=int) for name in profiles}
    for start in range(0, n, 512):
        stop = min(start + 512, n); query = np.arange(start, stop)
        for name in profiles:
            jdist[name][start:stop] = distances(features[name], query, ji[start:stop])
            edist[name][start:stop] = distances(features[name], query, ei[start:stop])
    for start, stop, controls, _ in matched_controls(energy, ji, .01, 10, 1234):
        query = np.arange(start, stop)
        for draw in range(controls.shape[2]):
            ids = controls[:, :, draw]
            outside = systems[ids] != systems[start:stop, None]
            for name in profiles:
                d = distances(features[name], query, ids)
                valid = np.isfinite(d)
                sums[name][start:stop] += np.where(valid, d, 0)
                counts[name][start:stop] += valid
                outside_sums[name][start:stop] += np.where(valid & outside, d, 0)
                outside_counts[name][start:stop] += valid & outside
    summaries = []
    detail = rm[['material_id', 'chemical_system']].rename_axis('reference_row').reset_index()
    examples = []
    joutside = systems[ji] != systems[:, None]
    for name in profiles:
        control = np.divide(sums[name], counts[name], out=np.full((n, max_k), np.nan), where=counts[name] > 0)
        outside_control = np.divide(outside_sums[name], outside_counts[name],
                                    out=np.full((n, max_k), np.nan), where=outside_counts[name] > 0)
        for k in KS:
            # All three methods use the same complete query rows.
            complete = (np.isfinite(jdist[name][:, :k]).all(1) & np.isfinite(edist[name][:, :k]).all(1)
                        & (counts[name][:, :k] == 10).all(1))
            if not complete.any():
                raise ValueError(f'No complete comparisons for {name}, k={k}')
            j = jdist[name][:, :k].mean(1); e = edist[name][:, :k].mean(1); c = control[:, :k].mean(1)
            difference = (j - c)[complete]
            outside_mask = joutside[:, :k] & np.isfinite(jdist[name][:, :k]) & np.isfinite(outside_control[:, :k])
            oj = row_average(jdist[name][:, :k], outside_mask)
            oc = row_average(outside_control[:, :k], outside_mask)
            outside_valid = np.isfinite(oj) & np.isfinite(oc)
            row = {
                'profile': name, 'k': k, 'complete_query_count': int(complete.sum()),
                'excluded_query_count': int((~complete).sum()),
                'mean_cosine_distance': {'jepa': float(j[complete].mean()), 'energy_only': float(e[complete].mean()),
                                         'energy_gap_matched_random': float(c[complete].mean())},
                'mean_jepa_minus_matched_control': float(difference.mean()),
                'fraction_queries_jepa_closer_than_matched_control': float((difference < 0).mean()),
                'different_system_only': {'query_count': int(outside_valid.sum()),
                    'jepa_edge_count': int(outside_mask.sum()),
                    'jepa_mean_query_distance': float(oj[outside_valid].mean()),
                    'matched_random_mean_query_distance': float(oc[outside_valid].mean()),
                    'note': 'Both compared neighbours differ from the query chemical system; control draws are filtered on this condition.'},
            }
            if k == 10:
                row['query_bootstrap_95_percent_interval_jepa_minus_matched_control'] = bootstrap_mean_ci(difference)
                row['different_system_only']['query_bootstrap_95_percent_interval_jepa_minus_matched_control'] = bootstrap_mean_ci((oj - oc)[outside_valid])
                detail[f'{name}_complete'] = complete
                detail[f'{name}_jepa_distance'] = j
                detail[f'{name}_energy_only_distance'] = e
                detail[f'{name}_matched_control_distance'] = c
                detail[f'{name}_jepa_minus_control'] = j - c
            summaries.append(row)
        # Save close and distant cross-system nearest pairs for later inspection.
        eligible = np.flatnonzero(joutside[:, 0] & np.isfinite(jdist[name][:, 0]))
        ordered = eligible[np.argsort(jdist[name][eligible, 0], kind='stable')]
        for category, rows in [('close_geometry', ordered[:10]), ('distant_geometry', ordered[-10:])]:
            for q in rows:
                r = int(ji[q, 0])
                examples.append({'profile': name, 'category': category, 'query_row': int(q), 'reference_row': r,
                                 'query_id': rm.iloc[q].material_id, 'reference_id': rm.iloc[r].material_id,
                                 'query_formula': rm.iloc[q].pretty_formula, 'reference_formula': rm.iloc[r].pretty_formula,
                                 'formation_energy_gap_eV_per_atom': float(abs(energy[q] - energy[r])),
                                 'geometry_cosine_distance': float(jdist[name][q, 0])})
    # Independent scalar checks for saved pair distances and query/control alignment.
    rng = np.random.default_rng(2026)
    for name in profiles:
        for q in rng.choice(n, 64, replace=False):
            for col in [0, 9, 49]:
                r = ji[q, col]
                expected = np.clip(1 - np.dot(features[name][q], features[name][r]), 0, 2)
                np.testing.assert_allclose(jdist[name][q, col], expected, atol=1e-12)
    detail.to_csv(out / 'structure_comparison_per_reference.csv', index=False)
    pd.DataFrame(examples).to_csv(out / 'structure_pair_examples.csv', index=False)
    paths = [args.data_dir / 'metadata.csv', out / 'reference_rows.csv', out / 'reference_knn_indices.npy',
             out / 'energy_knn_indices.npy', out / 'structure_fingerprint_provenance.json',
             *[out / f'soap_{name}.npy' for name in profiles], Path(__file__).resolve(), PACKAGE_DIR / 'baselines/energy.py']
    report = {
        'reference_count': n, 'primary_k': 10, 'primary_profile': 'density_normalized', 'comparisons': summaries,
        'distance': 'one minus dot product of unit-normalized averaged single-species SOAP vectors; lower is closer',
        'control': {'energy_gap_bin_width_eV_per_atom': .01, 'draws_per_edge': 10, 'seed': 1234},
        'verification': {'passed': True, 'scalar_dot_product_checks': 64 * 3 * len(profiles), 'all_finite_fingerprint_norms_checked': True},
        'bootstrap': '500 query-row resamples at k=10; graph edges are dependent and encoder pretraining may overlap',
        'limits': ['This measures one family of averaged local geometry descriptors, not exact structural equivalence.',
                   'No structure labels or JEPA outputs are used to construct the SOAP fingerprints.',
                   'Element identities are ignored; normalized fingerprints also discard density.',
                   'This does not establish candidate stability or useful reference selection for the screening loop.',
                   'Pair examples are selected extremes, not a representative sample.'],
        'sha256': {str(p.resolve()): sha256(p) for p in paths},
    }
    (out / 'structure_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([s for s in summaries if s['k'] == 10], indent=2))


if __name__ == '__main__':
    main()
