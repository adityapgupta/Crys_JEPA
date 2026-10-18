"""Describe reference graphs and overlap with Crys-JEPA chemical reference sets."""

from itertools import combinations
from pathlib import Path
import argparse
import json

from scipy import sparse
from scipy.sparse.csgraph import connected_components
import numpy as np
import pandas as pd

from reference_baselines.core.retrieval import KS, load_data
from reference_baselines.paths import ROOT, RESULTS_DIR


def describe(values):
    values = np.asarray(values)
    return {name: float(fn(values)) for name, fn in (
        ('mean', np.mean), ('median', np.median), ('p95', lambda v: np.quantile(v, .95)),
        ('max', np.max))}


def chemistry_masks(ref_meta, cand_meta):
    """Encode element sets in two 64-bit words; no chemical labels enter the graph."""
    systems = set(ref_meta.chemical_system) | set(cand_meta.chemical_system)
    elements = sorted({e for s in systems for e in s.split('-')})
    if len(elements) > 128:
        raise ValueError('Element masks support up to 128 elements.')
    positions = {e: i for i, e in enumerate(elements)}
    encoded = {}
    for s in systems:
        words = [0, 0]
        for e in s.split('-'):
            pos = positions[e]
            words[pos // 64] |= 1 << (pos % 64)
        encoded[s] = words
    return [np.array([encoded[s] for s in meta.chemical_system], dtype=np.uint64)
            for meta in (ref_meta, cand_meta)]


def eligible(neighbour_masks, query_masks):
    return np.all((neighbour_masks & query_masks[:, None, :]) == neighbour_masks, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    refs, candidates, rm, cm = load_data(args.data_dir)
    out = args.out_dir
    ri = np.load(out / 'reference_knn_indices.npy')
    rd = np.load(out / 'reference_knn_squared_distances.npy')
    ci = np.load(out / 'candidate_knn_indices.npy')
    cd = np.load(out / 'candidate_knn_squared_distances.npy')
    ref_masks, cand_masks = chemistry_masks(rm, cm)
    _, ref_codes = np.unique(rm.chemical_system, return_inverse=True)
    ref_exact = ref_codes[ri] == ref_codes[:, None]
    ref_eligible = eligible(ref_masks[ri], ref_masks)
    cand_eligible = eligible(ref_masks[ci], cand_masks)
    cand_exact = rm.chemical_system.to_numpy()[ci] == cm.chemical_system.to_numpy()[:, None]
    counts = rm.chemical_system.value_counts().to_dict()
    pool_sizes = {}
    for s in cm.chemical_system.unique():
        es = s.split('-')
        pool_sizes[s] = sum(counts.get('-'.join(sub), 0)
                            for n in range(1, len(es) + 1) for sub in combinations(es, n))
    sizes = cm.chemical_system.map(pool_sizes).to_numpy()
    detail = cm.rename_axis('candidate_row').reset_index().copy()
    detail['crys_jepa_reference_count'] = sizes
    graph_summaries, retrieval_summaries, hubs = [], [], []
    energy = rm.formation_energy_per_atom.to_numpy()
    rng = np.random.default_rng(1234)
    a = rng.integers(0, len(refs), 100000)
    b = (a + rng.integers(1, len(refs), len(a))) % len(refs)
    random_gaps = np.abs(energy[a] - energy[b])
    ref_random_exact = float(np.mean([(counts[s] - 1) / (len(refs) - 1) for s in rm.chemical_system]))
    for k in KS:
        graph = sparse.load_npz(out / f'reference_graph_k{k}.npz')
        ncomp, labels = connected_components(graph, directed=False)
        components = np.bincount(labels)
        directed = sparse.csr_matrix((np.ones(len(refs) * k),
                                     (np.repeat(np.arange(len(refs)), k), ri[:, :k].ravel())),
                                    shape=(len(refs), len(refs)))
        indegree = np.bincount(ri[:, :k].ravel(), minlength=len(refs))
        graph_summaries.append({
            'k': k, 'connected_components': int(ncomp),
            'largest_component_fraction': float(components.max() / len(refs)),
            'undirected_edges': graph.nnz // 2,
            'reciprocal_directed_edge_fraction': float(directed.multiply(directed.T).sum() / (len(refs) * k)),
            'same_chemical_system_fraction': float(ref_exact[:, :k].mean()),
            'random_same_system_fraction': ref_random_exact,
            'same_system_enrichment_over_random': float(ref_exact[:, :k].mean() / ref_random_exact),
            'chemically_eligible_neighbour_fraction': float(ref_eligible[:, :k].mean()),
            'formation_energy_gap_eV_per_atom': describe(np.abs(energy[ri[:, :k]] - energy[:, None])),
            'random_pair_formation_energy_gap_eV_per_atom': describe(random_gaps),
            'directed_in_degree': describe(indegree),
            'top_1_percent_nodes_share_of_incoming_edges': float(
                np.sort(indegree)[-max(1, int(np.ceil(.01 * len(refs)))):].sum() / (len(refs) * k)),
            'gaussian_sigma': float(np.median(np.sqrt(rd[:, k - 1]))),
        })
        overlap = cand_eligible[:, :k].sum(1)
        detail[f'overlap_k{k}'] = overlap
        detail[f'same_system_count_k{k}'] = cand_exact[:, :k].sum(1)
        detail[f'mean_squared_distance_k{k}'] = cd[:, :k].mean(1)
        detail[f'kth_distance_k{k}'] = np.sqrt(cd[:, k - 1])
        for name, mask in [('all', np.ones(len(cm), dtype=bool)), ('vun', cm.vun.to_numpy(dtype=bool))]:
            available = mask & (sizes > 0)
            random_fraction = float(np.mean(sizes[mask] / len(refs)))
            precision = float(np.mean(overlap[mask] / k))
            retrieval_summaries.append({
                'k': k, 'population': name, 'candidate_count': int(mask.sum()),
                'no_original_references_count': int((mask & (sizes == 0)).sum()),
                'original_reference_count': describe(sizes[mask]),
                'mean_overlap_count': float(overlap[mask].mean()),
                'fraction_knn_references_eligible': precision,
                'random_fraction_eligible': random_fraction,
                'eligibility_enrichment_over_random': precision / random_fraction,
                'same_chemical_system_fraction': float(cand_exact[mask, :k].mean()),
                'fraction_candidates_with_zero_overlap': float((overlap[mask] == 0).mean()),
                'mean_original_reference_recall_when_available': float(np.mean(overlap[available] / sizes[available])),
                'mean_jaccard': float(np.mean(overlap[mask] / (sizes[mask] + k - overlap[mask]))),
            })
        for kind, neighbours in [('reference_graph', ri), ('candidate_retrieval', ci)]:
            incoming = np.bincount(neighbours[:, :k].ravel(), minlength=len(refs))
            for position, row in enumerate(np.argsort(-incoming, kind='stable')[:10], 1):
                hubs.append({'k': k, 'source': kind, 'rank': position, 'reference_row': int(row),
                             'material_id': rm.iloc[row].material_id, 'formula': rm.iloc[row].pretty_formula,
                             'chemical_system': rm.iloc[row].chemical_system, 'incoming_count': int(incoming[row])})
    detail.to_csv(out / 'candidate_reference_overlap.csv', index=False)
    pd.DataFrame(hubs).to_csv(out / 'most_retrieved_references.csv', index=False)
    report = {
        'reference_graphs': graph_summaries, 'candidate_retrieval': retrieval_summaries,
        'energy_gap_population': 'directed reference-reference edges; candidate energies are unavailable',
        'random_pair_seed': 1234, 'random_pair_count': len(a),
        'random_comparison': 'uniform reference sampling; these descriptive controls are not a stability evaluation',
        'limits': ['Embedding proximity is not evidence of structural similarity or stability.',
                   'Chemical eligibility is overlap with Crys-JEPA, not a gold relevance label.',
                   'No candidate stability labels, learned reference selection, or spectral analysis yet.'],
    }
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print('k | components | same-system neighbours | mean formation-energy gap')
    for row in graph_summaries:
        print(f"{row['k']} | {row['connected_components']} | {row['same_chemical_system_fraction']:.2%} | "
              f"{row['formation_energy_gap_eV_per_atom']['mean']:.4f} eV/atom")
    print(f'Analysis saved to {out}', flush=True)


if __name__ == '__main__':
    main()
