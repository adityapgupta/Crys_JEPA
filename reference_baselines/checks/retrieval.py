"""Verify retrieval against independent direct-distance calculations and set checks."""

from pathlib import Path
import argparse
import json

from scipy import sparse
import numpy as np
import pandas as pd

from reference_baselines.core.retrieval import KS, exact_knn, load_data
from reference_baselines.paths import ROOT, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    refs, candidates, rm, cm = load_data(args.data_dir)
    out = args.out_dir
    rng = np.random.default_rng(2026)
    max_error = 0.
    # Test equal distances and explicit self-exclusion on a small, known example.
    toy = np.array([[0., 0.], [0., 0.], [1., 0.], [-1., 0.]])
    toy_ids, _ = exact_knn(toy, toy, 2, batch_size=2, exclude_self=True, label='tie fixture')
    assert toy_ids.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1]]
    checked = {}
    for name, queries in [('reference', refs), ('candidate', candidates)]:
        ids = np.load(out / f'{name}_knn_indices.npy')
        d2 = np.load(out / f'{name}_knn_squared_distances.npy')
        assert ids.shape == d2.shape == (len(queries), max(KS))
        assert np.all((ids >= 0) & (ids < len(refs)))
        assert np.isfinite(d2).all() and (d2 >= 0).all()
        assert (np.diff(d2, axis=1) >= 0).all()
        assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
        if name == 'reference':
            assert not (ids == np.arange(len(refs))[:, None]).any()
        sample = rng.choice(len(queries), 64, replace=False)
        for row in sample:
            delta = np.asarray(refs, dtype=np.float64) - np.asarray(queries[row], dtype=np.float64)
            direct = np.einsum('ij,ij->i', delta, delta)
            if name == 'reference':
                direct[row] = np.inf
            expected = np.lexsort((np.arange(len(refs)), direct))[:max(KS)]
            np.testing.assert_array_equal(ids[row], expected)
            np.testing.assert_allclose(d2[row], direct[expected], rtol=1e-9, atol=1e-10)
            max_error = max(max_error, float(np.abs(d2[row] - direct[expected]).max()))
        checked[name] = {'rows_checked_for_shape_order_ids': len(queries),
                         'exhaustive_direct_distance_queries': len(sample)}
    ref_ids = np.load(out / 'reference_knn_indices.npy')
    ref_d2 = np.load(out / 'reference_knn_squared_distances.npy')
    for k in KS:
        graph = sparse.load_npz(out / f'reference_graph_k{k}.npz')
        assert graph.shape == (len(refs), len(refs))
        assert (graph != graph.T).nnz == 0 and not graph.diagonal().any()
        sigma = float(np.median(np.sqrt(ref_d2[:, k - 1])))
        for row in rng.choice(len(refs), 16, replace=False):
            neighbours = ref_ids[row, :k]
            actual = graph[row, neighbours].toarray().ravel()
            np.testing.assert_allclose(actual, np.exp(-ref_d2[row, :k] / (2 * sigma * sigma)), rtol=1e-9, atol=1e-10)
    detail = pd.read_csv(out / 'candidate_reference_overlap.csv')
    pd.testing.assert_frame_equal(detail[cm.columns], cm, check_dtype=False)
    assert np.array_equal(detail.candidate_row, np.arange(len(cm)))
    ref_sets = [set(s.split('-')) for s in rm.chemical_system]
    ci = np.load(out / 'candidate_knn_indices.npy')
    for row in rng.choice(len(cm), 64, replace=False):
        elements = set(cm.iloc[row].chemical_system.split('-'))
        original = {i for i, s in enumerate(ref_sets) if s.issubset(elements)}
        assert detail.iloc[row].crys_jepa_reference_count == len(original)
        for k in KS:
            assert detail.iloc[row][f'overlap_k{k}'] == len(original & set(ci[row, :k]))
    report = {'passed': True, 'retrieval_checks': checked, 'chemistry_set_checks': 64,
              'graphs_checked': list(KS), 'tie_and_self_exclusion_fixture_passed': True,
              'max_squared_distance_absolute_error': max_error,
              'limit': 'Checks calculations and file alignment, not embedding provenance or physical reference quality.'}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
