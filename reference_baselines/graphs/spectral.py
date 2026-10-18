"""Describe spectral gaps and reference-property smoothness on the k-NN graphs."""

import json

from scipy import sparse
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from reference_baselines.paths import RESULTS_DIR


def main():
    out = RESULTS_DIR
    rm = pd.read_csv(out / 'reference_rows.csv')
    properties = {'formation_energy': rm.formation_energy_per_atom.to_numpy(), 'band_gap': rm.band_gap.to_numpy()}
    rng = np.random.default_rng(2026)
    results = []
    for k in [5, 10, 20, 50]:
        a = sparse.load_npz(out / f'reference_graph_k{k}.npz')
        degree = np.asarray(a.sum(1)).ravel()
        l = laplacian(a, normed=True)
        entry = {'k': k, 'components': int(connected_components(a, directed=False, return_labels=False)), 'properties': {}}
        for name, values in properties.items():
            f = values - np.average(values, weights=degree)
            def smoothness(x):
                return float((x @ (degree * x - a @ x)) / (degree @ (x * x)))
            observed = smoothness(f)
            random = [smoothness(rng.permutation(f)) for _ in range(30)]
            entry['properties'][name] = {'normalized_dirichlet_ratio': observed,
                                          'permuted_mean': float(np.mean(random)),
                                          'permutations': 30}
        if k == 10:
            try:
                eigenvalues, eigenvectors = eigsh(l, k=6, which='SM', tol=1e-5, maxiter=3000, v0=rng.normal(size=len(rm)))
                order = np.argsort(eigenvalues); eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                entry['eigenvalues'] = eigenvalues.tolist()
                entry['second_eigenvector_energy_spearman'] = float(spearmanr(eigenvectors[:, 1], properties['formation_energy']).statistic)
                np.save(out / 'spectral_k10_vectors.npy', eigenvectors)
                residual = np.linalg.norm(l @ eigenvectors - eigenvectors * eigenvalues, axis=0)
                entry['eigenpair_residual_norms'] = residual.tolist()
            except ArpackNoConvergence as exc:
                entry['eigenvalue_status'] = 'not converged; no spectral gap conclusion'
                entry['converged_eigenvalues'] = exc.eigenvalues.tolist()
        results.append(entry)
        print('Spectral check finished for k =', k, flush=True)
    (out / 'spectral_analysis.json').write_text(json.dumps({'graphs': results,
        'interpretation': 'Smaller Dirichlet ratio means smoother property values on this graph; it does not establish stability relevance.',
        'limits': 'Embedding and graph construction are energy-aware. Smoothness is descriptive, not an independent discovery metric.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
