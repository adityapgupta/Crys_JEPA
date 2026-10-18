"""Build exact k-NN reference graphs and retrieve references for candidates."""

from pathlib import Path
import argparse
import hashlib
import json
import platform
import time

from scipy import sparse
import numpy as np
import pandas as pd
import scipy
import torch

from reference_baselines.core.retrieval import KS, exact_knn, load_data
from reference_baselines.paths import ROOT, PACKAGE_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=RESULTS_DIR)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('batch-size and threads must be positive')
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    refs, candidates, ref_meta, cand_meta = load_data(args.data_dir)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ri, rd = exact_knn(refs, refs, max(KS), args.batch_size, True, 'reference graph')
    np.save(out / 'reference_knn_indices.npy', ri)
    np.save(out / 'reference_knn_squared_distances.npy', rd)
    for k in KS:
        rows = np.repeat(np.arange(len(refs)), k)
        cols = ri[:, :k].ravel()
        # One shared scale per graph, defined from the kth-neighbour distances.
        sigma = float(np.median(np.sqrt(rd[:, k - 1])))
        if sigma <= 0:
            raise ValueError('Gaussian graph scale is zero.')
        weights = np.exp(-rd[:, :k].ravel() / (2 * sigma * sigma))
        directed = sparse.csr_matrix((weights, (rows, cols)), shape=(len(refs), len(refs)))
        graph = directed.maximum(directed.T)
        sparse.save_npz(out / f'reference_graph_k{k}.npz', graph)
    ci, cd = exact_knn(candidates, refs, max(KS), args.batch_size, False, 'candidate retrieval')
    np.save(out / 'candidate_knn_indices.npy', ci)
    np.save(out / 'candidate_knn_squared_distances.npy', cd)
    ref_meta.rename_axis('reference_row').reset_index().to_csv(out / 'reference_rows.csv', index=False)
    cand_meta.rename_axis('candidate_row').reset_index().to_csv(out / 'candidate_rows.csv', index=False)

    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()

    paths = sorted(args.data_dir.glob('*.npy')) + sorted(args.data_dir.glob('*.csv'))
    paths += sorted((PACKAGE_DIR / 'graphs').glob('*.py'))
    provenance = {
        'metric': 'squared Euclidean distance on raw JEPA embeddings; float64 exhaustive search',
        'ks': KS, 'reference_count': len(refs), 'candidate_count': len(candidates),
        'dimension': refs.shape[1], 'self_edges': False, 'tie_rule': 'smaller reference row ID first',
        'graph': 'union of directed k-NN edges and their reverses; maximum edge weight',
        'edge_weight': 'exp(-squared_distance/(2*sigma^2)); sigma = median kth-neighbour distance',
        'batch_size': args.batch_size, 'threads': args.threads,
        'elapsed_seconds': time.perf_counter() - started,
        'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
        'scipy': scipy.__version__, 'torch': torch.__version__,
        'sha256': {str(p.resolve()): digest(p) for p in paths},
        'missing_eval_folders': sorted(set(range(400)) - set(cand_meta.eval_folder)),
        'limit': 'No learned model or independent candidate stability evaluation is performed.',
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Graph and retrieval files saved to {out}', flush=True)


if __name__ == '__main__':
    main()
