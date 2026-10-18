"""Shared data loading and exact Euclidean reference retrieval."""

import numpy as np
import pandas as pd
import torch


KS = (5, 10, 20, 50)


def load_data(data_dir):
    refs = np.load(data_dir / 'embeddings.npy', mmap_mode='r')
    candidates = np.load(data_dir / 'candidate_embeddings.npy', mmap_mode='r')
    ref_meta = pd.read_csv(data_dir / 'metadata.csv')
    cand_meta = pd.read_csv(data_dir / 'candidate_metadata.csv')
    if refs.ndim != 2 or candidates.ndim != 2 or refs.shape[1] != candidates.shape[1]:
        raise ValueError('Embedding matrices must have the same feature count.')
    if len(refs) != len(ref_meta) or len(candidates) != len(cand_meta):
        raise ValueError('Embedding and metadata row counts do not match.')
    if not np.isfinite(refs).all() or not np.isfinite(candidates).all():
        raise ValueError('Non-finite embedding values.')
    if not ref_meta.material_id.is_unique or cand_meta.duplicated(['eval_folder', 'idx']).any():
        raise ValueError('Duplicate crystal IDs.')
    if not cand_meta.vun.isin([True, False]).all():
        raise ValueError('Candidate vun flags must be Boolean.')
    for meta in (ref_meta, cand_meta):
        if meta.chemical_system.isna().any():
            raise ValueError('Missing chemical systems.')
        if not all(s == '-'.join(sorted(set(s.split('-')))) for s in meta.chemical_system.unique()):
            raise ValueError('Chemical systems must have sorted, unique element symbols.')
    if not np.isfinite(ref_meta.formation_energy_per_atom).all():
        raise ValueError('Missing reference formation energies.')
    return refs, candidates, ref_meta, cand_meta


def exact_knn(queries, references, k, batch_size=512, exclude_self=False, label='queries'):
    """Exhaustive search in float64; ties go to the smaller reference row ID.

    Distances are squared Euclidean distances on raw, unnormalized embeddings.
    Only one query batch by reference matrix is held in memory at a time.
    """
    if k >= len(references):
        raise ValueError('k must be smaller than the number of reference crystals.')
    if exclude_self and len(queries) != len(references):
        raise ValueError('Self-exclusion requires matching query and reference rows.')
    ref = torch.as_tensor(np.array(references, dtype=np.float64))
    norms = (ref * ref).sum(1)
    indices = np.empty((len(queries), k), dtype=np.int32)
    distances = np.empty((len(queries), k), dtype=np.float64)
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        q = torch.as_tensor(np.array(queries[start:stop], dtype=np.float64))
        d = ((q * q).sum(1)[:, None] + norms[None, :] - 2 * q @ ref.T).clamp_min_(0)
        if exclude_self:
            d[torch.arange(stop - start), torch.arange(start, stop)] = torch.inf
        values, ids = d.topk(k + 1, largest=False, sorted=True)
        values, ids = values.numpy(), ids.numpy()
        for row in range(stop - start):
            # Include all boundary ties before applying the row-ID tie rule.
            if values[row, k - 1] == values[row, k]:
                tied_ids = torch.where(d[row] <= values[row, k - 1])[0].numpy()
                tied_values = d[row, tied_ids].numpy()
                order = np.lexsort((tied_ids, tied_values))[:k]
                indices[start + row] = tied_ids[order]
                distances[start + row] = tied_values[order]
            else:
                order = np.lexsort((ids[row, :k], values[row, :k]))
                indices[start + row] = ids[row, :k][order]
                distances[start + row] = values[row, :k][order]
        if start == 0 or stop == len(queries) or (start // batch_size) % 25 == 0:
            print(f'{label}: {stop:,}/{len(queries):,}', flush=True)
    return indices, distances
