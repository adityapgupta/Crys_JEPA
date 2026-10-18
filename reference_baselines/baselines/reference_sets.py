"""Create fixed reference-pool baselines and a chemical-system split manifest."""

from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.core.retrieval import KS
from reference_baselines.paths import ROOT, RESULTS_DIR


def split_system(system):
    fraction = int.from_bytes(hashlib.sha256(('reference-v1|' + system).encode()).digest()[:8], 'big') / 2**64
    return 'train' if fraction < .70 else 'validation' if fraction < .85 else 'test'


def main():
    out = RESULTS_DIR
    rm = pd.read_csv(ROOT / 'embeddings/data/metadata.csv')
    cm = pd.read_csv(ROOT / 'embeddings/data/candidate_metadata.csv')
    ref = np.load(ROOT / 'embeddings/data/embeddings.npy', mmap_mode='r')
    candidate = np.load(ROOT / 'embeddings/data/candidate_embeddings.npy', mmap_mode='r')
    ni = np.load(out / 'candidate_knn_indices.npy')
    nd = np.load(out / 'candidate_knn_squared_distances.npy')
    groups = {system: np.asarray(ids, dtype=np.int32) for system, ids in rm.groupby('chemical_system').groups.items()}
    pools = {}
    for system in cm.chemical_system.unique():
        elements = system.split('-')
        arrays = [groups['-'.join(sub)] for n in range(1, len(elements) + 1)
                  for sub in combinations(elements, n) if '-'.join(sub) in groups]
        pools[system] = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int32)
    sizes = np.array([len(pools[s]) for s in cm.chemical_system], dtype=np.int32)
    indptr = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    indices = np.concatenate([pools[s] for s in cm.chemical_system])
    np.save(out / 'chemical_reference_indptr.npy', indptr)
    np.save(out / 'chemical_reference_indices.npy', indices)
    scores = {'crys_jepa': np.full(len(cm), np.nan)}
    for turn in range(40):
        mask = cm.eval_folder.between(turn * 10, turn * 10 + 9).to_numpy()
        values = np.load(ROOT / 'screen_out' / f'dis_{turn}.npy')
        assert values.shape == (int(mask.sum()),)
        scores['crys_jepa'][mask] = np.where(values < 1e9, values, np.nan)
    for k in KS:
        for method in ['knn', 'intersection', 'intersection_fallback', 'union', 'chemical_knn']:
            scores[f'{method}_k{k}'] = np.full(len(cm), np.nan)
    for temperature in [1., 10., 50.]:
        scores[f'chemical_softmax_t{int(temperature)}'] = np.full(len(cm), np.nan)
    reference_ids_by_system = {s: set(ids) for s, ids in pools.items()}
    max_error = 0.
    chemical_neighbours = np.full((len(cm), max(KS)), -1, dtype=np.int32)
    for count, (system, rows) in enumerate(cm.groupby('chemical_system').groups.items(), 1):
        rows = np.asarray(rows)
        ids = pools[system]
        allowed = reference_ids_by_system[system]
        overlap = np.array([[x in allowed for x in neighbours] for neighbours in ni[rows]], dtype=bool)
        # Recompute chemical distances in float64. The legacy mean is preserved as the baseline score.
        local = []
        if len(ids):
            for start in range(0, len(rows), 256):
                selected = rows[start:start + 256]
                q = np.asarray(candidate[selected], dtype=np.float64)
                r = np.asarray(ref[ids], dtype=np.float64)
                distance = np.maximum((q * q).sum(1)[:, None] + (r * r).sum(1)[None, :] - 2 * q @ r.T, 0)
                ordering = np.argsort(distance, axis=1, kind='stable')
                order = np.take_along_axis(distance, ordering, axis=1)
                chemical_neighbours[selected, :min(max(KS), len(ids))] = ids[ordering[:, :max(KS)]]
                mean = distance.mean(1)
                np.testing.assert_allclose(mean, scores['crys_jepa'][selected], rtol=2e-6, atol=3e-5)
                max_error = max(max_error, float(np.max(np.abs(mean - scores['crys_jepa'][selected]))))
                for k in KS:
                    scores[f'chemical_knn_k{k}'][selected] = order[:, :min(k, len(ids))].mean(1)
                for temperature in [1., 10., 50.]:
                    weights = np.exp(-(distance - distance.min(1)[:, None]) / temperature)
                    scores[f'chemical_softmax_t{int(temperature)}'][selected] = (distance * weights).sum(1) / weights.sum(1)
        for k in KS:
            distances = nd[rows, :k]
            mask = overlap[:, :k]
            n_overlap = mask.sum(1)
            overlap_sum = np.where(mask, distances, 0).sum(1)
            intersection = np.divide(overlap_sum, n_overlap, out=np.full(len(rows), np.nan), where=n_overlap > 0)
            scores[f'knn_k{k}'][rows] = distances.mean(1)
            scores[f'intersection_k{k}'][rows] = intersection
            scores[f'intersection_fallback_k{k}'][rows] = np.where(n_overlap > 0, intersection, scores['crys_jepa'][rows])
            total = np.nan_to_num(scores['crys_jepa'][rows]) * len(ids) + distances.sum(1) - overlap_sum
            scores[f'union_k{k}'][rows] = total / (len(ids) + k - n_overlap)
        if count % 10000 == 0:
            print(f'Reference pools scored: {count:,}/{cm.chemical_system.nunique():,}', flush=True)
    # A random reference set is a reference-selection control, separate from random candidate selection.
    rng = np.random.default_rng(2026)
    random_ids = np.array([rng.choice(len(rm), 10, replace=False) for _ in range(len(cm))], dtype=np.int32)
    random_scores = np.empty(len(cm))
    for start in range(0, len(cm), 512):
        stop = min(start + 512, len(cm))
        difference = np.asarray(ref[random_ids[start:stop]], dtype=np.float64) - np.asarray(candidate[start:stop], dtype=np.float64)[:, None, :]
        random_scores[start:stop] = np.square(difference).sum(2).mean(1)
    scores['random_references_k10'] = random_scores
    np.save(out / 'random_reference_indices.npy', random_ids)
    np.save(out / 'chemical_knn_indices.npy', chemical_neighbours)
    table = cm.rename_axis('candidate_row').reset_index()
    table['split'] = table.chemical_system.map(split_system)
    table['chemical_reference_count'] = sizes
    for name, values in scores.items():
        table[name] = values
    table.to_csv(out / 'candidate_reference_scores.csv', index=False)
    table[['candidate_row', 'eval_folder', 'idx', 'chemical_system', 'vun', 'split']].to_csv(out / 'split_manifest.csv', index=False)
    for split in ['train', 'validation', 'test']:
        ids = table.loc[(table.split == split) & table.vun, 'candidate_row'].to_numpy(dtype=np.int64)
        np.save(out / f'{split}_candidate_rows.npy', ids)
    report = {'score_methods': list(scores), 'method_count': len(scores), 'candidate_count': len(cm),
              'split_rule': 'SHA256 reference-v1|chemical_system; thresholds 70%,85%; no target labels used',
              'split_counts_vun': table[table.vun].split.value_counts().to_dict(),
              'split_counts_systems': table.groupby('split').chemical_system.nunique().to_dict(),
              'chemical_pool_total_links': len(indices), 'legacy_score_max_absolute_error': max_error,
              'test_policy': 'No baseline performance is evaluated on test rows before model selection.',
              'empty_intersection': 'NaN score means abstention; fallback is a separately named method',
              'pool_semantics': 'unique whole-crystal references; union removes duplicate row IDs',
              'sha256': {str(p.resolve()): sha256(p) for p in [ROOT / 'embeddings/data/metadata.csv',
                          ROOT / 'embeddings/data/candidate_metadata.csv', Path(__file__).resolve()]}}
    (out / 'reference_experiment_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
