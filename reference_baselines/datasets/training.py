"""Frozen features and disjoint local reference graphs; target values are never node features."""

from pathlib import Path

from scipy import sparse
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

from reference_baselines.paths import ROOT, RESULTS_DIR


OUT = RESULTS_DIR


class ReferenceDataset(Dataset):
    def __init__(self, split, pool='union', k=10):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(split)
        self.rows = np.load(OUT / f'training_{split}_rows.npy')
        self.query = np.load(ROOT / 'embeddings/data/candidate_embeddings.npy', mmap_mode='r')
        self.reference = np.load(ROOT / 'embeddings/data/embeddings.npy', mmap_mode='r')
        self.qcomp = np.load(OUT / 'candidate_composition.npy', mmap_mode='r')
        self.rcomp = np.load(OUT / 'reference_composition.npy', mmap_mode='r')
        self.labels = np.load(OUT / 'training_targets.npy', mmap_mode='r')
        self.knn = np.load(OUT / 'candidate_knn_indices.npy', mmap_mode='r')
        self.ptr = np.load(OUT / 'chemical_reference_indptr.npy', mmap_mode='r')
        self.ids = np.load(OUT / 'chemical_reference_indices.npy', mmap_mode='r')
        self.graph = sparse.load_npz(OUT / f'reference_graph_k{k}.npz').tocsr()
        self.pool, self.k = pool, k

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = int(self.rows[index])
        chemical = self.ids[self.ptr[row]:self.ptr[row + 1]]
        nearest = self.knn[row, :self.k]
        if self.pool == 'union':
            refs = np.union1d(chemical, nearest)
        elif self.pool == 'chemical':
            refs = np.sort(chemical)
        elif self.pool == 'knn':
            refs = np.sort(nearest)
        elif self.pool == 'intersection_fallback':
            refs = np.intersect1d(chemical, nearest)
            if not len(refs):
                refs = np.sort(chemical)
        else:
            raise ValueError(self.pool)
        # Candidate-only runs use the same loader and split as all reference models.
        x = np.concatenate([self.reference[refs], self.rcomp[refs]], axis=1).astype(np.float32)
        query = np.concatenate([self.query[row], self.qcomp[row]]).astype(np.float32)
        distance = np.square(np.asarray(self.reference[refs], dtype=np.float64) - self.query[row]).sum(1)
        eligible = np.isin(refs, chemical)
        pair = np.column_stack([np.log1p(distance), eligible]).astype(np.float32)
        local = self.graph[refs][:, refs].tocoo()
        # Every local node has a self loop. Other edges are the induced fixed reference graph.
        source = np.concatenate([local.col, np.arange(len(refs))])
        target = np.concatenate([local.row, np.arange(len(refs))])
        weight = np.concatenate([local.data, np.ones(len(refs))]).astype(np.float32)
        return {'row': row, 'reference_ids': refs, 'x': torch.from_numpy(x), 'query': torch.from_numpy(query),
                'pair': torch.from_numpy(pair), 'edge': torch.tensor(np.stack([source, target]), dtype=torch.long),
                'weight': torch.from_numpy(weight), 'target': float(self.labels[row])}


def collate(items):
    offset = 0
    edges, batch = [], []
    for i, item in enumerate(items):
        edges.append(item['edge'] + offset)
        batch.append(torch.full((len(item['x']),), i, dtype=torch.long))
        offset += len(item['x'])
    return {'rows': torch.tensor([x['row'] for x in items]), 'x': torch.cat([x['x'] for x in items]),
            'query': torch.stack([x['query'] for x in items]), 'pair': torch.cat([x['pair'] for x in items]),
            'edge': torch.cat(edges, dim=1), 'weight': torch.cat([x['weight'] for x in items]),
            'batch': torch.cat(batch), 'target': torch.tensor([x['target'] for x in items], dtype=torch.float32)}


def prepare():
    from pymatgen.core import Composition
    import json
    from reference_baselines.core.io import sha256
    labels = pd.read_csv(OUT / 'candidate_labels.csv')
    manifest = pd.read_csv(OUT / 'split_manifest.csv')
    np.testing.assert_array_equal(labels.candidate_row, manifest.candidate_row)
    for field in ['chemical_system', 'eval_folder', 'idx', 'vun']:
        np.testing.assert_array_equal(labels[field], manifest[field])
    assert manifest.groupby('chemical_system').split.nunique().max() == 1
    valid = manifest.vun & labels.hull_status.eq('ok')
    assert labels.loc[valid, 'near_stable_proxy'].isin([True, False]).all()
    target = np.full(len(labels), np.nan, dtype=np.float32)
    target[valid] = labels.loc[valid, 'near_stable_proxy'].astype(float)
    np.save(OUT / 'training_targets.npy', target)
    reference = pd.read_csv(ROOT / 'embeddings/data/metadata.csv', keep_default_na=False)
    comp = np.zeros((len(reference), 100), dtype=np.float32)
    for i, formula in enumerate(reference.pretty_formula):
        c = Composition(formula)
        for element in c:
            comp[i, element.Z] = c.get_atomic_fraction(element)
    np.testing.assert_allclose(comp.sum(1), 1, atol=1e-6)
    np.save(OUT / 'reference_composition.npy', comp)
    counts, systems, class_counts = {}, {}, {}
    for split in ['train', 'validation', 'test']:
        ids = np.flatnonzero(valid & manifest.split.eq(split))
        assert len(ids)
        np.save(OUT / f'training_{split}_rows.npy', ids)
        counts[split] = len(ids)
        systems[split] = set(manifest.loc[ids, 'chemical_system'])
        if split != 'test':
            class_counts[split] = dict(positive=int(target[ids].sum()), negative=int((1-target[ids]).sum()))
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not (systems[a] & systems[b])
    assert class_counts['train']['positive'] > 0 and class_counts['train']['negative'] > 0
    assert np.isfinite(np.load(OUT / 'candidate_composition.npy')).all()
    report = {'counts': counts, 'train_validation_class_counts': class_counts,
              'excluded_vun_rows_without_label': int((manifest.vun & ~valid).sum()),
              'split_overlap_chemical_systems': 0, 'test_metrics_computed': False,
              'features': '512 frozen JEPA dimensions plus 100 atomic fractions; no energies or target labels',
              'default_pool': 'all chemical references union top 10 JEPA references; duplicate IDs removed',
              'graph': 'weighted k=10 reference graph induced separately inside each query pool, with self loops',
              'target': 'MatterSim-corrected energy <= fixed MP2023 hull + 0.1 eV/atom',
              'scope': 'held-out generated chemical systems for supervised labels; known reference bank remains available to every split',
              'caveats': ['Not independent DFT ground truth.', 'VUN flags are release batch-local.',
                          'Encoder pretraining chemistry is not held out by this downstream split.'],
              'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [OUT / 'candidate_labels.csv',
                         OUT / 'split_manifest.csv', OUT / 'reference_graph_k10.npz', Path(__file__).resolve()]}}
    (OUT / 'training_data_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    prepare()
