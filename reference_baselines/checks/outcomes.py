"""Independent checks of hull arithmetic, decomposition balance, and reference scores."""

from itertools import combinations
from pathlib import Path
import json
import pickle

from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.optimize import linprog
import lmdb
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, PACKAGE_DIR, DATA_DIR, RESULTS_DIR


OUT = RESULTS_DIR


def main():
    labels = pd.read_csv(OUT / 'candidate_labels.csv', keep_default_na=False)
    scores = pd.read_csv(OUT / 'candidate_reference_scores.csv')
    hull = pd.read_csv(OUT / 'fixed_hull_entries.csv', keep_default_na=False)
    assert 'NaN' in set(hull.formula), 'Sodium nitride formula was lost during CSV parsing'
    entries = hull.to_dict('records')
    groups = {s: g.to_dict('records') for s, g in hull.groupby('chemical_system')}
    by_id = {r['entry_id']: r for r in entries}
    valid = labels[(scores.split == 'validation') & labels.hull_status.eq('ok')]
    rng = np.random.default_rng(2026)
    sample = pd.concat([valid.sample(48, random_state=2026), valid[valid.hull_method.eq('linear_program')].head(8)])
    max_error = 0.
    for record in sample.itertuples():
        elements = record.chemical_system.split('-')
        refs = [r for n in range(1, len(elements)+1) for sub in combinations(elements, n)
                for r in groups.get('-'.join(sub), [])]
        comp = Composition(record.formula)
        matrix = np.array([[Composition(r['formula']).get_atomic_fraction(e) for r in refs] for e in elements])
        target = np.array([comp.get_atomic_fraction(e) for e in elements])
        result = linprog([r['energy_per_atom'] for r in refs], A_eq=matrix, b_eq=target,
                         bounds=(0, None), method='highs')
        assert result.success
        np.testing.assert_allclose(result.fun, record.hull_energy_per_atom, atol=1e-7, rtol=1e-9)
        max_error = max(max_error, abs(result.fun-record.hull_energy_per_atom))
        products = json.loads(record.decomposition)
        balance = np.array([sum(p['fraction'] * Composition(p['formula']).get_atomic_fraction(e) for p in products) for e in elements])
        np.testing.assert_allclose(balance, target, atol=1e-7)
        decomposition_energy = sum(p['fraction']*by_id[p['entry_id']]['energy_per_atom'] for p in products)
        np.testing.assert_allclose(decomposition_energy, record.hull_energy_per_atom, atol=1e-7)
    # The stored correction must equal pymatgen's interpretation, including energy_adjustments.
    env = lmdb.open(str(DATA_DIR / 'hull/mp.lmdb'), subdir=False, readonly=True, lock=False)
    with env.begin() as transaction:
        for record in hull.sample(128, random_state=2026).itertuples():
            key = record.chemical_system + '.' + record.formula
            n = pickle.loads(transaction.get((key+'.length').encode()))
            energies = []
            for i in range(n):
                data = pickle.loads(transaction.get((key+'.'+str(i)).encode()))
                entry = ComputedEntry.from_dict(data)
                energies.append(entry.energy_per_atom)
            np.testing.assert_allclose(min(energies), record.energy_per_atom, atol=1e-8)
    env.close()
    # Independent explicit set operations and direct squared differences, not the sum formula.
    q = np.load(ROOT / 'embeddings/data/candidate_embeddings.npy', mmap_mode='r')
    r = np.load(ROOT / 'embeddings/data/embeddings.npy', mmap_mode='r')
    ptr = np.load(OUT / 'chemical_reference_indptr.npy')
    ids = np.load(OUT / 'chemical_reference_indices.npy')
    knn = np.load(OUT / 'candidate_knn_indices.npy')
    for row in rng.choice(len(scores), 64, replace=False):
        c = set(ids[ptr[row]:ptr[row+1]])
        for k in [5, 10, 20, 50]:
            n = set(knn[row, :k])
            for prefix, pool in [('union', c|n), ('intersection', c&n), ('knn', n)]:
                actual = scores.loc[row, f'{prefix}_k{k}']
                if not pool:
                    assert np.isnan(actual)
                else:
                    expected = np.square(np.asarray(r[sorted(pool)], dtype=np.float64)-q[row]).sum(1).mean()
                    np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=2e-6)
    report = {'status': 'passed', 'independent_hull_lp_queries': len(sample), 'max_hull_energy_error': max_error,
              'decomposition_balance_checks': len(sample), 'pymatgen_correction_checks': 128,
              'reference_set_scalar_checks': 64*4*3, 'sodium_nitride_csv_check': 'passed',
              'sha256': {p.name: sha256(p) for p in [Path(__file__).resolve(),
                         PACKAGE_DIR / 'datasets/labels.py', OUT/'candidate_labels.csv']}}
    (OUT / 'outcome_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
