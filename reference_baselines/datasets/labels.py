"""Prepare release-aligned MatterSim proxy labels against a fixed MP-2023 hull."""

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import argparse
import json
import pickle
import warnings

from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
import numpy as np
import pandas as pd
import torch

from eval.vsun.structure_summary import VasprunLike, IdentityCorrectionScheme
from reference_baselines.core.io import sha256
from reference_baselines.core.thermodynamics import FixedHull
from reference_baselines.paths import ROOT, DATA_DIR, RESULTS_DIR


HULL = {}


def candidate_entries(task):
    folder, path, metadata = task
    source = pd.read_json(path)
    indicators = torch.load(path.parent / 'indicators_mattersim_mp_20.pt', weights_only=False, map_location='cpu')
    if source.index.tolist() != metadata.idx.tolist():
        raise ValueError(f'Candidate row mismatch in folder {folder}')
    if not np.array_equal(indicators['vu'] * indicators['vn'], metadata.vun.to_numpy()):
        raise ValueError(f'VUN mismatch in folder {folder}')
    correction = MaterialsProject2020Compatibility()
    result = []
    composition = np.zeros((len(metadata), 100), dtype=np.float32)
    for offset, ((_, meta), (_, raw)) in enumerate(zip(metadata.iterrows(), source.iterrows())):
        record = {'candidate_row': int(meta.candidate_row), 'eval_folder': folder, 'idx': int(meta.idx),
                  'chemical_system': meta.chemical_system, 'vun': bool(meta.vun), 'status': 'ok'}
        try:
            if raw.exception or not np.isfinite(raw.e_relax):
                raise ValueError('Release relaxation exception or missing energy')
            structure = Structure.from_dict(raw.structure)
            if structure.composition.chemical_system != meta.chemical_system:
                raise ValueError('Source chemical system disagrees with embedding metadata')
            comp = structure.composition
            record.update(formula=comp.reduced_formula, num_atoms=len(structure),
                          raw_energy_eV=float(raw.e_relax), raw_energy_per_atom=float(raw.e_relax) / len(structure))
            for element, amount in comp.items():
                if element.Z >= 100:
                    raise ValueError('Element cannot fit the encoder composition representation')
                composition[offset, element.Z] = amount / len(structure)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                entry = VasprunLike(structure, float(raw.e_relax)).get_computed_entry(
                    energy_correction_scheme=IdentityCorrectionScheme())
                adjusted = correction.process_entry(entry)
            if adjusted is None:
                raise ValueError('MP2020 compatibility rejected the entry')
            record.update(corrected_energy_per_atom=float(adjusted.energy_per_atom),
                          correction_per_atom=float(adjusted.correction / len(structure)),
                          correction_warning_count=len(caught))
        except Exception as exc:
            record.update(status='entry_error', error=f'{type(exc).__name__}: {exc}')
        result.append(record)
    return result, composition, metadata.candidate_row.to_numpy()


def extract_hull(path, output):
    import lmdb
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=True)
    rows = []
    with env.begin() as transaction:
        systems = pickle.loads(transaction.get(b'chemical_systems'))
        for system in systems:
            formulas = pickle.loads(transaction.get((system + '.reduced_formulas').encode()))
            for formula in formulas:
                prefix = system + '.' + formula
                count = pickle.loads(transaction.get((prefix + '.length').encode()))
                best = None
                for index in range(count):
                    d = pickle.loads(transaction.get((prefix + '.' + str(index)).encode()))
                    n_atoms = sum(d['composition'].values())
                    e = (d['energy'] + d.get('correction', 0.)) / n_atoms
                    if np.isfinite(e) and (best is None or e < best['energy_per_atom']):
                        best = {'chemical_system': system, 'formula': formula,
                                'energy_per_atom': e, 'entry_id': str(d['entry_id'])}
                if best is not None:
                    rows.append(best)
    env.close()
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows


def init_hull(rows):
    global HULL
    for row in rows:
        HULL.setdefault(row['chemical_system'], []).append(row)


def hull_group(task):
    system, records = task
    elements = system.split('-')
    refs = [entry for size in range(1, len(elements) + 1) for sub in combinations(elements, size)
            for entry in HULL.get('-'.join(sub), [])]
    missing = [element for element in elements if element not in HULL]
    if missing:
        return [{'candidate_row': r['candidate_row'], 'hull_status': 'missing_elemental_reference',
                 'hull_error': ','.join(missing)} for r in records]
    solver = FixedHull(elements, refs)
    result = []
    cache = {}
    for record in records:
        formula = record['formula']
        try:
            if formula not in cache:
                hull_energy, products, method = solver.minimum(formula)
                cache[formula] = hull_energy, products, method
            hull_energy, products, method = cache[formula]
            delta = record['corrected_energy_per_atom'] - hull_energy
            result.append({'candidate_row': record['candidate_row'], 'hull_status': 'ok',
                           'hull_energy_per_atom': hull_energy, 'e_above_hull_proxy': delta,
                           'near_stable_proxy': delta <= .1, 'on_or_below_hull_proxy': delta <= 0.,
                           'decomposition': json.dumps(products, separators=(',', ':')), 'hull_method': method})
        except Exception as exc:
            result.append({'candidate_row': record['candidate_row'], 'hull_status': 'hull_error',
                           'hull_error': f'{type(exc).__name__}: {exc}'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['entries', 'hull'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = RESULTS_DIR
    metadata_path = ROOT / 'embeddings/data/candidate_metadata.csv'
    metadata = pd.read_csv(metadata_path).rename_axis('candidate_row').reset_index()
    source_root = DATA_DIR / 'release/exp_logs/base/mp_20/eval'
    if args.stage == 'entries':
        tasks = [(int(folder), source_root / str(folder) / 'relaxed_mattersim.json', group)
                 for folder, group in metadata.groupby('eval_folder', sort=True)]
        records = []
        compositions = np.zeros((len(metadata), 100), dtype=np.float32)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number, (rows, comp, ids) in enumerate(pool.map(candidate_entries, tasks, chunksize=1), 1):
                records.extend(rows); compositions[ids] = comp
                if number % 20 == 0 or number == len(tasks):
                    print(f'Candidate entry folders: {number}/{len(tasks)}', flush=True)
        table = pd.DataFrame(records).sort_values('candidate_row').reset_index(drop=True)
        np.testing.assert_array_equal(table.candidate_row, np.arange(len(metadata)))
        table.to_csv(out / 'candidate_entries.csv', index=False)
        np.save(out / 'candidate_composition.npy', compositions)
        hull_source = DATA_DIR / 'hull/mp.lmdb'
        rows = extract_hull(hull_source, out / 'fixed_hull_entries.csv')
        print('Entry status:', table.status.value_counts().to_dict(), 'hull entries:', len(rows), flush=True)
    else:
        # NaN is also a valid formula (sodium nitride); never parse formulas as CSV nulls.
        table = pd.read_csv(out / 'candidate_entries.csv', keep_default_na=False)
        rows = pd.read_csv(out / 'fixed_hull_entries.csv', keep_default_na=False).to_dict('records')
        tasks = [(system, group.to_dict('records')) for system, group in table[table.status == 'ok'].groupby('chemical_system')]
        results = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_hull, initargs=(rows,)) as pool:
            for number, group in enumerate(pool.map(hull_group, tasks, chunksize=16), 1):
                results.extend(group)
                if number % 2000 == 0 or number == len(tasks):
                    print(f'Candidate chemical systems labelled: {number}/{len(tasks)}', flush=True)
                    pd.DataFrame(results).to_csv(out / 'candidate_hull_checkpoint.csv', index=False)
        labels = table.merge(pd.DataFrame(results), on='candidate_row', how='left', validate='one_to_one')
        labels.hull_status = labels.hull_status.fillna('entry_error')
        labels.to_csv(out / 'candidate_labels.csv', index=False)
        report = {'label_source': 'released MatterSim energies with MP2020Compatibility corrections against pinned MP-2023 DFT hull',
                  'label_type': 'MLFF proxy, not independently computed DFT', 'candidate_count': len(labels),
                  'entry_status': labels.status.value_counts().to_dict(), 'hull_status': labels.hull_status.value_counts().to_dict(),
                  'threshold_eV_per_atom': .1, 'negative_hull_distances_retained': True,
                  'relaxation_convergence': 'not recorded in the release; cannot verify force convergence',
                  'sha256': {str(p.resolve()): sha256(p) for p in [metadata_path, out / 'candidate_entries.csv',
                              out / 'fixed_hull_entries.csv', Path(__file__).resolve()]}}
        (out / 'candidate_label_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
