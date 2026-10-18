"""Check duplicate structures across batches within each method's validation top 1%."""

from concurrent.futures import ProcessPoolExecutor
import json

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
import pandas as pd

from reference_baselines.paths import DATA_DIR, RESULTS_DIR


OUT = RESULTS_DIR


def load_folder(task):
    folder, records = task
    path = DATA_DIR / f'release/exp_logs/base/mp_20/eval/{folder}/relaxed_mattersim.json'
    table = pd.read_json(path)
    return [(int(row.candidate_row), Structure.from_dict(table.loc[row.idx, 'structure'])) for row in records.itertuples()]


def compare_group(task):
    formula, records = task
    matcher = StructureMatcher()  # Same defaults as the repository's VUN evaluator.
    pairs = []
    for i, (row, structure) in enumerate(records):
        for other_row, other_structure in records[:i]:
            if matcher.fit(structure, other_structure):
                pairs.append((row, other_row))
    return pairs


def main():
    selection = pd.read_csv(OUT / 'validation_top1_selections.csv')
    metadata = pd.read_csv(OUT / 'split_manifest.csv')
    chosen = metadata[metadata.candidate_row.isin(selection.candidate_row)]
    assert chosen.split.eq('validation').all()
    structures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(load_folder, list(chosen.groupby('eval_folder'))), 1):
            structures.extend(result)
            if i % 50 == 0:
                print('Selected-structure folders read:', i, flush=True)
    groups = {}
    for row, structure in structures:
        groups.setdefault(structure.composition.reduced_formula, []).append((row, structure))
    tasks = [(formula, sorted(records, key=lambda x: x[0])) for formula, records in groups.items() if len(records)>1]
    pairs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(compare_group, tasks, chunksize=4):
            pairs.extend(result)
    adjacency = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b); adjacency.setdefault(b, set()).add(a)
    records = []
    labels = pd.read_csv(OUT / 'candidate_labels.csv').set_index('candidate_row')
    for method, group in selection.groupby('method', sort=False):
        # Retain the original score order. This follows the repo's greedy unique filter.
        kept, rejected = [], []
        for row in group.candidate_row:
            if adjacency.get(row, set()).intersection(kept):
                rejected.append(row)
            else:
                kept.append(row)
        records.append({'method': method, 'selected_count': len(group), 'unique_count': len(kept),
                        'duplicate_count': len(rejected),
                        'unique_near_stable_count': int(labels.loc[kept, 'near_stable_proxy'].sum()),
                        'unique_near_stable_yield_per_slot': int(labels.loc[kept, 'near_stable_proxy'].sum())/len(group)})
    pd.DataFrame(records).to_csv(OUT / 'validation_top1_duplicate_audit.csv', index=False)
    (OUT / 'validation_duplicate_audit.json').write_text(json.dumps({
        'selected_union_count': len(chosen), 'matched_pairs': len(pairs),
        'policy': 'per-method greedy StructureMatcher() across all selected validation batches; no refill',
        'scope': 'top 1% only; novelty against MP2023 and synthesis success are not established',
        'test_structures_examined': 0}, indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == '__main__':
    main()
