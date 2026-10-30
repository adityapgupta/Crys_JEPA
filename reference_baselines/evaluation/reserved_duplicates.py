"""Structure-duplicate pairs for the reserved partition only.

The duplicate rule is the same pymatgen StructureMatcher() default used for
the earlier validation split. Only reserved relaxations are read. No label
or energy field is used.

Run: python -m reference_baselines.evaluation.reserved_duplicates
"""

from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.datasets.reference_graph import OUT as SPLIT_DIR
from reference_baselines.paths import DATA_DIR, RESULTS_DIR


OUT = RESULTS_DIR / "reference_gnn"


def load_folder(task):
    folder, records = task
    path = (
        DATA_DIR
        / f"release/exp_logs/base/mp_20/eval/{folder}/relaxed_mattersim.json"
    )
    table = pd.read_json(path)
    return [
        (int(candidate_row), Structure.from_dict(table.loc[row.idx, "structure"]))
        for candidate_row, row in records.iterrows()
    ]


def compare_group(task):
    formula, records = task
    matcher = StructureMatcher()
    pairs = []
    for i, (row, structure) in enumerate(records):
        for other_row, other_structure in records[:i]:
            if matcher.fit(structure, other_structure):
                pairs.append((row, other_row))
    return pairs


def build(workers=8):
    """Return the reserved duplicate-pair table and its audit record."""
    reserved = np.load(SPLIT_DIR / "reserved_rows.npy")
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    chosen = manifest.loc[reserved]
    structures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = list(chosen.groupby("eval_folder"))
        for result in pool.map(load_folder, tasks, chunksize=4):
            structures.extend(result)
    assert {row for row, _ in structures} == set(int(r) for r in reserved)
    groups = {}
    for row, structure in structures:
        groups.setdefault(structure.composition.reduced_formula, []).append((row, structure))
    tasks = [
        (formula, sorted(records, key=lambda item: item[0]))
        for formula, records in groups.items()
        if len(records) > 1
    ]
    pairs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(compare_group, tasks, chunksize=8):
            pairs.extend(result)
    frame = pd.DataFrame(pairs, columns=["candidate_row_a", "candidate_row_b"])
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / "reserved_duplicate_pairs.csv"
    frame.to_csv(table, index=False)
    record = {
        "status": "passed",
        "reserved_rows": int(len(reserved)),
        "formula_groups": len(groups),
        "structure_pairs_checked": int(sum(len(g) * (len(g) - 1) // 2 for g in groups.values())),
        "matching_pairs": int(len(frame)),
        "matcher": "pymatgen StructureMatcher() defaults",
        "labels_read": False,
        "table_sha256": sha256(table),
    }
    (OUT / "reserved_duplicate_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    return table, record


def adjacency(path=None):
    path = path or (OUT / "reserved_duplicate_pairs.csv")
    frame = read_csv(path)
    result = {}
    for a, b in frame.itertuples(index=False, name=None):
        result.setdefault(int(a), set()).add(int(b))
        result.setdefault(int(b), set()).add(int(a))
    return result


if __name__ == "__main__":
    print(json.dumps(build()[1], indent=2))
