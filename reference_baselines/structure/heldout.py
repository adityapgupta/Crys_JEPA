"""Independent cross-batch duplicate and fixed MP2023 novelty checks."""

from concurrent.futures import ProcessPoolExecutor
import pickle

import lmdb
import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.evaluation.selected_duplicates import (
    load_folder,
    compare_group,
)
from reference_baselines.paths import DATA_DIR


DATABASE = None


def init_novelty():
    global DATABASE
    DATABASE = lmdb.open(
        str(DATA_DIR / "hull/mp.lmdb"), subdir=False, readonly=True, lock=False
    )


def novelty_group(task):
    """Compare only identical reduced compositions, using all MP2023 polymorphs."""
    formula, records = task
    system = records[0][1].composition.chemical_system
    prefix = system + "." + formula
    references = []
    with DATABASE.begin() as txn:
        length = txn.get((prefix + ".length").encode())
        for index in range(pickle.loads(length) if length is not None else 0):
            entry = pickle.loads(txn.get((prefix + "." + str(index)).encode()))
            structure = Structure.from_dict(entry["structure"])
            structure.remove_oxidation_states()
            references.append((str(entry["entry_id"]), structure))
    matcher = StructureMatcher()
    result = []
    for row, structure in records:
        query = structure.copy()
        query.remove_oxidation_states()
        match = next(
            (
                identifier
                for identifier, known in references
                if matcher.fit(query, known)
            ),
            None,
        )
        result.append(
            {
                "candidate_row": row,
                "novel_against_mp2023": match is None,
                "matched_entry_id": match,
                "same_formula_references": len(references),
            }
        )
    return result


def selections(rows, names, matrix, adjacency, budget):
    result = {}
    for j, name in enumerate(names):
        available = np.isfinite(matrix[:, j])
        ranked = rows[available][np.lexsort((rows[available], matrix[available, j]))]
        result[name] = unique_order(ranked, adjacency)[:budget]
        if len(result[name]) != budget:
            raise ValueError(f"Cannot fill the fixed unique budget: {name}")
    return result


def audit(manifest, rows, names, matrix, budget, out, workers=4):
    chosen = manifest.loc[rows]
    assert chosen.split.eq("test").all() and chosen.vun.all()
    structures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, records in enumerate(
            pool.map(load_folder, list(chosen.groupby("eval_folder"))), 1
        ):
            structures.extend(records)
            if i % 50 == 0:
                print("Test folders read:", i, flush=True)
    if set(row for row, _ in structures) != set(rows) or len(structures) != len(rows):
        raise ValueError("Test structures and row IDs differ")
    groups = {}
    for row, structure in structures:
        groups.setdefault(structure.composition.reduced_formula, []).append(
            (row, structure)
        )
    tasks = [
        (formula, sorted(records))
        for formula, records in groups.items()
        if len(records) > 1
    ]
    pairs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, matches in enumerate(pool.map(compare_group, tasks, chunksize=8), 1):
            pairs.extend(matches)
            if i % 1000 == 0:
                print("Test formula groups checked:", i, "/", len(tasks), flush=True)
    pd.DataFrame(pairs, columns=["candidate_row_a", "candidate_row_b"]).to_csv(
        out / "duplicate_pairs.csv", index=False
    )
    adjacency = {}
    for a, b in pairs:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    selected = selections(rows, names, matrix, adjacency, budget)
    pd.DataFrame(
        [
            {"method": name, "rank": i + 1, "candidate_row": int(row)}
            for name, values in selected.items()
            for i, row in enumerate(values)
        ]
    ).to_csv(out / "selections.csv", index=False)
    selected_union = set(np.concatenate(list(selected.values())))
    tasks = [
        (formula, [(row, s) for row, s in records if row in selected_union])
        for formula, records in groups.items()
    ]
    tasks = [(formula, records) for formula, records in tasks if records]
    novelty = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_novelty) as pool:
        for i, records in enumerate(pool.map(novelty_group, tasks, chunksize=8), 1):
            novelty.extend(records)
            if i % 1000 == 0:
                print("MP2023 novelty groups checked:", i, "/", len(tasks), flush=True)
    pd.DataFrame(novelty).sort_values("candidate_row").to_csv(
        out / "novelty.csv", index=False
    )
    return {
        "test_structures": len(structures),
        "formula_groups": len(groups),
        "pairs_checked": sum(len(g) * (len(g) - 1) // 2 for g in groups.values()),
        "matched_pairs": len(pairs),
        "selected_union": len(selected_union),
        "matcher": "pymatgen StructureMatcher defaults; same reduced composition",
        "novelty_bank": "all polymorphs in pinned MP2023 database; oxidation states removed",
        "candidate_energy_labels_read": False,
    }
