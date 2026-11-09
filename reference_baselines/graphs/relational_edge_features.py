"""Precompute the relational edge-feature sources.

Outputs (results/relational_edge/):
  reference_decomp_frequency.npy   (REFERENCE_COUNT,)  fit-decomposition counts c_i
  cooccurrence_normaliser.npy      scalar N  (fit candidates with a mapped decomposition)
  hull_compatibility.npz           sparse 0/1; 1 where two references share a hull tie-line

Fit-side only: candidate outcomes and reserved labels are never read.

Hull energies are read from results/reference_hull_energies.csv with columns
``reference_row, formula, energy_per_atom``.  Point this at the project's MP2023
reference energies (the same ones behind the evaluation hull).
Compatible = the midpoint of references i, j decomposes into exactly {i, j} on the
convex hull of their union chemical system (a genuine two-phase tie-line).  This is
equivalent to "both stable and sharing a facet", since every edge of a hull simplex
is a tie-line; immiscible phases (no stabler intermediate) are correctly compatible.

Run: python -m reference_baselines.graphs.relational_edge_features
"""

import argparse
import json

import numpy as np
from scipy import sparse

from reference_baselines.core.io import sha256
from reference_baselines.datasets.decomposition import build_targets
from reference_baselines.datasets.reference_graph import OUT as SPLIT_DIR, REFERENCE_COUNT, load_graph
from reference_baselines.paths import RESULTS_DIR


OUT = RESULTS_DIR / "relational_edge"
ENERGY_CSV = RESULTS_DIR / "reference_hull_energies.csv"


def decomposition_frequency(fit_rows):
    """Per-reference count of fit decompositions containing it, and the document count N."""
    positives = build_targets(fit_rows)[0]
    frequency = np.zeros(REFERENCE_COUNT, dtype=np.float64)
    documents = 0
    for row in fit_rows:
        phases = positives[int(row)]
        if len(phases):
            documents += 1
            frequency[np.asarray(phases, dtype=np.int64)] += 1.0
    return frequency, float(documents)


def hull_compatibility(energy_csv):
    """Sparse 0/1 matrix.  For each reference pair that is an edge of the physical
    graph, 1 if the two references are tie-line adjacent on the convex hull of
    *their own* combined chemical system ``elements(i) | elements(j)``, else 0.

    Only physical-graph edges are evaluated -- those are the only pairs the dataset
    ever looks up -- and each pair's hull is built from the exact union system, so a
    pair is never judged inside an unrelated larger system.

    Two references are compatible iff the midpoint of their compositions
    decomposes into exactly those two phases -- a genuine two-phase tie-line, not
    mere co-membership of a higher-order facet (which in ternary+ systems would
    pair chemically unrelated phases).
    """
    import pandas as pd
    from pymatgen.core import Composition
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

    if not energy_csv.exists():
        raise FileNotFoundError(
            f"Missing {energy_csv} (columns: reference_row, formula, energy_per_atom), "
            "aligned to the 27,136 references. Produce it from the project's MP2023 "
            "reference energies."
        )
    print("[hull] two-phase tie-line test v2", flush=True)
    table = pd.read_csv(energy_csv)
    composition = {int(r.reference_row): Composition(str(r.formula)) for r in table.itertuples()}
    energy = {int(r.reference_row): float(r.energy_per_atom) for r in table.itertuples()}
    elements = {row: frozenset(comp.elements) for row, comp in composition.items()}

    physical = load_graph("physical").tocoo()
    edges = {
        (int(i), int(j))
        for i, j in zip(physical.row, physical.col)
        if int(i) < int(j) and int(i) in composition and int(j) in composition
    }
    # Group edges by their exact union chemical system; build one hull per system.
    groups = {}
    for i, j in edges:
        groups.setdefault(elements[i] | elements[j], []).append((i, j))

    source, destination = [], []
    for system, system_edges in groups.items():
        members = [row for row, els in elements.items() if els <= system]
        if len(members) < 2:
            continue
        entries = [PDEntry(composition[r], energy[r] * composition[r].num_atoms, attribute=r) for r in members]
        try:
            diagram = PhaseDiagram(entries)
        except Exception:
            continue
        stable = {e.attribute for e in diagram.stable_entries}
        for i, j in system_edges:  # i < j
            if i not in stable or j not in stable:
                continue
            midpoint = composition[i].fractional_composition + composition[j].fractional_composition
            try:
                decomposition = diagram.get_decomposition(midpoint)
            except Exception:
                continue
            rows = {entry.attribute for entry, amount in decomposition.items() if amount > 1e-6}
            if rows == {i, j}:  # the i-j join is a genuine two-phase tie-line
                source += [i, j]
                destination += [j, i]
    hull = sparse.csr_matrix(
        (np.ones(len(source), dtype=np.float32), (source, destination)),
        shape=(REFERENCE_COUNT, REFERENCE_COUNT),
    )
    hull.sum_duplicates()
    hull.data[:] = 1.0
    return hull.tocsr()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-csv", default=str(ENERGY_CSV))
    parser.add_argument("--skip-hull", action="store_true", help="Only build frequency/normaliser")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fit_rows = np.load(SPLIT_DIR / "train_rows.npy").astype(np.int64)
    frequency, documents = decomposition_frequency(fit_rows)
    np.save(OUT / "reference_decomp_frequency.npy", frequency)
    np.save(OUT / "cooccurrence_normaliser.npy", np.asarray(documents, dtype=np.float64))

    report = {
        "status": "frequency_done",
        "fit_rows": int(len(fit_rows)),
        "documents_with_decomposition": documents,
        "references_with_positive_frequency": int((frequency > 0).sum()),
        "candidate_labels_read": False,
    }
    if not args.skip_hull:
        from pathlib import Path

        hull = hull_compatibility(Path(args.energy_csv))
        sparse.save_npz(OUT / "hull_compatibility.npz", hull)
        report["status"] = "complete"
        report["code_version"] = "tieline-v2"
        report["hull_compatible_pairs"] = int(hull.nnz)
        report["hull_sha256"] = sha256(OUT / "hull_compatibility.npz")
    (OUT / "relational_edge_features.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
