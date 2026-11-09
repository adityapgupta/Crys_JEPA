"""Build results/reference_hull_energies.csv for the relational edge-feature hull.

Columns: ``reference_row, formula, energy_per_atom`` (formation energy per atom),
aligned to the 27,136 references.  Source: results/reference_rows.csv.

Formation energies are used because convex-hull tie-line adjacency is invariant to
the linear total-vs-formation shift, so the facets (and thus compatibility) are the
same as with total energies -- provided each system has its elemental references,
which pymatgen needs.  Systems missing an elemental reference are skipped by the
hull precompute (those edges get compatibility 0).

Design note: the hull compatibility is built from the hull of *the references
themselves* (per union system), not the full MP2023 fixed hull.  On the full hull,
any two references that co-occur in a decomposition are automatically facet
adjacent, so that feature would be ~1 on every physical edge -- uninformative.  The
reference-subset hull gives a non-trivial relation that varies across edges.

Run: python -m reference_baselines.graphs.build_reference_hull_energies
"""

import pandas as pd

from reference_baselines.paths import RESULTS_DIR


def main():
    source = RESULTS_DIR / "reference_rows.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}")
    table = pd.read_csv(source)
    out = table[["reference_row", "pretty_formula", "formation_energy_per_atom"]].rename(
        columns={"pretty_formula": "formula", "formation_energy_per_atom": "energy_per_atom"}
    )
    if not (out.reference_row.to_numpy() == range(len(out))).all():
        raise ValueError("reference_row is not a contiguous 0..N-1 index")
    if out.energy_per_atom.isna().any():
        raise ValueError("Missing reference energies")
    dest = RESULTS_DIR / "reference_hull_energies.csv"
    out.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(out)} rows)")


if __name__ == "__main__":
    main()
