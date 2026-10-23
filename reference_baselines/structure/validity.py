"""Recheck selected relaxed structures and explicit SMACT14 composition rules."""

from reference_baselines.core.audit_io import read_csv

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from importlib.metadata import version
import json
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import DATA_DIR


def geometry_checks(structure):
    finite = bool(
        np.isfinite(structure.frac_coords).all()
        and np.isfinite(structure.lattice.matrix).all()
    )
    volume = float(structure.volume)
    distances = structure.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    minimum = float(distances.min()) if len(structure) > 1 else np.inf
    # Also check images of the same site, which the original diagonal mask skips.
    periodic = structure.get_neighbor_list(4, exclude_self=True)[3]
    periodic_minimum = float(periodic.min()) if len(periodic) else 4.0
    return {
        "finite_structure": finite,
        "ordered_structure": structure.is_ordered,
        "volume_A3": volume,
        "volume_per_atom_A3": volume / len(structure),
        "density_g_cm3": float(structure.density),
        "minimum_site_distance_A": minimum if np.isfinite(minimum) else None,
        "minimum_periodic_distance_within4A": periodic_minimum,
        "legacy_geometry_valid": bool(finite and volume >= 0.1 and minimum >= 0.5),
        "periodic_geometry_valid": bool(
            finite
            and structure.is_ordered
            and volume >= 0.1
            and periodic_minimum >= 0.5
        ),
    }


@lru_cache(maxsize=None)
def composition_checks(formula):
    """Fixed integer ratios: charge-neutral iff sum(count_i * ox_i) == 0.

    Same single-oxidation-state assumption as the repository. Explicit SMACT14
    states keep this heuristic stable across SMACT versions. No mixed-valence
    or metallicity rescue is silently applied; metal-only alloys are allowed.
    """
    import smact
    from smact.screening import pauling_test
    from pymatgen.core import Composition

    comp = Composition(formula)
    symbols = sorted(comp.get_el_amt_dict())
    counts = np.rint([comp[s] for s in symbols]).astype(int)
    if len(symbols) == 1 or all(s in smact.metals for s in symbols):
        return {
            "composition_valid": True,
            "composition_reason": "single element or metal-only alloy",
            "oxidation_witness": "not required",
        }
    space = smact.element_dictionary(symbols)
    elements = [space[s] for s in symbols]
    states = [
        getattr(e, "oxidation_states_smact14", e.oxidation_states) for e in elements
    ]
    electronegativities = [e.pauling_eneg for e in elements]
    neutral_seen = False
    for oxidation in product(*states):
        if np.dot(counts, oxidation) != 0:
            continue
        neutral_seen = True
        try:
            allowed = pauling_test(oxidation, electronegativities)
        except TypeError:
            allowed = True  # Mirrors the repository's missing-electronegativity rule.
        if allowed:
            return {
                "composition_valid": True,
                "composition_reason": "charge balance and Pauling test",
                "oxidation_witness": json.dumps(dict(zip(symbols, oxidation))),
            }
    return {
        "composition_valid": False,
        "composition_reason": "no Pauling-consistent state"
        if neutral_seen
        else "no single-state charge balance",
        "oxidation_witness": "",
    }


def audit_folder(task):
    warnings.filterwarnings(
        "ignore", message="gcd is deprecated", category=FutureWarning
    )
    folder, group = task
    raw = pd.read_json(
        DATA_DIR / f"release/exp_logs/base/mp_20/eval/{folder}/relaxed_mattersim.json"
    )
    records = []
    for row in group.itertuples():
        source = raw.loc[row.idx]
        structure = Structure.from_dict(source.structure)
        if structure.composition.reduced_formula != row.formula:
            raise ValueError("Source formula and candidate label disagree")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            geometry = geometry_checks(structure)
            chemistry = composition_checks(row.formula)
        records.append(
            {
                "candidate_row": row.candidate_row,
                "formula": row.formula,
                **geometry,
                **chemistry,
                "saved_relaxation_exception": bool(source.exception),
                "saved_energy_finite": bool(np.isfinite(source.e_relax)),
                "saved_force_available": any(
                    k in raw.columns for k in ["forces", "fmax", "max_force"]
                ),
                "recorded_convergence_available": any(
                    k in raw.columns for k in ["converged", "relaxation_converged"]
                ),
                "all_validity_checks_pass": bool(
                    geometry["periodic_geometry_valid"]
                    and chemistry["composition_valid"]
                    and not source.exception
                    and np.isfinite(source.e_relax)
                ),
            }
        )
    return records


def main():
    candidates = read_csv(OUT / "candidate_audit.csv")
    selected = candidates[candidates.selection_group.ne("neither")]
    records = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, rows in enumerate(
            pool.map(audit_folder, list(selected.groupby("eval_folder"))), 1
        ):
            records.extend(rows)
            if i % 50 == 0:
                print("Validity folders checked:", i, flush=True)
    checked = pd.DataFrame(records).set_index("candidate_row")
    if not checked.index.is_unique or set(checked.index) != set(selected.candidate_row):
        raise ValueError("Validity rows differ from selected candidate union")
    checked.to_csv(OUT / "validity_checks.csv")
    merged = selected.join(checked.drop(columns="formula"), on="candidate_row")
    summaries = []
    for (split, group), frame in merged.groupby(["split", "selection_group"]):
        summaries.append(
            {
                "split": split,
                "selection_group": group,
                "count": len(frame),
                "proxy_stability_passing": int(frame.passing.sum()),
                "periodic_geometry_valid": int(frame.periodic_geometry_valid.sum()),
                "composition_valid": int(frame.composition_valid.sum()),
                "all_valid": int(frame.all_validity_checks_pass.sum()),
                "valid_and_proxy_passing": int(
                    (frame.all_validity_checks_pass & frame.passing).sum()
                ),
            }
        )
    report = {
        "rules": "volume >= 0.1 A^3; periodic distances >=0.5 A; finite ordered structure; SMACT14 single-state charge balance/Pauling; metal-only alloys allowed",
        "smact_version": version("smact"),
        "summaries": summaries,
        "force_records_available": int(checked.saved_force_available.sum()),
        "convergence_records_available": int(
            checked.recorded_convergence_available.sum()
        ),
        "limitations": [
            "These are plausibility tests, not a proof of synthesis or dynamic stability.",
            "Single-state chemistry tests can reject real mixed-valence compounds.",
            "Saved energies were used for stability; fresh force/convergence checks are separate.",
            "Original frozen evaluation and selections are unchanged.",
        ],
    }
    (OUT / "validity_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
