"""Model-native, fixed-bank hull labels; never mix screening pools into a hull."""

from pathlib import Path
import inspect
import json
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.core.thermodynamics import FixedHull
from .common import fingerprint, finish, read_json, receipt
from .indexing import subsystems


def label_candidates(candidates, phases, candidate_predictions, phase_predictions, energy_kind):
    """Pure calculation usable by both model backends and small offline fixtures."""
    candidate_predictions = candidate_predictions.set_index("id", verify_integrity=True)
    phase_predictions = phase_predictions.set_index("id", verify_integrity=True)
    phase_groups = {system: group for system, group in phases.groupby("chemical_system")}
    results, audits = [], []
    for system, group in candidates.groupby("chemical_system", sort=True):
        elements = system.split("-")
        bank = pd.concat([phase_groups[sub] for sub in subsystems(system) if sub in phase_groups], ignore_index=True) if any(sub in phase_groups for sub in subsystems(system)) else phases.iloc[:0]
        missing_elements = [e for e in elements if e not in phase_groups]
        failed = [row.id for row in bank.itertuples() if row.id not in phase_predictions.index or phase_predictions.loc[row.id, "status"] != "ok"]
        audit = {"chemical_system": system, "phase_count": len(bank), "failed_phase_count": len(failed),
                 "missing_elements": ",".join(missing_elements), "complete": not missing_elements and not failed,
                 "failed_phase_ids": json.dumps(failed)}
        audits.append(audit)
        refs, unary = [], {}
        solver = None
        if audit["complete"]:
            for phase in bank.itertuples():
                energy = float(phase_predictions.loc[phase.id, "energy_per_atom"])
                refs.append({"entry_id": phase.id, "formula": phase.formula, "energy_per_atom": energy})
                if phase.chemical_system in elements:
                    unary[phase.chemical_system] = min(unary.get(phase.chemical_system, np.inf), energy)
            solver = FixedHull(elements, refs)
        for candidate in group.itertuples():
            row = {"candidate_row": candidate.candidate_row, "id": candidate.id,
                   "eval_folder": candidate.eval_folder, "idx": str(candidate.idx), "turn": candidate.turn,
                   "chemical_system": system, "formula": candidate.formula,
                   "structure_hash": candidate.structure_hash, "energy_kind": energy_kind,
                   "hull_status": "ok", "hull_error": "", "phase_count": len(bank),
                   "energy_per_atom": np.nan, "formation_energy_per_atom": np.nan,
                   "hull_energy_per_atom": np.nan, "e_above_hull_proxy": np.nan,
                   "decomposition": "", "hull_method": ""}
            if candidate.id not in candidate_predictions.index or candidate_predictions.loc[candidate.id, "status"] != "ok":
                row.update(hull_status="candidate_prediction_error", hull_error="Candidate prediction unavailable or failed")
            elif solver is None:
                row.update(hull_status="incomplete_reference_hull", hull_error=f"Missing elements: {missing_elements}; failed phases: {len(failed)}")
            else:
                try:
                    from pymatgen.core import Composition
                    energy = float(candidate_predictions.loc[candidate.id, "energy_per_atom"])
                    hull, products, method = solver.minimum(candidate.formula)
                    comp = Composition(candidate.formula)
                    formation = energy if energy_kind == "formation_energy_per_atom" else energy - sum(comp.get_atomic_fraction(e)*unary[e] for e in elements)
                    row.update(energy_per_atom=energy, formation_energy_per_atom=formation,
                               hull_energy_per_atom=hull, e_above_hull_proxy=energy-hull,
                               decomposition=json.dumps(products, separators=(",", ":")), hull_method=method)
                except Exception as exc:
                    row.update(hull_status="hull_error", hull_error=f"{type(exc).__name__}:{exc}")
            results.append(row)
    return pd.DataFrame(results).sort_values("candidate_row"), pd.DataFrame(audits)


def evaluate_hulls(config, name):
    output = Path(config["output"])
    status = read_json(output / f"{name}_prediction_status.json")
    for file, key in (("phase_manifest.csv", "phase_manifest_sha256"), ("evaluation_candidates.csv", "evaluation_manifest_sha256")):
        if sha256(output / file) != status[key]:
            raise ValueError("Prediction manifest is stale; resume the predict stage")
    if not status["complete"]:
        raise ValueError(f"{name} predictions remain pending. Resume predict before calculating hulls.")
    for kind in ("candidate", "phase"):
        if sha256(output / f"{name}_{kind}_predictions.csv") != status["prediction_export_sha256"][kind]:
            raise ValueError("Interrupted or changed prediction export; resume predict")
    identity = fingerprint({"model": status["fingerprint"], "phases": sha256(output / "phase_manifest.csv"),
                            "candidates": sha256(output / "evaluation_candidates.csv"),
                            "phase_predictions": sha256(output / f"{name}_phase_predictions.csv"),
                            "candidate_predictions": sha256(output / f"{name}_candidate_predictions.csv"),
                            "code": sha256(Path(__file__)), "solver": sha256(Path(inspect.getfile(FixedHull)))})
    marker = output / f"{name}_hull.json"
    if marker.exists() and read_json(marker)["identity"] == identity and receipt(marker, identity):
        return
    candidates = pd.read_csv(output / "evaluation_candidates.csv", keep_default_na=False)
    phases = pd.read_csv(output / "phase_manifest.csv", keep_default_na=False)
    cp = pd.read_csv(output / f"{name}_candidate_predictions.csv", keep_default_na=False)
    pp = pd.read_csv(output / f"{name}_phase_predictions.csv", keep_default_na=False)
    for expected, actual in ((candidates, cp), (phases, pp)):
        joined = expected[["id", "structure_hash"]].merge(actual[["id", "structure_hash"]], on="id", validate="one_to_one", suffixes=("_a", "_b"))
        if len(joined) != len(expected) or not (joined.structure_hash_a == joined.structure_hash_b).all():
            raise ValueError("Predictions do not match required structure identities")
    labels, audits = label_candidates(candidates, phases, cp, pp, "formation_energy_per_atom" if name == "megnet" else "total_energy_per_atom")
    labels.to_csv(output / f"{name}_labels.csv", index=False)
    audits.to_csv(output / f"{name}_hull_coverage.csv", index=False)
    finish(marker, identity, [output / f"{name}_labels.csv", output / f"{name}_hull_coverage.csv"],
           label_status_counts=labels.hull_status.value_counts().to_dict(), native_energy_corrections="none",
           fixed_external_hull=True, candidate_entries_in_hull=False)
