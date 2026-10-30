"""Fresh MatterSim energies/forces for fixed selections; optional bounded re-relaxation."""

from reference_baselines.core.audit_io import read_csv

import argparse
from importlib.metadata import version
import json
import time
import warnings

import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.evaluation.selection_audit import OUT
from reference_baselines.paths import DATA_DIR, RESULTS_DIR
from reference_baselines.structure.case_study import load_candidates


def load_potential(device="cpu"):
    import torch
    from mattersim.forcefield.potential import Potential
    from mattersim.forcefield.m3gnet.m3gnet import M3Gnet

    checkpoint_path = DATA_DIR / "mattersim/mattersim-v1.0.0-1M.pth"
    provenance = json.loads((checkpoint_path.parent / "source.json").read_text())
    if sha256(checkpoint_path) != provenance["sha256"]:
        raise ValueError("Checkpoint hash differs from recorded download")
    # Use the documented model classes directly to keep all files in the repo.
    # Potential.from_checkpoint unconditionally creates a directory in the home folder.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = M3Gnet(device=device, **checkpoint["model_args"]).to(device)
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    # New MatterSim versions register this fixed analytical basis table as a
    # buffer. The original checkpoint predates that registration. It is not a
    # learned parameter; reject any other missing or unexpected state.
    if set(loaded.missing_keys) - {"sbf.coef"} or loaded.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint incompatibility: {loaded}")
    provenance["default_analytic_buffers"] = loaded.missing_keys
    model.eval()
    return Potential(model, device=device), provenance


def pilot_rows(frame, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for group in ["added", "removed"]:
        for passing in [False, True]:
            pool = frame[
                frame.selection_group.eq(group) & frame.passing.eq(passing)
            ].candidate_row.to_numpy()
            rows.extend(
                rng.choice(np.sort(pool), min(3, len(pool)), replace=False).tolist()
            )
    return rows


def infer(potential, atoms):
    from mattersim.datasets.utils.build import build_dataloader

    # MatterSim 1.2.5's legacy converter returns plain PyG Data, whose
    # three_body_indices are not offset between graphs. Its batch_to_dict
    # assumes global indices. Single-graph batches avoid this inconsistency.
    # Compare this path with the public ASE calculator in the integration test.
    loader = build_dataloader(atoms, batch_size=1, only_inference=True, shuffle=False)
    return potential.predict_properties(
        loader, include_forces=True, include_stresses=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 checks the full union of original/angle test selections",
    )
    parser.add_argument(
        "--relax-steps",
        type=int,
        default=0,
        help="Bounded relaxation of 12 seeded pilot rows plus illustrative cases",
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--tighten-forces",
        action="store_true",
        help="Enforce atomic as well as cell-filter force convergence on completed shards",
    )
    parser.add_argument(
        "--rerelax-all",
        action="store_true",
        help="Re-relax the full fixed selection union, with per-row resume",
    )
    parser.add_argument(
        "--shard",
        type=int,
        nargs=2,
        metavar=("INDEX", "COUNT"),
        help="Disjoint re-relaxation shard for parallel cluster/CPU workers",
    )
    args = parser.parse_args()
    warnings.filterwarnings(
        "ignore", message="gcd is deprecated", category=FutureWarning
    )
    import torch
    from pymatgen.io.ase import AseAtomsAdaptor

    torch.set_num_threads(args.threads)
    if args.tighten_forces:
        potential, _ = load_potential()
        labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index(
            "candidate_row"
        )
        tighten_forces(potential, labels)
        return
    frame = read_csv(OUT / "candidate_audit.csv")
    selected = frame[
        frame.split.eq("test") & frame.selection_group.ne("neither")
    ].sort_values("candidate_row")
    if args.limit:
        selected = selected.head(args.limit)
    if args.shard:
        index, count = args.shard
        if not args.rerelax_all or not 0 <= index < count:
            parser.error("Sharding needs --rerelax-all and 0 <= INDEX < COUNT")
        selected = selected.iloc[index::count]
    structures = load_candidates(selected)
    potential, provenance = load_potential()
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    if args.rerelax_all:
        suffix = f"_shard{args.shard[0]}of{args.shard[1]}" if args.shard else ""
        relax_all(
            selected, structures, potential, labels, args.relax_steps or 300, suffix
        )
        return
    records = []
    started = time.perf_counter()
    for offset in range(0, len(selected), 32):
        chunk = selected.iloc[offset : offset + 32]
        atoms = [
            AseAtomsAdaptor.get_atoms(structures[row]) for row in chunk.candidate_row
        ]
        energies, forces, stresses = infer(potential, atoms)
        for row, energy, force, stress in zip(
            chunk.itertuples(), energies, forces, stresses
        ):
            label = labels.loc[row.candidate_row]
            fresh_hull = (
                energy / row.num_atoms
                + label.correction_per_atom
                - label.hull_energy_per_atom
            )
            records.append(
                {
                    "candidate_row": row.candidate_row,
                    "selection_group": row.selection_group,
                    "formula": row.formula,
                    "saved_hull_eV_per_atom": row.e_above_hull_proxy,
                    "fresh_hull_eV_per_atom": fresh_hull,
                    "fresh_raw_energy_eV": energy,
                    "raw_energy_difference_eV_per_atom": energy / row.num_atoms
                    - label.raw_energy_per_atom,
                    "fmax_eV_per_A": float(np.linalg.norm(force, axis=1).max()),
                    "stress_max_abs_GPa": float(np.abs(stress).max()),
                    "saved_passing": bool(row.passing),
                    "fresh_passing": bool(fresh_hull <= 0.1),
                }
            )
        if (offset // 32 + 1) % 10 == 0 or offset + 32 >= len(selected):
            print(
                "Fresh MatterSim checks:",
                min(offset + 32, len(selected)),
                "/",
                len(selected),
                flush=True,
            )
    result = pd.DataFrame(records)
    suffix = "" if not args.limit else f"_first{args.limit}"
    result.to_csv(OUT / f"fresh_mattersim{suffix}.csv", index=False)
    summary = {
        "checkpoint": provenance,
        "mattersim_version": version("mattersim"),
        "graph_batch_size": 1,
        "candidate_count": len(result),
        "seconds": time.perf_counter() - started,
        "all_selected_checked": not bool(args.limit),
        "relaxed_again": False,
        "model_training": False,
        "energy_absolute_difference_per_atom_max": float(
            result.raw_energy_difference_eV_per_atom.abs().max()
        ),
        "energy_absolute_difference_per_atom_median": float(
            result.raw_energy_difference_eV_per_atom.abs().median()
        ),
        "pass_label_disagreements": int(
            result.saved_passing.ne(result.fresh_passing).sum()
        ),
        "forces_above_0_05": int(result.fmax_eV_per_A.gt(0.05).sum()),
        "force_max_eV_per_A": float(result.fmax_eV_per_A.max()),
        "stress_limit": "Stress is recorded, not silently treated as a force-convergence certificate.",
        "energy_method": "Fresh MatterSim total energy + saved MP correction of the identical structure, minus the unchanged fixed MP2023 hull",
        "limits": "Same MLIP checkpoint is a reproducibility check, not an independent model or DFT validation.",
    }
    (OUT / f"fresh_mattersim{suffix}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.relax_steps:
        relax_pilot(selected, structures, potential, labels, args.relax_steps)


def relax_all(frame, structures, potential, labels, steps, suffix=""):
    """Same optimizer/settings for both fixed selections, without reranking."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from eval.vsun.structure_summary import VasprunLike, IdentityCorrectionScheme
    from reference_baselines.structure.validity import geometry_checks

    target = OUT / "rerelaxed_all"
    target.mkdir(exist_ok=True)
    result_path = OUT / f"rerelaxed_all{suffix}.csv"
    config_path = OUT / f"rerelaxed_all_protocol{suffix}.json"
    config = {
        "candidate_rows": frame.candidate_row.tolist(),
        "max_steps": steps,
        "force_cutoff": 0.05,
        "optimizer": "ASE FIRE + FrechetCellFilter",
        "energy_corrections": "recompute MP2020 after relaxation; unchanged MP2023 hull",
        "selection": "Fixed original and angle-rule test selection union; no reranking or refill",
        "checkpoint_sha256": sha256(DATA_DIR / "mattersim/mattersim-v1.0.0-1M.pth"),
    }
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Cannot resume with changed re-relaxation settings")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    records = read_csv(result_path).to_dict("records") if result_path.exists() else []
    completed = {int(row["candidate_row"]) for row in records}
    calculator = MatterSimCalculator(potential=potential, device="cpu")
    for row in frame.itertuples():
        if row.candidate_row in completed:
            continue
        record = {
            "candidate_row": row.candidate_row,
            "selection_group": row.selection_group,
            "formula": row.formula,
            "saved_hull": row.e_above_hull_proxy,
            "status": "ok",
        }
        try:
            atoms = AseAtomsAdaptor.get_atoms(structures[row.candidate_row])
            atoms.calc = calculator
            initial = float(atoms.get_potential_energy())
            initial_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            filtered = FrechetCellFilter(atoms)
            optimizer = FIRE(filtered, logfile=None)
            converged = bool(optimizer.run(fmax=0.05, steps=steps))
            energy = float(atoms.get_potential_energy())
            if not np.isfinite(energy):
                raise ValueError("Nonfinite final energy")
            structure = AseAtomsAdaptor.get_structure(atoms)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                entry = VasprunLike(structure, energy).get_computed_entry(
                    energy_correction_scheme=IdentityCorrectionScheme()
                )
                corrected = MaterialsProject2020Compatibility().process_entry(entry)
            if corrected is None:
                raise ValueError("MP compatibility rejected the final structure")
            hull = float(
                corrected.energy_per_atom
                - labels.loc[row.candidate_row, "hull_energy_per_atom"]
            )
            structure.to(filename=str(target / f"candidate_{row.candidate_row}.cif"))
            record.update(
                converged=converged,
                steps=optimizer.nsteps,
                initial_fmax=initial_force,
                final_fmax=float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                filtered_fmax=float(
                    np.linalg.norm(filtered.get_forces(), axis=1).max()
                ),
                energy_change_per_atom=(energy - initial) / len(atoms),
                correction_per_atom=float(corrected.correction / len(atoms)),
                correction_warning_count=len(caught),
                rerelaxed_hull=hull,
                rerelaxed_passing=bool(hull <= 0.1),
                final_geometry_valid=geometry_checks(structure)[
                    "periodic_geometry_valid"
                ],
            )
        except Exception as exc:
            record.update(
                status="error",
                converged=False,
                rerelaxed_passing=False,
                final_geometry_valid=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)
        pd.DataFrame(records).to_csv(result_path, index=False)
        if len(records) % 25 == 0 or len(records) == len(frame):
            print("Full re-relaxation:", len(records), "/", len(frame), flush=True)


def tighten_forces(potential, labels):
    """A deformed cell filter can report a smaller norm than actual atom forces.

    Apply the same extra check to both methods, with the same total 300-step cap.
    This phase is resumable and does not use candidate energy to choose retries.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from eval.vsun.structure_summary import VasprunLike, IdentityCorrectionScheme
    from reference_baselines.structure.validity import geometry_checks

    calculator = MatterSimCalculator(potential=potential, device="cpu")
    refined = []
    for shard in range(4):
        path = OUT / f"rerelaxed_all_shard{shard}of4.csv"
        table = read_csv(path).set_index("candidate_row", drop=False)
        for row in table[table.converged & table.final_fmax.gt(0.05)].itertuples():
            target = OUT / f"rerelaxed_all/candidate_{row.candidate_row}.cif"
            atoms = AseAtomsAdaptor.get_atoms(Structure.from_file(target))
            atoms.calc = calculator
            filtered = FrechetCellFilter(atoms)
            optimizer = FIRE(filtered, logfile=None)
            optimizer.run(fmax=0.04, steps=max(0, 300 - int(row.steps)))
            physical = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            filter_force = float(np.linalg.norm(filtered.get_forces(), axis=1).max())
            converged = bool(physical <= 0.05 and filter_force <= 0.05)
            energy = float(atoms.get_potential_energy())
            structure = AseAtomsAdaptor.get_structure(atoms)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                entry = VasprunLike(structure, energy).get_computed_entry(
                    energy_correction_scheme=IdentityCorrectionScheme()
                )
                corrected = MaterialsProject2020Compatibility().process_entry(entry)
            if corrected is None:
                raise ValueError("MP compatibility rejected an atom-force refinement")
            label = labels.loc[row.candidate_row]
            old_raw_per_atom = (
                row.rerelaxed_hull
                + label.hull_energy_per_atom
                - row.correction_per_atom
            )
            hull = float(corrected.energy_per_atom - label.hull_energy_per_atom)
            updates = {
                "converged": converged,
                "steps": int(row.steps) + optimizer.nsteps,
                "final_fmax": physical,
                "filtered_fmax": filter_force,
                "energy_change_per_atom": energy / len(atoms)
                - old_raw_per_atom
                + row.energy_change_per_atom,
                "correction_per_atom": float(corrected.correction / len(atoms)),
                "rerelaxed_hull": hull,
                "rerelaxed_passing": hull <= 0.1,
                "final_geometry_valid": geometry_checks(structure)[
                    "periodic_geometry_valid"
                ],
            }
            for key, value in updates.items():
                table.loc[row.candidate_row, key] = value
            structure.to(filename=str(target))
            table.to_csv(path, index=False)
            refined.append(row.candidate_row)
    (OUT / "atomic_force_refinement.json").write_text(
        json.dumps(
            {
                "candidate_rows": refined,
                "rule": "Previously cell-filter-converged but actual maximum atomic force >0.05; both methods, no energy-based retries",
                "total_step_cap": 300,
                "refinement_filter_target": 0.04,
                "accepted_convergence": "Actual atomic and cell-filter norms both <=0.05 eV/angstrom",
            },
            indent=2,
        )
        + "\n"
    )
    print("Atomic-force refinement completed:", len(refined), "structures", flush=True)


def relax_pilot(frame, structures, potential, labels, steps):
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from eval.vsun.structure_summary import VasprunLike, IdentityCorrectionScheme

    sampled = pilot_rows(frame)
    cases = read_csv(OUT / "cases.csv").candidate_row.tolist()
    selected_rows = sorted(set(sampled + cases).intersection(structures))
    calculator = MatterSimCalculator(potential=potential, device="cpu")
    records = []
    target_dir = OUT / "rerelaxed"
    target_dir.mkdir(exist_ok=True)
    for i, row in enumerate(selected_rows, 1):
        atoms = AseAtomsAdaptor.get_atoms(structures[row])
        atoms.calc = calculator
        initial = float(atoms.get_potential_energy())
        initial_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        filtered = FrechetCellFilter(atoms)
        optimizer = FIRE(filtered, logfile=str(target_dir / f"candidate_{row}.log"))
        converged = bool(optimizer.run(fmax=0.05, steps=steps))
        energy = float(atoms.get_potential_energy())
        structure = AseAtomsAdaptor.get_structure(atoms)
        entry = VasprunLike(structure, energy).get_computed_entry(
            energy_correction_scheme=IdentityCorrectionScheme()
        )
        corrected = MaterialsProject2020Compatibility().process_entry(entry)
        if corrected is None:
            raise ValueError(
                "Fresh MP compatibility rejected a relaxed pilot structure"
            )
        hull = float(
            corrected.energy_per_atom - labels.loc[row, "hull_energy_per_atom"]
        )
        structure.to(filename=str(target_dir / f"candidate_{row}.cif"))
        records.append(
            {
                "candidate_row": row,
                "sampling_group": "seeded_random"
                if row in sampled
                else "illustrative_case",
                "converged": converged,
                "steps": optimizer.nsteps,
                "initial_fmax": initial_force,
                "final_fmax": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
                "filtered_fmax_includes_cell": float(
                    np.linalg.norm(filtered.get_forces(), axis=1).max()
                ),
                "energy_change_per_atom": (energy - initial) / len(atoms),
                "rerelaxed_hull": hull,
                "rerelaxed_passing": bool(hull <= 0.1),
                "saved_hull": float(labels.loc[row, "e_above_hull_proxy"]),
            }
        )
        pd.DataFrame(records).to_csv(OUT / "rerelaxed_pilot.csv", index=False)
        print(
            "Re-relaxed pilot:",
            i,
            "/",
            len(selected_rows),
            "row",
            row,
            "converged",
            converged,
            flush=True,
        )
    (OUT / "rerelaxed_protocol.json").write_text(
        json.dumps(
            {
                "random_sampling": "3 seeded rows from each of added/removed x saved pass/fail; seed=2026; plus illustrative cases",
                "max_steps": steps,
                "force_cutoff_eV_per_A": 0.05,
                "optimizer": "ASE FIRE with FrechetCellFilter; atoms and cell relaxed",
                "unconverged": "Retain explicit failure flag; do not count as confirmed stable",
                "purpose": "Diagnostic convergence pilot, not an unbiased full-set new performance estimate",
                "new_MP_corrections_computed_after_relaxation": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
