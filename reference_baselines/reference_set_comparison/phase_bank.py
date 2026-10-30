"""Import all relevant MP2023 structures, without DFT polymorph pruning."""

from pathlib import Path
import json
import pickle
import pandas as pd

from reference_baselines.core.io import sha256
from .common import canonical, database, fingerprint, finish, init_structures, receipt, save_structure, structure_record
from .indexing import subsystems


def lmdb_entries(path, wanted):
    import lmdb
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as transaction:
            available = pickle.loads(transaction.get(b"chemical_systems"))
            for system in sorted(set(available) & wanted):
                formulas = pickle.loads(transaction.get((system + ".reduced_formulas").encode()))
                for formula in formulas:
                    prefix = system + "." + formula
                    count = pickle.loads(transaction.get((prefix + ".length").encode()))
                    for index in range(count):
                        key = prefix + "." + str(index)
                        # Trusted, hash-verified published LMDB, as in the existing hull loader.
                        entry = pickle.loads(transaction.get(key.encode()))
                        yield key, system, formula, entry["structure"]
    finally:
        env.close()


def prepare_phases(config):
    from pymatgen.core import Structure

    output = Path(config["output"])
    source = Path(config["hull_database"])
    if not source.is_file():
        raise FileNotFoundError(f"Download the pinned MP2023 database first: {source}")
    # Independent of which method selects a candidate: use ALL eligible candidate systems.
    candidates = pd.read_csv(output / "candidate_metadata.csv", keep_default_na=False)
    wanted = {s for system in candidates.chemical_system.unique() for s in subsystems(system)}
    identity = fingerprint({"source": sha256(source), "systems": sorted(wanted), "code": sha256(Path(__file__))})
    marker = output / "phases.json"
    if receipt(marker, identity):
        return
    failures, counts, phase_manifest = [], {}, []
    with database(output / "structures.sqlite") as db:
        init_structures(db)
        db.execute("DELETE FROM structures WHERE kind='phase'")
        for key, system, formula, raw in lmdb_entries(source, wanted):
            identifier = "phase:" + key
            try:
                structure = raw if isinstance(raw, Structure) else Structure.from_dict(raw)
                if not structure.is_ordered or not len(structure):
                    raise ValueError("Unsupported empty/disordered reference structure")
                record = structure_record(identifier, "phase", structure)
                if record[2] != system:
                    raise ValueError("LMDB chemical system disagrees with structure")
            except Exception as exc:
                # Keep failed entries in the bank manifest: their systems cannot get a complete hull.
                failures.append({"id": identifier, "chemical_system": system, "error": f"{type(exc).__name__}:{exc}"})
                record = (identifier, "phase", system, formula, 0, fingerprint([key, str(raw)]), canonical({"invalid": str(raw)}))
            save_structure(db, record)
            phase_manifest.append({"id": identifier, "chemical_system": system, "structure_hash": record[5], "formula": formula})
            counts[system] = counts.get(system, 0) + 1
            if len(phase_manifest) % 10000 == 0:
                db.commit()
                print(f"Imported {len(phase_manifest):,} reference phases", flush=True)
    if not phase_manifest:
        raise ValueError("No reference phases found for this candidate chemistry")
    manifest_path = output / "phase_manifest.csv"
    pd.DataFrame(phase_manifest).to_csv(manifest_path, index=False)
    finish(marker, identity, [manifest_path], phase_count=len(phase_manifest), systems=counts,
           missing_systems=sorted(wanted - set(counts)), source_sha256=sha256(source),
           structure_failures=failures, selection_independent=True, dft_energy_pruning=False)
