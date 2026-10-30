"""Read the published relaxed batches without silently losing source identities."""

from pathlib import Path
import numpy as np
import pandas as pd

from .common import database, finish, fingerprint, init_structures, receipt, save_structure, structure_record
from reference_baselines.core.io import sha256


def binary_mask(value, count, name):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != (count,) or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name}: expected a binary vector of length {count}, got {array.shape}")
    return array.astype(bool)


def prepare_release(config):
    import torch
    from pymatgen.core import Structure

    output = Path(config["output"])
    sources = []
    for batch in range(config["start_batch"], config["end_batch"] + 1):
        for name in ("relaxed_mattersim.json", "indicators_mattersim_mp_20.pt"):
            path = Path(config["release_root"]) / str(batch) / name
            if not path.is_file():
                raise FileNotFoundError(f"Required batch file missing: {path}")
            sources.append({"path": str(path), "sha256": sha256(path)})
    identity = fingerprint(sources)
    marker = output / "prepare.json"
    if receipt(marker, identity):
        return
    rows = []
    embedding_row = 0
    with database(output / "structures.sqlite") as db:
        init_structures(db)
        db.execute("DELETE FROM structures WHERE kind='candidate'")
        for batch in range(config["start_batch"], config["end_batch"] + 1):
            directory = Path(config["release_root"]) / str(batch)
            frame = pd.read_json(directory / "relaxed_mattersim.json")
            if not frame.index.is_unique or not {"structure", "e_relax", "num_sites", "exception"}.issubset(frame):
                raise ValueError(f"Invalid JSON schema or duplicate indices in batch {batch}")
            # Official release only: contains pickled NumPy arrays as well as tensors.
            indicators = torch.load(directory / "indicators_mattersim_mp_20.pt", map_location="cpu", weights_only=False)
            if not {"vu", "vn"}.issubset(indicators) or not set(indicators).issubset({"vu", "vn", "vun"}):
                raise ValueError(f"Unexpected indicator keys in batch {batch}")
            vu = binary_mask(indicators["vu"], len(frame), "vu")
            vn = binary_mask(indicators["vn"], len(frame), "vn")
            vun = vu & vn
            if "vun" in indicators and not np.array_equal(vun, binary_mask(indicators["vun"], len(frame), "vun")):
                raise ValueError(f"VUN mismatch in batch {batch}")
            for position, (index, raw) in enumerate(frame.iterrows()):
                identifier = f"candidate:{batch}:{index}"
                try:
                    raw_energy = float(raw.e_relax)
                except (TypeError, ValueError):
                    raw_energy = float("nan")
                row = {"candidate_row": len(rows), "id": identifier, "eval_folder": batch,
                       "idx": str(index), "source_position": position, "turn": batch // 10,
                       "vu": bool(vu[position]), "vn": bool(vn[position]), "vun": bool(vun[position]),
                       "embedding_row": -1, "eligible": False, "chemical_system": "", "formula": "",
                       "structure_hash": "", "num_atoms": 0, "raw_energy_eV": raw_energy,
                       "exclusion": ""}
                errors = []
                try:
                    structure = Structure.from_dict(raw.structure)
                    if not structure.is_ordered or len(structure) < 1:
                        raise ValueError("Need a nonempty ordered structure")
                    if len(structure) != raw.num_sites:
                        raise ValueError("num_sites disagrees with the saved structure")
                    record = structure_record(identifier, "candidate", structure)
                    row.update(chemical_system=record[2], formula=record[3], num_atoms=record[4], structure_hash=record[5])
                    save_structure(db, record)
                except Exception as exc:
                    errors.append(f"structure_error:{type(exc).__name__}:{exc}")
                if bool(raw.exception):
                    errors.append("relaxation_exception")
                if not np.isfinite(row["raw_energy_eV"]):
                    errors.append("nonfinite_energy")
                if not vun[position]:
                    errors.append("not_vun")
                if not errors:
                    row.update(eligible=True, embedding_row=embedding_row)
                    embedding_row += 1
                row["exclusion"] = ";".join(errors)
                rows.append(row)
            print(f"Prepared batch {batch}: {len(frame)} source rows", flush=True)
    manifest = pd.DataFrame(rows)
    if manifest.id.duplicated().any() or not embedding_row:
        raise ValueError("Duplicate candidate IDs or no eligible candidates")
    path = output / "candidates.csv"
    manifest.to_csv(path, index=False)
    finish(marker, identity, [path], source_files=sources, source_rows=len(manifest), eligible_rows=embedding_row,
           exclusions=manifest.loc[~manifest.eligible, "exclusion"].value_counts().to_dict())
