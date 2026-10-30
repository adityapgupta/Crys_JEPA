"""Check equivalent representations; require the real scaler for encoder claims."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher

from reference_baselines.baselines.formula_balance import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, DATA_DIR


def lattice_vector(structure):
    """Same float32 polar factor and six entries as original V_screen.py."""
    matrix = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    u, singular, vt = torch.linalg.svd(matrix)
    rotation = u @ vt
    positive = vt.T @ torch.diag(singular) @ vt
    result = rotation @ positive @ rotation.T
    result[torch.abs(result) < 1e-5] = 0
    tri = torch.triu_indices(3, 3)
    return result[tri[0], tri[1]] / len(structure) ** (1 / 3)


def variants(structure):
    rng = np.random.default_rng(20261022)
    perm = rng.permutation(len(structure))
    shuffled = Structure(
        structure.lattice,
        [structure[i].specie for i in perm],
        structure.frac_coords[perm],
    )
    shifted = structure.copy()
    shifted.translate_sites(
        range(len(structure)),
        [0.137, 0.271, 0.389],
        frac_coords=True,
        to_unit_cell=True,
    )
    angle = np.deg2rad(37)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    rotated = Structure(
        Lattice(structure.lattice.matrix @ rotation.T),
        structure.species,
        structure.frac_coords,
    )
    p = np.eye(3)[[1, 2, 0]]
    basis = Structure(
        Lattice(p @ structure.lattice.matrix),
        structure.species,
        structure.frac_coords @ p.T,
    )
    supercell = structure.copy()
    supercell.make_supercell([2, 1, 1])
    return {
        "identity": structure.copy(),
        "atom_order": shuffled,
        "origin_shift": shifted,
        "rigid_rotation": rotated,
        "basis_permutation": basis,
        "supercell": supercell,
    }


def encode(model, structure, scaler):
    matrix = (lattice_vector(structure) - scaler[0]) / scaler[1]
    coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    numbers = torch.tensor(structure.atomic_numbers, dtype=torch.long)
    x = torch.cat(
        [
            coords,
            F.one_hot(numbers, 100).float(),
            matrix[None, :].repeat(len(structure), 1),
        ],
        dim=1,
    )
    with torch.inference_mode():
        return model.encode(x, torch.zeros(len(structure), dtype=torch.long))[0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scaler", type=Path, help="Exact mean_std_scaler.pt used for saved embeddings"
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    torch.set_num_threads(2)
    source = read_csv(DATA_DIR / "mp_20_train.csv.gz")
    metadata = read_csv(ROOT / "embeddings/data/metadata.csv")
    source = source.set_index("material_id")
    rng = np.random.default_rng(20261022)
    sampled = np.sort(rng.choice(len(metadata), 16, replace=False))
    model, scaler = None, None
    if args.scaler:
        from components.jepa.frame.jepa import JEPA

        scaler = torch.load(args.scaler, map_location="cpu", weights_only=True)
        if (
            scaler.shape != (2, 6)
            or not torch.isfinite(scaler).all()
            or not (scaler[1] > 0).all()
        ):
            raise ValueError("Invalid exact scaler")
        config = SimpleNamespace(
            model=SimpleNamespace(hidden_dim=512, layers=8, attn_head=16, dropout=0),
            device="cpu",
        )
        model = JEPA(config).eval()
        checkpoint = torch.load(
            DATA_DIR / "jepa_audit/model_1972.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(
            {
                k.removeprefix("module."): v
                for k, v in checkpoint["model_state_dict"].items()
            },
            strict=True,
        )
    saved = np.load(ROOT / "embeddings/data/embeddings.npy", mmap_mode="r")
    records, reproduction = [], []
    matcher = StructureMatcher(primitive_cell=True, attempt_supercell=True)
    for row in sampled:
        identifier = metadata.iloc[row].material_id
        original = Structure.from_str(source.loc[identifier, "cif"], fmt="cif")
        original_vector = lattice_vector(original)
        baseline = encode(model, original, scaler) if model is not None else None
        if baseline is not None:
            reproduction.append(float(np.max(np.abs(baseline - saved[row]))))
        for kind, structure in variants(original).items():
            record = {
                "reference_row": int(row),
                "material_id": identifier,
                "variant": kind,
                "structure_match": bool(matcher.fit(original, structure)),
                "lattice_feature_max_change": float(
                    torch.max(torch.abs(lattice_vector(structure) - original_vector))
                ),
                "atoms": len(structure),
            }
            if baseline is not None:
                embedding = encode(model, structure, scaler)
                cosine = np.dot(baseline, embedding) / (
                    np.linalg.norm(baseline) * np.linalg.norm(embedding)
                )
                record["angle_degrees"] = float(
                    np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
                )
                record["relative_embedding_change"] = float(
                    np.linalg.norm(embedding - baseline) / np.linalg.norm(baseline)
                )
            records.append(record)
    pd.DataFrame(records).to_csv(OUT / "encoder_representation_checks.csv", index=False)
    reproduced = bool(reproduction and max(reproduction) <= 3e-5)
    report = {
        "sample": "16 seeded known MP20 crystals, six representations each",
        "date": "2026-10-22",
        "scaler_provided": bool(args.scaler),
        "saved_embeddings_reproduced": reproduced,
        "max_saved_embedding_error": max(reproduction) if reproduction else None,
        "status": "measured_on_reproduced_encoder"
        if reproduced
        else "blocked_exact_preprocessing_provenance",
        "reason": "Need original mean_std_scaler.pt and extraction script; published archive has checkpoint but no scaler. Do not substitute a scaler fitted on MP20.",
        "feature_checks_are_not_encoder_invariance": True,
        "sha256": {
            "checkpoint": sha256(DATA_DIR / "jepa_audit/model_1972.pt"),
            "scaler": sha256(args.scaler) if args.scaler else None,
        },
    }
    (OUT / "encoder_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
