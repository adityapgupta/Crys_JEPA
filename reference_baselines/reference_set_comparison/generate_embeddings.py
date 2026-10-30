"""Download the published JEPA checkpoint and embed MP-20 in input CSV order.

Run from the repository root with ``python -m
reference_baselines.reference_set_comparison.generate_embeddings --help``.
By default, reconstruct an approximate lattice scaler from the published JEPA
pretraining data. Cache it for reuse with generated candidates.
"""

import argparse
import json
from pathlib import Path
import shutil
import ssl
from types import SimpleNamespace
import urllib.request
import zipfile

from reference_baselines.core.io import sha256
from reference_baselines.datasets.fetch_release import BASE, ZIP_SIZE, REVISION, RangeFile
from reference_baselines.paths import ROOT


HERE = Path(__file__).resolve().parent
CHECKPOINT_MEMBER = "exp_logs/jepa/saved_model/model_1972.pt"
PREPROCESSING = "primitive-reduced-type-sorted-float32-v1"
TRAIN_FILE = "data/base/mp_20/train.csv.gz"
SCALER_FILES = ("data/jepa/mp.csv.gz", "data/jepa/mptrj.csv.gz")
DATA_HASHES = {
    TRAIN_FILE: "237437a664c436e1a092223cce5c00e8795cb3eb9d5ea958346a00adaa6cfa9f",
    SCALER_FILES[0]: "a721fb3ca25dd8ea434640a83a68f853fdc457e74f7208883f448ae48e3a6c91",
    SCALER_FILES[1]: "66782e586dbc806c4269b5986eefe2ab21e4313724fae4e572e9a26ae98a71a8",
}


def download_data_file(filename, directory):
    """Fetch one pinned HF file and verify its published LFS SHA-256."""
    import certifi

    if filename not in DATA_HASHES:
        raise ValueError(f"Unsupported release file: {filename}")
    target = directory / filename
    if target.exists():
        if sha256(target) != DATA_HASHES[filename]:
            raise ValueError(f"Cached dataset hash mismatch: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        print(f"Downloading {filename}", flush=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(BASE + filename, context=context, timeout=120) as source:
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if sha256(temporary) != DATA_HASHES[filename]:
            raise ValueError(f"Downloaded dataset hash mismatch: {filename}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def download_checkpoint(directory):
    """Use checked HTTP ranges to extract one member, not the entire 7.39 GB ZIP."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "model_1972.pt"
    receipt = directory / "checkpoint_source.json"
    if target.exists():
        if not receipt.exists():
            raise FileExistsError(f"Unverified cached checkpoint: {target}")
        source = json.loads(receipt.read_text(encoding="utf-8"))
        if (source.get("revision") != REVISION
                or source.get("member") != CHECKPOINT_MEMBER
                or source.get("sha256") != sha256(target)):
            raise ValueError("Cached checkpoint does not match its download receipt")
        return target

    temporary = target.with_suffix(".pt.part")
    url = BASE + "exp_logs.zip"
    try:
        with zipfile.ZipFile(RangeFile(url, ZIP_SIZE)) as archive:
            info = archive.getinfo(CHECKPOINT_MEMBER)
            print(f"Downloading checkpoint: {info.compress_size:,} compressed bytes", flush=True)
            # ZipExtFile verifies the uncompressed member's CRC when reaching EOF.
            with archive.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        if temporary.stat().st_size != info.file_size:
            raise ValueError("Incomplete checkpoint download")
        checksum = sha256(temporary)
        temporary.replace(target)
        receipt.write_text(json.dumps({
            "url": url, "revision": REVISION, "member": CHECKPOINT_MEMBER,
            "bytes": info.file_size, "zip_crc32": info.CRC, "sha256": checksum,
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_train(path, limit=None):
    import pandas as pd

    # Keep strings such as the valid formula NaN; do not sort or drop any rows.
    table = pd.read_csv(path, keep_default_na=False, nrows=limit)
    if not {"material_id", "cif"}.issubset(table.columns):
        raise ValueError("Training CSV must contain material_id and cif")
    if table.empty or not table.material_id.is_unique or table.material_id.eq("").any():
        raise ValueError("Training CSV must have nonempty, unique material IDs")
    reserved = {"embedding_row", "source_row"}.intersection(table.columns)
    if reserved:
        raise ValueError(f"Input already contains reserved output columns: {reserved}")
    return table


def lattice_features(structure):
    """Six float32 polar-decomposed lattice features scaled by atom count."""
    import torch

    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    w, singular, vt = torch.linalg.svd(lattice)
    v = vt.transpose(0, 1)
    u = w @ vt
    polar = u @ (v @ torch.diag_embed(singular) @ vt) @ u.transpose(0, 1)
    polar[torch.abs(polar) < 1e-5] = 0.0
    indices = torch.triu_indices(3, 3)
    count = torch.tensor(len(structure), dtype=torch.float32)
    return polar[indices[0], indices[1]] / count.pow(1 / 3)


def prepare_structure(structure):
    """Follow the supplied embed.py: primitive cell, reduction, then type sort."""
    import torch

    structure = structure.get_primitive_structure().get_reduced_structure()
    if not structure.is_ordered or not 1 <= len(structure) <= 499:
        raise ValueError("Need an ordered structure with 1..499 sites")
    if any(not 1 <= number < 100 for number in structure.atomic_numbers):
        raise ValueError("Encoder supports atomic numbers 1..99")
    coordinates = torch.tensor(structure.frac_coords, dtype=torch.float32)
    numbers = torch.tensor(structure.atomic_numbers, dtype=torch.float32)
    key = (1000 * numbers + 100 * coordinates[:, 0]
           + 10 * coordinates[:, 1] + coordinates[:, 2])
    order = torch.argsort(key)
    features = lattice_features(structure)
    if not torch.isfinite(features).all() or not torch.isfinite(coordinates).all():
        raise ValueError("Nonfinite structure features")
    return structure, coordinates[order], numbers[order].long(), features


def resolve_scaler(args):
    """Reuse a supplied scaler or fit/cache the approved sampled approximation."""
    import pandas as pd
    import torch
    from pymatgen.core import Structure

    if args.scaler is not None:
        if not args.scaler.is_file():
            raise FileNotFoundError(args.scaler)
        return args.scaler, {"mode": "supplied", "preprocessing": PREPROCESSING}

    name = f"{PREPROCESSING}-cap{args.scaler_cap}-seed{args.scaler_seed}"
    target = args.artifact_dir / "scalers" / (name + ".pt")
    receipt = target.with_suffix(".json")
    if target.exists():
        if not receipt.is_file():
            raise FileExistsError(f"Scaler has no provenance receipt: {target}")
        info = json.loads(receipt.read_text(encoding="utf-8"))
        if (info.get("scaler_sha256") != sha256(target)
                or info.get("revision") != REVISION
                or info.get("preprocessing") != PREPROCESSING
                or info.get("scaler_cap") != args.scaler_cap
                or info.get("scaler_seed") != args.scaler_seed):
            raise ValueError(f"Cached scaler settings/hash mismatch: {target}")
        print(f"Reusing reconstructed scaler: {target}", flush=True)
        return target, info

    matrices, sources, failures = [], [], []
    for filename in SCALER_FILES:
        path = download_data_file(filename, args.artifact_dir / "hf")
        frame = pd.read_csv(path, keep_default_na=False)
        selected = frame.sample(min(args.scaler_cap, len(frame)), random_state=args.scaler_seed) if args.scaler_cap else frame
        good = 0
        for number, (row, cif) in enumerate(selected.cif.items(), 1):
            try:
                matrices.append(prepare_structure(Structure.from_str(cif, fmt="cif"))[3])
                good += 1
            except Exception as exc:
                failures.append({"file": filename, "source_row": int(row),
                                 "error": f"{type(exc).__name__}: {exc}"})
            if number % 1000 == 0 or number == len(selected):
                print(f"Scaler {path.name}: {number:,}/{len(selected):,}", flush=True)
        sources.append({"file": filename, "sha256": sha256(path), "total_rows": len(frame),
                        "sampled_source_rows": [int(i) for i in selected.index], "used_rows": good})
        del frame, selected
    if len(matrices) < 2:
        raise ValueError("Not enough valid structures to reconstruct the scaler")
    values = torch.stack(matrices)
    packed = torch.stack([values.mean(0), values.std(0)])
    if not torch.isfinite(packed).all() or not (packed[1] > 0).all():
        raise ValueError("Reconstructed scaler has invalid feature statistics")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.part")
    torch.save(packed, temporary)
    temporary.replace(target)
    info = {
        "mode": "reconstructed_approximate", "revision": REVISION,
        "preprocessing": PREPROCESSING, "scaler_cap": args.scaler_cap,
        "scaler_seed": args.scaler_seed, "used_rows": len(matrices),
        "sources": sources, "failed_rows": failures,
        "torch_version": str(torch.__version__), "pymatgen_version": package_version("pymatgen"),
        "scaler_sha256": sha256(target),
        "interpretation": "Reconstructed statistics; not verified against the original training scaler",
    }
    receipt.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved approximate scaler: {target}; skipped {len(failures)} scaler-fit rows", flush=True)
    return target, info


def package_version(name):
    from importlib.metadata import version
    return version(name)


def encode_batch(prepared, model, means, stds, device):
    import torch
    import torch.nn.functional as functional

    counts = torch.tensor([len(item[0]) for item in prepared], device=device)
    matrices = torch.stack([item[3] for item in prepared]).to(device)
    matrices = (matrices - means) / stds
    batch = torch.repeat_interleave(torch.arange(len(prepared), device=device), counts)
    coordinates = torch.cat([item[1] for item in prepared]).to(device)
    numbers = torch.cat([item[2] for item in prepared]).to(device)
    features = torch.cat([
        coordinates, functional.one_hot(numbers, 100).float(), matrices[batch],
    ], dim=-1)
    return model.encode(features, batch).detach().cpu().numpy()


def load_encoder(checkpoint_path, scaler_path, config_path, device):
    """Shared strict loader for reference and candidate exports."""
    import torch
    import yaml
    from components.jepa.frame.jepa import JEPA

    device = torch.device(device)
    packed = torch.load(scaler_path, map_location="cpu", weights_only=True)
    if isinstance(packed, (tuple, list)):
        packed = torch.stack(list(packed))
    if not isinstance(packed, torch.Tensor) or packed.shape != (2, 6):
        raise ValueError("Expected scaler tensor (2, 6) or a tuple of two six-element tensors")
    if not torch.isfinite(packed).all() or not (packed[1] > 0).all():
        raise ValueError("Scaler must be finite with strictly positive standard deviations")
    config_data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = SimpleNamespace(model=SimpleNamespace(**config_data["model"]), device=device)
    model = JEPA(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value
                           for key, value in checkpoint["model_state_dict"].items()}, strict=True)
    model.eval()
    means, stds = packed.float().to(device)
    return model, means, stds, config


def run(args):
    if args.out_dir.exists():
        raise FileExistsError(f"Use a new output directory: {args.out_dir}")

    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from pymatgen.core import Structure
    from components.jepa.frame.jepa import JEPA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but no CUDA device is available")

    train_path = args.train_csv or download_data_file(TRAIN_FILE, args.artifact_dir / "hf")
    table = load_train(train_path, args.limit)
    scaler_path, scaler_info = resolve_scaler(args)
    checkpoint_path = args.checkpoint or download_checkpoint(args.artifact_dir)
    model, means, stds, config = load_encoder(checkpoint_path, scaler_path, args.config, device)

    embeddings = np.empty((len(table), config.model.hidden_dim), dtype=np.float32)
    metadata = table.drop(columns=["cif"]).copy()
    metadata.insert(0, "source_row", np.arange(len(table)))
    metadata.insert(0, "embedding_row", np.arange(len(table)))
    formulas, systems, atom_counts = [], [], []
    with torch.inference_mode():
        for start in range(0, len(table), args.batch_size):
            stop = min(start + args.batch_size, len(table))
            prepared = []
            for row in range(start, stop):
                try:
                    item = prepare_structure(Structure.from_str(table.iloc[row].cif, fmt="cif"))
                    structure = item[0]
                    prepared.append(item)
                    formulas.append(structure.composition.reduced_formula)
                    systems.append(structure.composition.chemical_system)
                    atom_counts.append(len(structure))
                except Exception as exc:
                    raise ValueError(
                        f"Failed source row {row}, material_id={table.iloc[row].material_id}; "
                        "no rows were silently skipped"
                    ) from exc
            values = encode_batch(prepared, model, means, stds, device)
            if values.shape != (stop - start, config.model.hidden_dim) or not np.isfinite(values).all():
                raise ValueError(f"Invalid embeddings for source rows {start}:{stop}")
            # Check that padding other crystals in the batch does not change row 0.
            if start == 0:
                alone = encode_batch(prepared[:1], model, means, stds, device)
                np.testing.assert_allclose(values[:1], alone, rtol=1e-4, atol=1e-5)
            embeddings[start:stop] = values
            print(f"Embedded {stop:,}/{len(table):,} on {device}", flush=True)

    metadata["formula"] = formulas
    metadata["chemical_system"] = systems
    metadata["num_atoms"] = atom_counts
    if "dft_band_gap" in table:
        metadata["band_gap"] = table["dft_band_gap"]
    # Hash source files so candidate embeddings can later use the exact same inputs.
    provenance = {
        "train_csv": str(train_path.resolve()), "train_sha256": sha256(train_path),
        "dataset_source": "local override" if args.train_csv else "liun-online/Crys_JEPA",
        "release_revision": REVISION,
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "scaler": str(scaler_path.resolve()), "scaler_sha256": sha256(scaler_path),
        "scaler_info": scaler_info,
        "config_sha256": sha256(args.config), "rows": len(table),
        "embedding_shape": list(embeddings.shape), "batch_size": args.batch_size,
        "limit": args.limit, "device": str(device), "torch_version": str(torch.__version__),
        "preprocessing": PREPROCESSING,
        "pymatgen_version": package_version("pymatgen"),
        "encoder_source_sha256": sha256(ROOT / "components/jepa/frame/jepa.py"),
        "export_script_sha256": sha256(Path(__file__)),
        "alignment": "embeddings[i] == metadata.iloc[i]; embedding_row == source_row == input CSV position",
        "first_batch_singleton_check": "passed",
    }
    args.out_dir.mkdir(parents=True)
    np.save(args.out_dir / "embeddings.npy", embeddings, allow_pickle=False)
    metadata.to_csv(args.out_dir / "metadata.csv", index=False)
    saved = pd.read_csv(args.out_dir / "metadata.csv", keep_default_na=False)
    np.testing.assert_array_equal(saved.embedding_row, np.arange(len(table)))
    np.testing.assert_array_equal(saved.source_row, np.arange(len(table)))
    np.testing.assert_array_equal(saved.material_id.astype(str), table.material_id.astype(str))
    np.testing.assert_array_equal(
        np.load(args.out_dir / "embeddings.npy", allow_pickle=False), embeddings
    )
    provenance["embeddings_sha256"] = sha256(args.out_dir / "embeddings.npy")
    provenance["metadata_sha256"] = sha256(args.out_dir / "metadata.csv")
    provenance["saved_alignment_check"] = "passed"
    # The manifest is the completion marker, written only after output checks pass.
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {embeddings.shape} embeddings and aligned metadata in {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, help="Local override; default downloads the pinned HF MP-20 train CSV")
    parser.add_argument("--out-dir", type=Path, default=HERE / "outputs/mp20_train")
    parser.add_argument("--artifact-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jepa/mp.yml")
    parser.add_argument("--checkpoint", type=Path, help="Use a local checkpoint instead of downloading")
    parser.add_argument("--scaler", type=Path, help="Reuse an existing scaler instead of reconstructing one")
    parser.add_argument("--scaler-cap", type=int, default=20000, help="Crystals per JEPA pretraining file; 0 uses all rows")
    parser.add_argument("--scaler-seed", type=int, default=0)
    parser.add_argument("--download-only", action="store_true", help="Download the checkpoint without inference")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Embed the first N input rows for a smoke test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    if args.scaler_cap < 0 or not 0 <= args.scaler_seed < 2**32:
        parser.error("--scaler-cap must be nonnegative; --scaler-seed must be in [0, 2**32)")
    if args.download_only:
        print(download_checkpoint(args.artifact_dir))
    else:
        run(args)


if __name__ == "__main__":
    main()
