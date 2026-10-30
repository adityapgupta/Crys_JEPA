"""Isolated model adapters and model/structure-addressed resumable prediction cache."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from .common import database, fingerprint, read_json, write_json


def installed(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def download_model(config, name):
    from huggingface_hub import HfApi, snapshot_download
    spec = config["models"][name]
    directory = Path(spec["directory"])
    marker = directory / "download_receipt.json"
    if marker.exists():
        model_identity(config, name)  # Validate existing receipt; do not silently update weights.
        print(f"Verified existing {name} weights in {directory}")
        return
    revision = HfApi().model_info(spec["repository"], revision=spec.get("revision", "main")).sha
    patterns = [spec["filename"]] if name == "esen" else ["model.json", "model.pt", "state.pt"]
    snapshot_download(spec["repository"], revision=revision, local_dir=str(directory), allow_patterns=patterns)
    paths = [directory / pattern for pattern in patterns]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Incomplete downloaded {name} weights: {paths}")
    write_json(marker, {"repository": spec["repository"], "revision": revision,
                        "files": {p.name: sha256(p) for p in paths}})
    print(f"Pinned {name} to revision {revision}")


def model_identity(config, name):
    spec = config["models"][name]
    directory = Path(spec["directory"])
    source = read_json(directory / "download_receipt.json")
    if source["repository"] != spec["repository"]:
        raise ValueError("Downloaded weights belong to a different model repository")
    for filename, digest in source["files"].items():
        if sha256(directory / filename) != digest:
            raise ValueError(f"Changed model weights: {directory / filename}")
    packages = {p: installed(p) for p in ("torch", "numpy", "pymatgen", "pymatgen-core", "matgl",
                                         "fairchem-core", "ase", "torch-geometric", "e3nn", "lightning")}
    return {"name": name, "spec": spec, "weights": source, "packages": packages,
            "adapter_sha256": sha256(Path(__file__)), "geometry": "single point on stored geometry",
            "extra_energy_corrections": "none"}


def load_megnet_weights(directory):
    """Load verified MatGL weights with their serialized target normalizer.

    The published wrapper stores a Normalizer instance in model.pt. Keep
    weights_only=True in MatGL and allow this known class only during loading.
    """
    import matgl
    import torch
    from matgl.data.transformer import Normalizer

    with torch.serialization.safe_globals([Normalizer]):
        return matgl.load_model(directory)


class MEGNetAdapter:
    energy_kind = "formation_energy_per_atom"

    def __init__(self, spec, device):
        import torch
        from matgl.ext.pymatgen import Structure2Graph
        self.model = load_megnet_weights(spec["directory"])
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        base = getattr(self.model, "model", self.model)
        self.converter = Structure2Graph(element_types=base.element_types, cutoff=base.cutoff)

    def predict(self, structures):
        import torch
        from torch_geometric.data import Batch
        graphs, states = [], []
        for structure in structures:
            graph, lattice, state = self.converter.get_graph(structure)
            lattice = torch.as_tensor(lattice, dtype=torch.float32)
            graph.pbc_offshift = graph.pbc_offset @ lattice[0]
            graph.pos = graph.frac_coords @ lattice[0]
            graphs.append(graph)
            states.append(torch.as_tensor(state, dtype=torch.float32).reshape(-1))
        # The convenience predict_structure builds CPU graphs; move an explicit PyG
        # batch to the model device instead. Calling the wrapper preserves target inverse scaling.
        batch = Batch.from_data_list(graphs).to(self.device)
        state = torch.stack(states).to(self.device)
        with torch.inference_mode():
            return self.model(g=batch, state_attr=state).detach().cpu().numpy().reshape(-1)


class ESENAdapter:
    energy_kind = "total_energy_per_atom"

    def __init__(self, spec, device):
        from fairchem.core import OCPCalculator
        self.calculator = OCPCalculator(checkpoint_path=spec["checkpoint"], cpu=(device == "cpu"))

    def predict(self, structures):
        from pymatgen.io.ase import AseAtomsAdaptor
        values = []
        for structure in structures:
            atoms = AseAtomsAdaptor.get_atoms(structure)
            atoms.calc = self.calculator
            values.append(float(atoms.get_potential_energy()) / len(atoms))
        return np.array(values)


def predict_with_failures(adapter, structures):
    """Batch failures are retried individually so one bad graph does not lose other rows."""
    try:
        values = np.asarray(adapter.predict(structures), dtype=float)
        if values.shape != (len(structures),) or not np.isfinite(values).all():
            raise ValueError("Nonfinite prediction or invalid output shape")
        return [("ok", float(v), "") for v in values]
    except Exception:
        results = []
        for structure in structures:
            try:
                value = float(np.asarray(adapter.predict([structure])).reshape(-1)[0])
                if not np.isfinite(value):
                    raise ValueError("Nonfinite model prediction")
                results.append(("ok", value, ""))
            except Exception as exc:
                results.append(("prediction_error", None, f"{type(exc).__name__}:{exc}"))
        return results


def init_predictions(db):
    db.execute("""CREATE TABLE IF NOT EXISTS predictions (
        model_fingerprint TEXT, structure_hash TEXT, status TEXT,
        energy_per_atom REAL, error TEXT, seconds REAL,
        PRIMARY KEY(model_fingerprint, structure_hash))""")


def smoke_model(config, name):
    """Real-weight prerequisite: units, device, batching and cell-replication checks."""
    from pymatgen.core import Structure
    output = Path(config["output"])
    identity = model_identity(config, name)
    adapter = (MEGNetAdapter if name == "megnet" else ESENAdapter)(config["models"][name], config["device"])
    with database(output / "structures.sqlite") as db:
        records = db.execute("SELECT structure_json FROM structures WHERE kind='candidate' ORDER BY id LIMIT 4").fetchall()
    if not records:
        raise ValueError("Prepare candidates before the real-weight smoke test")
    structures = [Structure.from_dict(json.loads(row[0])) for row in records]
    chunk = np.asarray(adapter.predict(structures), dtype=float)
    alone = np.array([adapter.predict([s])[0] for s in structures], dtype=float)
    if chunk.shape != (len(structures),) or not np.isfinite(chunk).all():
        raise ValueError("Invalid smoke predictions")
    np.testing.assert_allclose(chunk, alone, rtol=1e-4, atol=1e-5)
    repeated = structures[0] * (2, 1, 1)
    repeat_energy = float(adapter.predict([repeated])[0])
    np.testing.assert_allclose(repeat_energy, alone[0], rtol=1e-4, atol=1e-3)
    # On CPU also compare the explicit PyG batch with the official convenience path.
    if name == "megnet" and config["device"] == "cpu":
        official = float(adapter.model.predict_structure(structures[0]).detach().reshape(-1)[0])
        np.testing.assert_allclose(official, alone[0], rtol=1e-4, atol=1e-5)
    write_json(output / f"{name}_smoke.json", {"fingerprint": fingerprint(identity), "passed": True,
        "energy_kind": adapter.energy_kind, "predictions_eV_per_atom": alone.tolist(),
        "replicated_cell_energy_per_atom": repeat_energy, "device": config["device"]})
    print(f"PASS {name}: real weights, finite per-atom energies, singleton equivalence and replicated-cell units")


def predict(config, name, max_items=None, retry_errors=False):
    from pymatgen.core import Structure
    output = Path(config["output"])
    cache = Path(config["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    identity = model_identity(config, name)
    token = fingerprint(identity)
    smoke_path = output / f"{name}_smoke.json"
    if not smoke_path.exists() or not read_json(smoke_path).get("passed") or read_json(smoke_path).get("fingerprint") != token:
        raise ValueError(f"Run smoke-model --model {name} in this exact environment before predict")
    write_json(output / f"{name}_model.json", {"fingerprint": token, **identity})
    selected = pd.read_csv(output / "evaluation_candidates.csv", keep_default_na=False)
    phases = pd.read_csv(output / "phase_manifest.csv", keep_default_na=False)
    required = set(selected.structure_hash) | set(phases.structure_hash)
    total = 0
    adapter = None
    with database(cache / f"{name}.sqlite") as predictions, database(output / "structures.sqlite") as structures:
        init_predictions(predictions)
        complete = {row[0] for row in predictions.execute(
            "SELECT structure_hash FROM predictions WHERE model_fingerprint=?" + (" AND status='ok'" if retry_errors else ""), (token,))}
        pending = required - complete
        batch, hashes = [], []

        def flush():
            nonlocal total, adapter
            if not batch:
                return
            if adapter is None:
                adapter = (MEGNetAdapter if name == "megnet" else ESENAdapter)(config["models"][name], config["device"])
            start = time.monotonic()
            results = predict_with_failures(adapter, batch)
            # Check a supported row without aborting on an unsupported first crystal.
            if not hasattr(adapter, "singleton_check"):
                for structure, (status, value, _) in zip(batch, results):
                    if status == "ok":
                        alone = float(np.asarray(adapter.predict([structure])).reshape(-1)[0])
                        np.testing.assert_allclose(value, alone, rtol=1e-4, atol=1e-5)
                        adapter.singleton_check = True
                        break
            elapsed = (time.monotonic() - start) / len(batch)
            for digest, (status, energy, error) in zip(hashes, results):
                predictions.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?)", (token, digest, status, energy, error, elapsed))
            predictions.commit()
            total += len(batch)
            print(f"{name}: committed {total} new predictions; {len(pending)} still pending", flush=True)
            batch.clear(); hashes.clear()

        for record in structures.execute("SELECT * FROM structures ORDER BY kind,id"):
            digest = record["structure_hash"]
            if digest not in pending:
                continue
            if max_items is not None and total + len(batch) >= max_items:
                break
            pending.remove(digest)
            try:
                raw = json.loads(record["structure_json"])
                if fingerprint(raw) != digest:
                    raise ValueError("Stored structure checksum mismatch")
                structure = Structure.from_dict(raw)
                if not structure.is_ordered or not len(structure):
                    raise ValueError("Unsupported reference structure")
            except Exception as exc:
                predictions.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?)",
                                    (token, digest, "structure_error", None, str(exc), 0.))
                predictions.commit()
                total += 1
                continue
            batch.append(structure); hashes.append(digest)
            if len(batch) >= config["prediction_batch_size"]:
                flush()
        flush()
        rows = {row["structure_hash"]: dict(row) for row in predictions.execute(
            "SELECT * FROM predictions WHERE model_fingerprint=?", (token,))}
        missing = required - set(rows)
        errors = sum(rows[h]["status"] != "ok" for h in required if h in rows)
        if required and not missing and errors == len(required):
            examples = [rows[h]["error"] for h in sorted(required)][:3]
            raise RuntimeError(f"Every {name} prediction failed. Check the model environment/smoke test. Examples: {examples}")
        # Lightweight exports support inspection without reading the cache database.
        export_hashes = {}
        for kind, table in (("candidate", selected), ("phase", phases)):
            records = [{"id": row.id, "structure_hash": row.structure_hash,
                        **rows.get(row.structure_hash, {"status": "pending", "energy_per_atom": None, "error": ""})}
                       for row in table.itertuples()]
            path = output / f"{name}_{kind}_predictions.csv"
            temporary = path.with_suffix(".csv.part")
            pd.DataFrame(records).to_csv(temporary, index=False)
            temporary.replace(path)
            export_hashes[kind] = sha256(path)
        # Completion marker LAST: interrupted exports cannot masquerade as complete.
        write_json(output / f"{name}_prediction_status.json", {
            "fingerprint": token, "required_structures": len(required), "pending": len(missing),
            "errors": errors, "complete": not missing, "new_predictions": total,
            "phase_manifest_sha256": sha256(output / "phase_manifest.csv"),
            "evaluation_manifest_sha256": sha256(output / "evaluation_candidates.csv"),
            "prediction_export_sha256": export_hashes})
    if missing:
        print(f"{name}: {len(missing)} predictions remain. Re-submit the same stage to resume.", flush=True)
        return False
    return True
