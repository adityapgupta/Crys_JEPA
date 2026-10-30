"""Resume candidate export only in the verified reference embedding space."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT
from .common import database, fingerprint, finish, read_json, receipt, write_json
from .generate_embeddings import PREPROCESSING, encode_batch, load_encoder, prepare_structure


def verify_reference(config):
    directory = Path(config["reference_export"])
    source = read_json(directory / "provenance.json")
    for filename, key in (("embeddings.npy", "embeddings_sha256"), ("metadata.csv", "metadata_sha256")):
        if sha256(directory / filename) != source[key]:
            raise ValueError(f"Reference export checksum mismatch: {filename}")
    for key in ("checkpoint", "scaler"):
        if sha256(Path(config[key])) != source[key + "_sha256"]:
            raise ValueError(f"Reference/candidate {key} mismatch")
    if (sha256(Path(config["jepa_config"])) != source["config_sha256"]
            or source["preprocessing"] != PREPROCESSING
            or sha256(ROOT / "components/jepa/frame/jepa.py") != source["encoder_source_sha256"]):
        raise ValueError("Reference encoder/config/preprocessing differs from candidate encoder")
    return source


def embed_candidates(config):
    import torch
    from pymatgen.core import Structure

    source = verify_reference(config)
    output = Path(config["output"])
    manifest = pd.read_csv(output / "candidates.csv", keep_default_na=False)
    manifest = manifest[manifest.eligible].reset_index(drop=True)
    np.testing.assert_array_equal(manifest.embedding_row, np.arange(len(manifest)))
    identity = fingerprint({"candidates": sha256(output / "candidates.csv"), "reference": source,
                            "export_source": sha256(Path(__file__)), "preprocessing_source": sha256(Path(prepare_structure.__code__.co_filename))})
    marker = output / "embed.json"
    if receipt(marker, identity):
        return
    progress = output / "embed_progress.json"
    path = output / "candidate_embeddings.npy"
    count = 0
    if progress.exists():
        state = read_json(progress)
        if state["identity"] != identity:
            raise ValueError("Candidate embedding checkpoint is stale")
        count = state["completed_rows"]
    model, means, stds, model_config = load_encoder(config["checkpoint"], config["scaler"], config["jepa_config"], config["device"])
    values = np.lib.format.open_memmap(path, mode="r+" if count else "w+", dtype=np.float32,
                                       shape=(len(manifest), model_config.model.hidden_dim))
    if values.shape != (len(manifest), model_config.model.hidden_dim) or not np.isfinite(values[:count]).all():
        raise ValueError("Invalid embedding checkpoint")
    with database(output / "structures.sqlite") as db, torch.inference_mode():
        for start in range(count, len(manifest), config["embedding_batch_size"]):
            stop = min(start + config["embedding_batch_size"], len(manifest))
            prepared = []
            for row in manifest.iloc[start:stop].itertuples():
                stored = db.execute("SELECT * FROM structures WHERE id=?", (row.id,)).fetchone()
                if stored is None or stored["structure_hash"] != row.structure_hash:
                    raise ValueError(f"Modified/missing source structure: {row.id}")
                prepared.append(prepare_structure(Structure.from_dict(json.loads(stored["structure_json"]))))
            batch = encode_batch(prepared, model, means, stds, config["device"])
            singleton = encode_batch(prepared[:1], model, means, stds, config["device"])
            np.testing.assert_allclose(batch[:1], singleton, rtol=1e-4, atol=1e-5)
            if not np.isfinite(batch).all() or batch.shape != (stop-start, values.shape[1]):
                raise ValueError(f"Invalid candidate embeddings at rows {start}:{stop}")
            values[start:stop] = batch
            values.flush()
            write_json(progress, {"identity": identity, "completed_rows": stop})
            print(f"Embedded {stop}/{len(manifest)}", flush=True)
    meta = output / "candidate_metadata.csv"
    manifest.to_csv(meta, index=False)
    finish(marker, identity, [path, meta], reference_checkpoint_sha256=source["checkpoint_sha256"],
           scaler_sha256=source["scaler_sha256"], preprocessing=PREPROCESSING,
           pymatgen_version=__import__("importlib.metadata", fromlist=["version"]).version("pymatgen"))
