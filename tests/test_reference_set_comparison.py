"""Offline checks for input order and checkpoint download integrity."""

import io
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from reference_baselines.reference_set_comparison import generate_embeddings as export


class ReferenceExportTests(unittest.TestCase):
    def test_input_order_is_csv_position_not_id_or_saved_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({
                "Unnamed: 0": [8, 2, 19],
                "material_id": ["mp-z", "mp-a", "mp-m"],
                "cif": ["cif-z", "cif-a", "cif-m"],
                "pretty_formula": ["NaN", "Fe", "O2"],
            }).to_csv(path, index=False)
            table = export.load_train(path)
            self.assertEqual(table.material_id.tolist(), ["mp-z", "mp-a", "mp-m"])
            self.assertEqual(table.pretty_formula.iloc[0], "NaN")
            self.assertEqual(table["Unnamed: 0"].tolist(), [8, 2, 19])
            self.assertEqual(export.load_train(path, 2).material_id.tolist(), ["mp-z", "mp-a"])

    def test_duplicate_materials_fail_instead_of_deduplicating(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            pd.DataFrame({"material_id": ["a", "a"], "cif": ["x", "y"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                export.load_train(path)

    def test_only_checkpoint_extracted_and_cache_checked(self):
        payload = b"test checkpoint bytes"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(export.CHECKPOINT_MEMBER, payload)
            archive.writestr("exp_logs/unrelated/file.txt", b"do not extract")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "artifacts"
            with patch.object(export, "RangeFile", return_value=io.BytesIO(buffer.getvalue())):
                checkpoint = export.download_checkpoint(destination)
            self.assertEqual(checkpoint.read_bytes(), payload)
            self.assertFalse((destination / "exp_logs").exists())
            self.assertFalse(checkpoint.with_suffix(".pt.part").exists())
            receipt = json.loads((destination / "checkpoint_source.json").read_text())
            self.assertEqual(receipt["revision"], export.REVISION)
            with patch.object(export, "RangeFile", side_effect=AssertionError("Unexpected network access")):
                self.assertEqual(export.download_checkpoint(destination), checkpoint)
                checkpoint.write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "receipt"):
                    export.download_checkpoint(destination)

    def test_dataset_download_checks_published_hash_and_cache(self):
        payload = b"fixture compressed dataset"
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(export.DATA_HASHES, {export.TRAIN_FILE: hashlib.sha256(payload).hexdigest()}):
                with patch.object(export.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                    path = export.download_data_file(export.TRAIN_FILE, Path(directory))
                self.assertEqual(path.read_bytes(), payload)
                with patch.object(export.urllib.request, "urlopen", side_effect=AssertionError("Network")):
                    self.assertEqual(export.download_data_file(export.TRAIN_FILE, Path(directory)), path)
                    path.write_bytes(b"bad cache")
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        export.download_data_file(export.TRAIN_FILE, Path(directory))

    def test_bad_dataset_download_leaves_no_final_or_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(export.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    export.download_data_file(export.TRAIN_FILE, Path(directory))
            self.assertEqual(list(Path(directory).rglob("*.gz")), [])
            self.assertEqual(list(Path(directory).rglob("*.part")), [])


INFERENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None
                          for name in ("torch", "torch_geometric", "pymatgen", "yaml"))


@unittest.skipUnless(INFERENCE_AVAILABLE, "Requires the inference dependencies")
class SmallInferenceTests(unittest.TestCase):
    def test_scaler_reconstruction_and_row_aligned_export_with_small_jepa(self):
        """Exercise the real encoder on tiny fixtures, without network or open weights."""
        import argparse
        import numpy as np
        import torch
        from types import SimpleNamespace
        from pymatgen.core import Lattice, Structure
        from components.jepa.frame.jepa import JEPA

        structures = []
        for i in range(1, 9):
            lattice = Lattice.from_parameters(4 + .13*i, 5 + .17*i, 6 + .11*i,
                                             70 + 1.1*i, 75 + .9*i, 80 + 1.3*i)
            species = ["Na"] if i == 1 else ["Na", "Cl"]
            coordinates = [[0, 0, 0]] if i == 1 else [[0, 0, 0], [.4, .5, .6]]
            structures.append(Structure(lattice, species, coordinates))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = {}
            for name, offset in zip(export.SCALER_FILES, (0, 2)):
                frame = pd.DataFrame({"cif": [s.to(fmt="cif") for s in structures[offset:offset+6]]})
                path = root / Path(name).name
                frame.to_csv(path, index=False, compression="gzip")
                frames[name] = path
            # Deliberately unsorted IDs and unrelated saved source indices.
            train = pd.DataFrame({"material_id": ["mp-z", "mp-a", "mp-m"],
                                  "Unnamed: 0": [9, 2, 17],
                                  "cif": [s.to(fmt="cif") for s in structures[:3]],
                                  "pretty_formula": ["NaN", "NaCl", "NaCl"]})
            train_path = root / "train.csv"
            train.to_csv(train_path, index=False)
            frames[export.TRAIN_FILE] = train_path
            model_config = {"hidden_dim": 16, "layers": 1, "attn_head": 4, "dropout": 0.0}
            config_path = root / "config.yml"
            config_path.write_text(json.dumps({"model": model_config}))
            torch.manual_seed(0)
            model = JEPA(SimpleNamespace(model=SimpleNamespace(**model_config), device="cpu"))
            checkpoint = root / "checkpoint.pt"
            torch.save({"model_state_dict": {"module."+k: v for k, v in model.state_dict().items()}}, checkpoint)
            args = argparse.Namespace(scaler=None, artifact_dir=root / "artifacts", scaler_cap=3,
                                      scaler_seed=0, train_csv=None, out_dir=root / "out",
                                      config=config_path, checkpoint=checkpoint, limit=None,
                                      batch_size=2, device="cpu")
            with patch.object(export, "download_data_file", side_effect=lambda name, _: frames[name]):
                export.run(args)
            vectors = np.load(args.out_dir / "embeddings.npy")
            metadata = pd.read_csv(args.out_dir / "metadata.csv", keep_default_na=False)
            provenance = json.loads((args.out_dir / "provenance.json").read_text())
            self.assertEqual(vectors.shape, (3, 16))
            self.assertTrue(np.isfinite(vectors).all())
            self.assertEqual(metadata.material_id.tolist(), train.material_id.tolist())
            self.assertEqual(metadata.embedding_row.tolist(), [0, 1, 2])
            self.assertEqual(metadata.source_row.tolist(), [0, 1, 2])
            self.assertEqual(metadata["Unnamed: 0"].tolist(), [9, 2, 17])
            self.assertEqual(metadata.pretty_formula.iloc[0], "NaN")
            self.assertEqual(provenance["scaler_info"]["mode"], "reconstructed_approximate")
            self.assertEqual(provenance["scaler_info"]["used_rows"], 6)
            with patch.object(export, "download_data_file", side_effect=AssertionError("Unexpected download")):
                cached_path, _ = export.resolve_scaler(args)
            self.assertEqual(str(cached_path.resolve()), provenance["scaler"])
            means, stds = torch.load(cached_path, weights_only=True)
            model.eval()
            # Compare each saved row against a separate single-crystal encode.
            with torch.inference_mode():
                for i, structure in enumerate(structures[:3]):
                    expected = export.encode_batch([export.prepare_structure(structure)], model, means, stds, "cpu")
                    np.testing.assert_allclose(vectors[i:i+1], expected, rtol=1e-4, atol=1e-5)
            # The supplied script's tuple format is also accepted.
            tuple_scaler = root / "tuple_scaler.pt"
            torch.save((means, stds), tuple_scaler)
            args.scaler = tuple_scaler
            args.train_csv = train_path
            args.out_dir = root / "out_tuple"
            export.run(args)
            np.testing.assert_array_equal(np.load(args.out_dir / "embeddings.npy"), vectors)


if __name__ == "__main__":
    unittest.main()
