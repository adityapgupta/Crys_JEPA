"""Portable configuration: all relative paths are relative to the repository."""

from copy import deepcopy
from pathlib import Path
import yaml

from reference_baselines.paths import ROOT


PATH_KEYS = ("release_root", "reference_export", "checkpoint", "scaler", "jepa_config",
             "hull_database", "output", "cache")


def load_config(path, start=None, end=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        p = Path(config[key]).expanduser()
        config[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    if start is not None:
        config["start_batch"] = start
    if end is not None:
        config["end_batch"] = end
    if not 0 <= config["start_batch"] <= config["end_batch"]:
        raise ValueError("Need 0 <= start_batch <= end_batch (inclusive)")
    if not config["neighbours"] or any(int(k) != k or k < 1 for k in config["neighbours"]):
        raise ValueError("neighbours must contain positive integers")
    config["neighbours"] = sorted(set(config["neighbours"]))
    if not config["fractions"] or any(not 0 < p <= 1 for p in config["fractions"]):
        raise ValueError("fractions must be in (0,1]")
    if any(config[k] < 1 for k in ("embedding_batch_size", "retrieval_batch_size", "prediction_batch_size", "bootstrap_repeats")):
        raise ValueError("Batch sizes and bootstrap_repeats must be positive")
    config["fractions"] = sorted(set(config["fractions"]))
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if set(config["models"]) != {"megnet", "esen"}:
        raise ValueError("Configure both megnet and esen for the cross-proxy report")
    for model, spec in config["models"].items():
        for key in ("directory", "checkpoint"):
            if key in spec:
                p = Path(spec[key]).expanduser()
                spec[key] = str((ROOT / p).resolve() if not p.is_absolute() else p.resolve())
    return config


def run_identity(config):
    result = deepcopy(config)
    # Device and throughput settings may change when resuming on another node.
    for key in ("device", "embedding_batch_size", "retrieval_batch_size", "prediction_batch_size"):
        result.pop(key, None)
    return result
