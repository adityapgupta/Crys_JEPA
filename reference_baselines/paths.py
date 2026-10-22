"""Repository paths shared by every experiment, independent of the working directory."""

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
RESULTS_DIR = PACKAGE_DIR / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "nonparametric"
CONFIG_DIR = PACKAGE_DIR / "configs"
DOCS_DIR = PACKAGE_DIR / "docs"

SOURCE_PACKAGES = (
    "core",
    "datasets",
    "graphs",
    "baselines",
    "structure",
    "evaluation",
    "models",
    "training",
    "checks",
    "reporting",
    "embedding_structure",
)
