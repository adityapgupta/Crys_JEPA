"""Load the fixed non-neural experiment grid."""

import tomllib
from reference_baselines.paths import CONFIG_DIR


def baseline_settings():
    return tomllib.loads((CONFIG_DIR / "nonparametric.toml").read_text())
