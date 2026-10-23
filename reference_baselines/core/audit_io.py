"""Read audit tables without treating the chemical formula NaN as missing."""

import pandas as pd


def read_csv(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kwargs)
