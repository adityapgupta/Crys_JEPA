"""Shared greedy uniqueness rule used by fixed baselines and learned models."""

import numpy as np
import pandas as pd

from reference_baselines.paths import RESULTS_DIR


def unique_order(rows, adjacency):
    kept, order = set(), []
    for row in rows:
        row = int(row)
        if not adjacency.get(row, set()).intersection(kept):
            kept.add(row); order.append(row)
    return np.array(order, dtype=np.int64)


def load_adjacency():
    pairs = pd.read_csv(RESULTS_DIR / 'validation_duplicate_pairs.csv')
    result = {}
    for a, b in pairs.itertuples(index=False, name=None):
        result.setdefault(a,set()).add(b); result.setdefault(b,set()).add(a)
    return result
