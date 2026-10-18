"""Cosine distance and angle cutoffs with explicit zero-vector handling."""

import numpy as np


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms == 0):
        raise ValueError("Cosine search needs finite, nonzero vectors.")
    return values / norms


def within_angle(cosine_distance, degrees):
    if not 0 <= degrees <= 180:
        raise ValueError("Angle must be in [0,180] degrees.")
    return cosine_distance <= (1 - np.cos(np.deg2rad(degrees))) + 1e-14


def raw_distances(query, references):
    return np.square(np.asarray(references, dtype=np.float64) - query).sum(axis=-1)
