"""Aggregate selections made separately in each batch, never rerank candidates."""

import numpy as np
import pandas as pd


REQUEST_KEYS = ["method", "policy", "fraction"]


def aggregate_batch_metrics(rows):
    if rows.empty or not rows.scope.eq("batch").all():
        raise ValueError("Need per-batch metrics, not pooled/turn selections")
    keys = REQUEST_KEYS + ["proxy", "threshold_eV_per_atom"]
    if rows.duplicated(keys + ["group"]).any():
        raise ValueError("Repeated batch metric")
    np.testing.assert_array_equal(rows.budget, np.floor(rows.fraction * rows.population))
    count_columns = ["population", "budget", "selected_count", "evaluated_count", "unknown_count", "stable_count",
                     "scoreable_count", "raw_unique_without_refill", "raw_selected_count",
                     "exact_threshold_count", "below_minus_0_5_count"]
    counts = [name for name in count_columns if name in rows]
    grouped = rows.groupby(keys, sort=True)
    result = grouped[counts].sum()
    result["batch_count"] = grouped["group"].nunique()
    result["precision_evaluated"] = result.stable_count / result.evaluated_count.replace(0, np.nan)
    result["valid_label_coverage"] = result.evaluated_count / result.selected_count.replace(0, np.nan)
    result["stable_yield_per_requested_slot"] = result.stable_count / result.budget.replace(0, np.nan)
    result["stable_yield_lower_bound"] = result.stable_yield_per_requested_slot
    result["stable_yield_upper_bound"] = (result.stable_count + result.unknown_count) / result.budget.replace(0, np.nan)
    if {"raw_unique_without_refill", "raw_selected_count"}.issubset(result):
        result["duplicate_fraction_raw_selection"] = 1 - result.raw_unique_without_refill / result.raw_selected_count.replace(0, np.nan)
    if "mean_hull_distance" in rows:
        weighted = rows.assign(_total=pd.to_numeric(rows.mean_hull_distance, errors="coerce") * rows.evaluated_count)
        result["mean_hull_distance"] = weighted.groupby(keys)._total.sum() / result.evaluated_count.replace(0, np.nan)
    # Medians, quantiles and counts of distinct chemical systems cannot be summed.
    result = result.reset_index()
    result["threshold_rule"] = np.where(result.threshold_eV_per_atom.eq(0), "<=", "<")
    result["scope"], result["group"] = "batch_aggregate", -1
    return result


def aggregate_batch_consensus(rows):
    if rows.empty or not rows.scope.eq("batch").all():
        raise ValueError("Need per-batch consensus")
    if rows.duplicated(REQUEST_KEYS + ["group"]).any():
        raise ValueError("Repeated batch consensus")
    columns = ["population", "budget", "selected_count", "both_evaluated", "unknown_either",
               "both_pass", "neither_pass", "megnet_only_pass", "esen_only_pass"]
    columns = [c for c in columns if c in rows]
    grouped = rows.groupby(REQUEST_KEYS, sort=True)
    result = grouped[columns].sum()
    result["batch_count"] = grouped["group"].nunique()
    result["agreement_fraction"] = (result.both_pass + result.neither_pass) / result.both_evaluated.replace(0, np.nan)
    result = result.reset_index()
    result["scope"], result["group"] = "batch_aggregate", -1
    return result


def paired_batch_interval(baseline, alternative, repeats=2000, seed=2026):
    """Descriptive paired resampling of batch counts, distinct from chemical-system CIs."""
    a = baseline.set_index("group", verify_integrity=True).sort_index()
    b = alternative.set_index("group", verify_integrity=True).sort_index()
    if not a.index.equals(b.index):
        raise ValueError("Baseline and alternative must include the same batches")
    if len(a) < 2:
        return np.nan, np.nan
    columns = ["stable_count", "evaluated_count"]
    a, b = a[columns].to_numpy(dtype=float), b[columns].to_numpy(dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(a), size=(repeats, len(a)))
    aa, bb = a[draws].sum(axis=1), b[draws].sum(axis=1)
    valid = (aa[:, 1] > 0) & (bb[:, 1] > 0)
    if not valid.any():
        return np.nan, np.nan
    return tuple(np.quantile(bb[valid, 0]/bb[valid, 1] - aa[valid, 0]/aa[valid, 1], [.025, .975]))
