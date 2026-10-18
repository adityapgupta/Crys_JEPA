"""Evaluate the complete label-free baseline grid on validation only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reference_baselines.core.io import sha256
from reference_baselines.core.settings import baseline_settings
from reference_baselines.evaluation.metrics import load_adjacency, unique_order
from reference_baselines.paths import (
    BASELINE_RESULTS_DIR,
    CONFIG_DIR,
    ROOT,
    RESULTS_DIR,
)


def family(name):
    if name.startswith("theta1_gate_"):
        return "angle gate control"
    if name.startswith("conductance_"):
        return "conductance reference sweep"
    if name.startswith("random_") and "_seed" in name:
        return "matched random reference control"
    if name.startswith("strict_"):
        return "strict-stable bank"
    if name.startswith("spectral_"):
        return "spectral retrieval"
    if "_resolvent_" in name:
        return "Laplacian resolvent"
    if "_heat_" in name:
        return "Laplacian heat"
    if name.startswith("cosine_theta"):
        return "cosine angle cutoff"
    if name.startswith("cosine_"):
        return "cosine retrieval"
    return "original grid"


def paired_interval(selected, original, metadata, targets, seed, draws):
    a = (
        pd.DataFrame(
            {
                "system": metadata.loc[original, "chemical_system"].to_numpy(),
                "target": targets[original],
            }
        )
        .groupby("system")
        .target.agg(["sum", "count"])
    )
    b = (
        pd.DataFrame(
            {
                "system": metadata.loc[selected, "chemical_system"].to_numpy(),
                "target": targets[selected],
            }
        )
        .groupby("system")
        .target.agg(["sum", "count"])
    )
    systems = a.index.union(b.index)
    aa = a.reindex(systems, fill_value=0).to_numpy()
    bb = b.reindex(systems, fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(systems), (draws, len(systems)))
    x = aa[index].sum(1)
    y = bb[index].sum(1)
    good = (x[:, 1] > 0) & (y[:, 1] > 0)
    delta = y[good, 0] / y[good, 1] - x[good, 0] / x[good, 1]
    return np.quantile(delta, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-spectral", action="store_true")
    args = parser.parse_args()
    cfg = baseline_settings()
    out = BASELINE_RESULTS_DIR
    inputs = [out / "validation_angular_scores.npz"] + [
        out / f"validation_diffusion_{m}.npz" for m in ["euclidean", "cosine"]
    ]
    inputs.append(out / "validation_random_controls.npz")
    inputs.append(out / "validation_conductance.npz")
    for metric in ["euclidean", "cosine"]:
        path = out / f"validation_spectral_{metric}.npz"
        status = out / f"{metric}_eigenmodes.json"
        accepted = (
            status.exists() and json.loads(status.read_text())["status"] == "passed"
        )
        if path.exists() and accepted:
            inputs.append(path)
        elif args.require_spectral:
            raise FileNotFoundError(path)
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv").set_index(
        "candidate_row"
    )
    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv").set_index(
        "candidate_row"
    )
    rows = np.load(inputs[0])["candidate_rows"]
    assert (
        manifest.loc[rows, "split"].eq("validation").all()
        and manifest.loc[rows, "vun"].all()
    )
    assert labels.loc[rows, "hull_status"].eq("ok").all()
    matrices = []
    names = []
    for path in inputs:
        source = np.load(path)
        np.testing.assert_array_equal(source["candidate_rows"], rows)
        names.extend(source["methods"].tolist())
        matrices.append(source["scores"])
    assert len(set(names)) == len(names)
    matrix = np.column_stack(matrices)
    assert not np.isinf(matrix).any()
    hull = labels.e_above_hull_proxy.to_numpy(dtype=float)
    targets = (hull <= 0.1).astype(int)
    adjacency = load_adjacency()
    records = []
    selected_by_method = {}
    per_turn = []
    extreme_sensitivity = []
    base_rate = float(targets[rows].mean())
    for j, name in enumerate(names):
        score = matrix[:, j]
        available = np.isfinite(score)
        ranked = rows[available][np.lexsort((rows[available], score[available]))]
        unique = unique_order(ranked, adjacency)
        for lower_bound in [-1.0, -0.5, -0.25]:
            retained = rows[hull[rows] >= lower_bound]
            budget = int(len(retained) * 0.05)
            robust = unique_order(ranked[hull[ranked] >= lower_bound], adjacency)[
                :budget
            ]
            extreme_sensitivity.append(
                {
                    "method": name,
                    "lower_bound_eV_per_atom": lower_bound,
                    "excluded_rows": len(rows) - len(retained),
                    "budget": budget,
                    "unique_yield_per_slot": float(targets[robust].sum() / budget),
                }
            )
        correlation = float(
            spearmanr(score[available], hull[rows[available]]).statistic
        )
        for fraction in cfg["selection_fractions"]:
            budget = int(len(rows) * fraction)
            selected = unique[:budget]
            raw = ranked[:budget]
            norefill = unique_order(raw, adjacency)
            for threshold in cfg["stability_thresholds"]:
                passed = int((hull[selected] <= threshold).sum())
                records.append(
                    {
                        "method": name,
                        "family": family(name),
                        "selection_fraction": fraction,
                        "threshold_eV_per_atom": threshold,
                        "budget": budget,
                        "selected_unique": len(selected),
                        "score_coverage": float(available.mean()),
                        "near_stable_unique_count": passed,
                        "unique_yield_per_slot": passed / budget,
                        "precision_selected": passed / len(selected)
                        if len(selected)
                        else np.nan,
                        "raw_positive_fraction": float((hull[raw] <= threshold).mean())
                        if len(raw)
                        else np.nan,
                        "no_refill_unique_yield": float(
                            (hull[norefill] <= threshold).sum() / budget
                        ),
                        "chemical_systems_selected": manifest.loc[
                            selected, "chemical_system"
                        ].nunique(),
                        "score_hull_spearman": correlation,
                    }
                )
            if fraction == 0.05:
                selected_by_method[name] = selected
        # Original turn grouping, with the same fixed unique budget inside each turn.
        for turn in range(40):
            turn_rows = rows[
                (manifest.loc[rows, "eval_folder"].to_numpy() // 10) == turn
            ]
            budget = int(len(turn_rows) * 0.05)
            order = ranked[np.isin(ranked, turn_rows)]
            chosen = unique_order(order, adjacency)[:budget]
            per_turn.append(
                {
                    "method": name,
                    "turn": turn,
                    "budget": budget,
                    "selected": len(chosen),
                    "near_stable": int(targets[chosen].sum()),
                }
            )
        if (j + 1) % 50 == 0:
            print("Evaluated methods:", j + 1, "/", len(names), flush=True)
    result = pd.DataFrame(records)
    result.to_csv(out / "validation_metrics.csv", index=False)
    pd.DataFrame(per_turn).to_csv(out / "validation_turn_metrics.csv", index=False)
    pd.DataFrame(extreme_sensitivity).to_csv(
        out / "validation_extreme_sensitivity.csv", index=False
    )
    primary = result[
        result.selection_fraction.eq(0.05) & result.threshold_eV_per_atom.eq(0.1)
    ]
    # Fixed controls plus one exploratory best setting per family; never describe these as unseen-test winners.
    chosen = [
        "crys_jepa",
        "intersection_fallback_k5",
        "knn_k10",
        "union_k10",
        "cosine_knn_raw_k5",
        "cosine_knn_cos_k5",
        "cosine_intersection_fallback_k5",
        "strict_chemical_mean",
        "strict_euclidean_intersection_fallback_k5",
        "strict_cosine_intersection_fallback_k5",
    ]
    best = (
        primary.sort_values(
            ["unique_yield_per_slot", "method"], ascending=[False, True]
        )
        .groupby("family", sort=False)
        .head(1)
    )
    chosen = list(dict.fromkeys(chosen + best.method.tolist()))
    original = selected_by_method["crys_jepa"]
    intervals = []
    for method in chosen:
        selected = selected_by_method[method]
        interval = paired_interval(
            selected, original, manifest, targets, cfg["seed"], cfg["bootstrap_draws"]
        )
        intervals.append(
            {
                "method": method,
                "selection_fraction": 0.05,
                "selection_status": "exploratory family best"
                if method in best.method.to_list()
                else "fixed control",
                "difference_vs_crys_jepa": float(
                    targets[selected].mean() - targets[original].mean()
                ),
                "paired_cluster_interval": json.dumps(interval),
                "multiple_comparisons_adjusted": False,
            }
        )
    pd.DataFrame(intervals).to_csv(out / "validation_paired_intervals.csv", index=False)
    # Reuse the already validated random-selection benchmark under the identical uniqueness rule.
    random = pd.read_csv(RESULTS_DIR / "validation_unique_baseline_metrics.csv")
    random = random[random.method.eq("random_candidates")]
    random.to_csv(out / "random_control.csv", index=False)
    report = {
        "status": "passed",
        "candidate_count": len(rows),
        "method_count": len(names),
        "base_rate": base_rate,
        "split": "validation only",
        "primary": "unique yield at 5% of original validation count, 0.1 eV/atom proxy cutoff, with refill",
        "coverage": "NaN abstains; selected_unique and yield per requested slot expose budget shortfalls",
        "test_performance_evaluated": False,
        "neural_models_trained": False,
        "reference_source": "known MP20 training bank; strict bank from recorded reference e_above_hull, not candidate labels",
        "limits": [
            "MatterSim proxy labels, no independent DFT",
            "Many settings on one validation split; family bests are exploratory",
            "Intervals are paired fixed-selection chemical-system bootstraps, unadjusted for multiple comparisons",
        ],
        "sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in inputs
            + [CONFIG_DIR / "nonparametric.toml", Path(__file__).resolve()]
        },
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        primary.sort_values("unique_yield_per_slot", ascending=False)
        .head(12)[["method", "unique_yield_per_slot", "score_coverage"]]
        .to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
