"""Freeze, score, audit, and evaluate one fixed held-out comparison.

Run each phase in order. No grid search or test-based winner selection is used.
"""

import argparse
from datetime import datetime, timezone
import json
import tomllib

import numpy as np
import pandas as pd

from reference_baselines.baselines.heldout import fixed_scores
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.paths import (
    BASELINE_RESULTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DOCS_DIR,
    PACKAGE_DIR,
    ROOT,
    RESULTS_DIR,
)
from reference_baselines.structure.heldout import audit


OUT = RESULTS_DIR / "heldout"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def input_paths():
    paths = list(PACKAGE_DIR.rglob("*.py"))
    paths = [
        p for p in paths if "environment" not in p.parts and "results" not in p.parts
    ]
    paths += list((ROOT / "tests").glob("test_*.py"))
    paths += list(
        (DATA_DIR / "release/exp_logs/base/mp_20/eval").glob("*/relaxed_mattersim.json")
    )
    paths += [DATA_DIR / "release/archive_manifest.json"]
    paths += [CONFIG_DIR / "heldout.toml", CONFIG_DIR / "nonparametric.toml"]
    paths += [
        ROOT / "embeddings/data" / name
        for name in [
            "embeddings.npy",
            "metadata.csv",
            "candidate_embeddings.npy",
            "candidate_metadata.csv",
        ]
    ]
    paths += [
        RESULTS_DIR / name
        for name in [
            "split_manifest.csv",
            "candidate_reference_scores.csv",
            "candidate_labels.csv",
            "candidate_entries.csv",
            "chemical_reference_indptr.npy",
            "chemical_reference_indices.npy",
            "candidate_knn_indices.npy",
            "candidate_cosine_indices.npy",
            "candidate_cosine_distances.npy",
            "reference_graph_k10.npz",
            "reference_cosine_graph_k10.npz",
            "candidate_label_provenance.json",
        ]
    ]
    paths += [DATA_DIR / "hull/source.json", DATA_DIR / "hull/mp.lmdb"]
    paths += [
        BASELINE_RESULTS_DIR / name
        for name in [
            "validation_angular_scores.npz",
            "validation_random_controls.npz",
            "validation_diffusion_euclidean.npz",
            "validation_diffusion_cosine.npz",
            "validation_metrics.csv",
        ]
    ]
    return sorted(set(paths))


def manifest_rows():
    metadata = pd.read_csv(RESULTS_DIR / "split_manifest.csv").set_index(
        "candidate_row", drop=False
    )
    np.testing.assert_array_equal(metadata.index, np.arange(len(metadata)))
    rows = metadata.index[metadata.vun & metadata.split.eq("test")].to_numpy()
    if not len(rows):
        raise ValueError("Empty test set")
    test_systems = set(metadata.loc[rows, "chemical_system"])
    if test_systems.intersection(
        metadata.loc[metadata.split.ne("test"), "chemical_system"]
    ):
        raise ValueError("Chemical systems cross the held-out split")
    return metadata, rows


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Protocol already fixed; use the remaining phases")
    cfg = tomllib.loads((CONFIG_DIR / "heldout.toml").read_text())
    metadata, rows = manifest_rows()
    # Reproduce stored validation scores before touching any test outcome.
    stored = {}
    validation_rows = None
    for name in [
        "angular_scores",
        "random_controls",
        "diffusion_euclidean",
        "diffusion_cosine",
    ]:
        with np.load(BASELINE_RESULTS_DIR / f"validation_{name}.npz") as data:
            if validation_rows is None:
                validation_rows = data["candidate_rows"]
            np.testing.assert_array_equal(data["candidate_rows"], validation_rows)
            stored.update(zip(data["methods"].tolist(), data["scores"].T))
    values, diagnostics = fixed_scores(validation_rows, cfg)
    differences = {}
    for j, method in enumerate(cfg["methods"]):
        np.testing.assert_allclose(
            values[:, j], stored[method], rtol=1e-10, atol=1e-9, equal_nan=True
        )
        valid = np.isfinite(values[:, j])
        np.testing.assert_array_equal(valid, np.isfinite(stored[method]))
        np.testing.assert_array_equal(
            np.lexsort((validation_rows[valid], values[valid, j])),
            np.lexsort((validation_rows[valid], stored[method][valid])),
        )
        differences[method] = float(
            np.max(np.abs(values[valid, j] - stored[method][valid]))
        )
    record = {
        "frozen_at_utc": now(),
        "config": cfg,
        "test_count": len(rows),
        "test_chemical_systems": metadata.loc[rows, "chemical_system"].nunique(),
        "budget": int(len(rows) * cfg["selection_fraction"]),
        "primary_hypothesis": "1-degree gate then original score improves fixed-budget proxy stability yield versus original Crys-JEPA",
        "secondary_methods": "Fixed validation-selected comparisons; not selected again using test results",
        "novelty": "Check selected union against all same-composition MP2023 structures; no novelty-based refill",
        "uncertainty": "1000 paired chemical-system bootstrap draws; primary 95% interval; secondary intervals descriptive and unadjusted",
        "novelty_caveat": "Novelty means unmatched under this fixed matcher and bank, not a new physical phase or synthesis proof",
        "validation_reproduction_max_errors": differences,
        "validation_filter_checks": diagnostics,
        "test_outcomes_opened": False,
        "sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths()},
    }
    write_json(OUT / "protocol.json", record)
    print(
        "Frozen protocol:",
        len(rows),
        "test candidates;",
        record["budget"],
        "unique slots",
        flush=True,
    )


def protocol():
    record = json.loads((OUT / "protocol.json").read_text())
    for name, expected in record["sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen source/input changed: {name}")
    return record


def score():
    record = protocol()
    if (OUT / "scores.npz").exists():
        raise FileExistsError("Held-out scores already exist")
    _, rows = manifest_rows()
    values, diagnostics = fixed_scores(rows, record["config"])
    if np.isinf(values).any():
        raise ValueError("Infinite score")
    np.savez_compressed(
        OUT / "scores.npz",
        candidate_rows=rows,
        methods=np.array(record["config"]["methods"]),
        scores=values,
    )
    write_json(
        OUT / "score_checks.json",
        {
            "filter_checks": diagnostics,
            "candidate_energy_labels_read": False,
            "scores_sha256": sha256(OUT / "scores.npz"),
        },
    )


def structures(workers):
    record = protocol()
    if (OUT / "structure_checks.json").exists():
        raise FileExistsError("Held-out structures already audited")
    data = np.load(OUT / "scores.npz")
    metadata, rows = manifest_rows()
    np.testing.assert_array_equal(rows, data["candidate_rows"])
    checks = audit(
        metadata,
        rows,
        data["methods"].tolist(),
        data["scores"],
        record["budget"],
        OUT,
        workers,
    )
    checks["sha256"] = {
        name: sha256(OUT / name)
        for name in [
            "scores.npz",
            "duplicate_pairs.csv",
            "selections.csv",
            "novelty.csv",
        ]
    }
    write_json(OUT / "structure_checks.json", checks)
    print("Structure checks finished", flush=True)


def shared_group_summary(group_records, cfg):
    """Compare methods within the same groups; keep population weights fixed."""
    table = pd.DataFrame(group_records)
    rows = []
    for kind in ["chemical_system", "composition"]:
        frame = table[table.group_type.eq(kind)]
        base = frame[frame.method.eq(cfg["comparator"])].set_index("group")
        for method in cfg["methods"]:
            selected = frame[frame.method.eq(method)].set_index("group")
            for minimum in [1, cfg["minimum_selected_per_shared_group"]]:
                common = base.index.intersection(selected.index)
                common = common[
                    (base.loc[common, "selected"] >= minimum)
                    & (selected.loc[common, "selected"] >= minimum)
                ]
                a = base.loc[common]
                b = selected.loc[common]
                delta = b.passing / b.selected - a.passing / a.selected
                weights = a.pool_count.to_numpy(dtype=float)
                rows.append(
                    {
                        "method": method,
                        "group_type": kind,
                        "minimum_selected_each": minimum,
                        "shared_groups": len(common),
                        "pool_rows_in_shared_groups": int(weights.sum()),
                        "mean_within_group_gain": float(delta.mean())
                        if len(delta)
                        else np.nan,
                        "population_weighted_within_group_gain": float(
                            np.average(delta, weights=weights)
                        )
                        if len(delta)
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def evaluate():
    record = protocol()
    cfg = record["config"]
    if (OUT / "evaluation.json").exists():
        raise FileExistsError(
            "Test results already evaluated; do not rerun a method search"
        )
    checks = json.loads((OUT / "structure_checks.json").read_text())
    for name, expected in checks["sha256"].items():
        if sha256(OUT / name) != expected:
            raise ValueError(f"Test artifact changed after structure checks: {name}")
    # Explicit record of the first opening of held-out energy outcomes.
    marker = OUT / "outcomes_opened.json"
    if not marker.exists():
        write_json(
            marker,
            {"opened_at_utc": now(), "protocol_sha256": sha256(OUT / "protocol.json")},
        )
    metadata, rows = manifest_rows()
    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv").set_index(
        "candidate_row"
    )
    if not labels.loc[rows, "hull_status"].eq("ok").all():
        raise ValueError("Missing test energy labels")
    hull = labels.reindex(metadata.index).e_above_hull_proxy.to_numpy(float)
    targets = (hull <= cfg["stability_threshold"]).astype(int)
    selections = pd.read_csv(OUT / "selections.csv")
    selected = {
        name: group.sort_values("rank").candidate_row.to_numpy()
        for name, group in selections.groupby("method", sort=False)
    }
    novelty = pd.read_csv(OUT / "novelty.csv").set_index("candidate_row")
    score_data = np.load(OUT / "scores.npz")
    np.testing.assert_array_equal(score_data["candidate_rows"], rows)
    angle = np.rad2deg(
        np.arccos(
            np.clip(
                1
                - np.load(
                    RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r"
                )[:, 0],
                -1,
                1,
            )
        )
    )
    angle_bins = pd.Series(
        pd.cut(angle, cfg["angle_bin_edges"], include_lowest=True), index=metadata.index
    )
    composition = (
        pd.read_csv(
            RESULTS_DIR / "candidate_entries.csv", usecols=["candidate_row", "formula"]
        )
        .set_index("candidate_row")
        .formula
    )
    groupings = {
        "chemical_system": metadata.chemical_system,
        "composition": composition,
        "nearest_reference_angle": angle_bins,
    }
    original = selected[cfg["comparator"]]
    metrics, groups, sensitivity = [], [], []
    for j, name in enumerate(cfg["methods"]):
        chosen = selected[name]
        if (
            len(chosen) != record["budget"]
            or len(np.unique(chosen)) != len(chosen)
            or not np.isin(chosen, rows).all()
        ):
            raise ValueError("Invalid fixed-budget selection")
        novel = novelty.loc[chosen, "novel_against_mp2023"].to_numpy(bool)
        lo, hi = paired_interval(
            chosen, original, metadata, targets, cfg["seed"], cfg["bootstrap_draws"]
        )
        metrics.append(
            {
                "method": name,
                "role": "primary"
                if name == cfg["primary_method"]
                else "comparator"
                if name == cfg["comparator"]
                else "secondary",
                "selected": len(chosen),
                "passing": int(targets[chosen].sum()),
                "pass_rate": float(targets[chosen].mean()),
                "gain_vs_original": float(
                    targets[chosen].mean() - targets[original].mean()
                ),
                "gain_interval_low": lo,
                "gain_interval_high": hi,
                "score_coverage": float(np.isfinite(score_data["scores"][:, j]).mean()),
                "novel_count": int(novel.sum()),
                "novel_fraction": float(novel.mean()),
                "novel_and_passing": int((novel & targets[chosen].astype(bool)).sum()),
                "chemical_systems": metadata.loc[chosen, "chemical_system"].nunique(),
                "compositions": composition.loc[chosen].nunique(),
                "median_nearest_reference_angle": float(np.median(angle[chosen])),
                "below_extreme_floor": int(
                    (hull[chosen] < cfg["extreme_energy_floor"]).sum()
                ),
                "overlap_with_original": int(np.intersect1d(chosen, original).size),
            }
        )
        for threshold in cfg["secondary_thresholds"]:
            sensitivity.append(
                {
                    "method": name,
                    "check": f"hull_at_most_{threshold}",
                    "count": int((hull[chosen] <= threshold).sum()),
                    "denominator": len(chosen),
                }
            )
        retained = chosen[hull[chosen] >= cfg["extreme_energy_floor"]]
        sensitivity.append(
            {
                "method": name,
                "check": "exclude_extreme_no_refill",
                "count": int(targets[retained].sum()),
                "denominator": len(retained),
            }
        )
        for kind, grouping in groupings.items():
            pool = (
                pd.DataFrame(
                    {"group": grouping.loc[rows].to_numpy(), "target": targets[rows]}
                )
                .groupby("group", observed=True)
                .target.agg(["size", "sum"])
            )
            frame = pd.DataFrame(
                {"group": grouping.loc[chosen].to_numpy(), "target": targets[chosen]}
            )
            for key, values in frame.groupby("group", observed=True):
                groups.append(
                    {
                        "method": name,
                        "group_type": kind,
                        "group": str(key),
                        "selected": len(values),
                        "passing": int(values.target.sum()),
                        "pool_count": int(pool.loc[key, "size"]),
                        "pool_passing": int(pool.loc[key, "sum"]),
                    }
                )
        print("Evaluated fixed test method:", name, flush=True)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "group_metrics.csv", index=False)
    shared_group_summary(groups, cfg).to_csv(
        OUT / "shared_group_comparison.csv", index=False
    )
    pd.DataFrame(sensitivity).to_csv(OUT / "sensitivity.csv", index=False)
    write_json(
        OUT / "evaluation.json",
        {
            "completed_at_utc": now(),
            "test_count": len(rows),
            "unselected_pool_pass_rate": float(targets[rows].mean()),
            "primary_method": cfg["primary_method"],
            "methods_evaluated": len(metrics),
            "threshold": cfg["stability_threshold"],
            "budget": record["budget"],
            "protocol_sha256": sha256(OUT / "protocol.json"),
            "test_used_for_tuning": False,
            "independent_dft_performed": False,
            "secondary_intervals_multiplicity_adjusted": False,
            "energy_source": "MatterSim proxy versus fixed MP2023 DFT hull",
            "sha256": {
                name: sha256(OUT / name)
                for name in [
                    "metrics.csv",
                    "group_metrics.csv",
                    "shared_group_comparison.csv",
                    "sensitivity.csv",
                ]
            },
        },
    )


def report():
    record = json.loads((OUT / "protocol.json").read_text())
    cfg = record["config"]
    metrics = pd.read_csv(OUT / "metrics.csv").set_index("method")
    groups = pd.read_csv(OUT / "shared_group_comparison.csv")
    original, primary = (
        metrics.loc[cfg["comparator"]],
        metrics.loc[cfg["primary_method"]],
    )
    lines = [
        "# Fixed test comparison",
        "",
        "This run uses settings fixed before test energy outcomes were opened.",
        "",
        f"- Test candidates: {record['test_count']:,}. Reference bank: 27,136 known MP20 training crystals.",
        f"- Each method selects {record['budget']:,} unique candidates (5%, rounded down).",
        "- Pass: MatterSim proxy energy at most 0.1 eV/atom above the fixed MP2023 hull.",
        "- Known references define scores. They do not replace the evaluation hull.",
        "",
        "## Main comparison",
        "",
        f"Original Crys-JEPA: {int(original.passing):,} passing ({100 * original.pass_rate:.1f}%).",
        f"Fixed 1-degree gate then original score: {int(primary.passing):,} passing ({100 * primary.pass_rate:.1f}%).",
        f"Change: {int(primary.passing - original.passing):+d} passing candidates; {100 * primary.gain_vs_original:+.2f} percentage points.",
        f"Paired chemical-system bootstrap 95% interval: [{100 * primary.gain_interval_low:+.2f}, {100 * primary.gain_interval_high:+.2f}] percentage points.",
        "This interval describes this fixed comparison. It does not include MatterSim error.",
        "",
        "## Fixed methods",
        "",
        "| Method | Passing | Pass rate | Novel against MP2023 | Novel and passing |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in metrics.iterrows():
        lines.append(
            f"| {name} | {int(row.passing)} | {100 * row.pass_rate:.1f}% | {int(row.novel_count)} | {int(row.novel_and_passing)} |"
        )
    lines += [
        "",
        "All other methods are secondary checks. Their settings were chosen from validation, then fixed. We do not choose another rule from these test results.",
        "",
        "## Bias checks",
        "",
        f"Selected chemical systems: original {int(original.chemical_systems)}; angle gate {int(primary.chemical_systems)}.",
        f"Median angle to a known reference: original {original.median_nearest_reference_angle:.2f} degrees; gate {primary.median_nearest_reference_angle:.2f} degrees.",
        "Angle measures embedding proximity. It is not a measure of physical novelty.",
        f"MP2023 novelty: original {100 * original.novel_fraction:.1f}%; gate {100 * primary.novel_fraction:.1f}%.",
        "Novelty was checked again with StructureMatcher against every same-composition MP2023 polymorph. There is no refill after the novelty check.",
    ]
    for kind in ["chemical_system", "composition"]:
        row = groups[
            (groups.method == cfg["primary_method"])
            & (groups.group_type == kind)
            & (groups.minimum_selected_each == cfg["minimum_selected_per_shared_group"])
        ].iloc[0]
        lines.append(
            f"Within shared {kind.replace('_', ' ')} groups with at least 5 selections from each method: {int(row.shared_groups)} groups; population-weighted gain {100 * row.population_weighted_within_group_gain:+.2f} percentage points."
        )
    lines += [
        "These within-group summaries are descriptive. The shared groups cover only part of the test pool; they do not prove a cause.",
        "",
        "## Limits and next step",
        "",
        "No new model was trained. No independent DFT check was done. Release force convergence is not recorded.",
        "The split keeps generated chemical systems separate. JEPA pretraining and the known reference bank can contain these chemical systems.",
        "This test set is now open. Do not use it to tune a new rule and call the next result unseen.",
        "Review these results, then plan an independent DFT sample. New method choices need another held-out set or new generated crystals.",
        "",
        "## Reproduce",
        "",
        "Run these commands from the repo root with the prepared local inputs:",
        "",
        "```sh",
        "python -m reference_baselines.evaluation.heldout freeze",
        "python -m reference_baselines.evaluation.heldout score",
        "python -m reference_baselines.evaluation.heldout structures --workers 4",
        "python -m reference_baselines.evaluation.heldout evaluate",
        "python -m reference_baselines.evaluation.heldout report",
        "```",
        "",
        "Freeze, score, structures, and evaluate refuse to overwrite a completed phase. The protocol stores source and input hashes.",
        "Local outputs: `reference_baselines/results/heldout/`. Input and output data stay out of Git.",
        "",
    ]
    (DOCS_DIR / "heldout_experiment.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["freeze", "score", "structures", "evaluate", "report"]
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.phase == "structures":
        structures(args.workers)
    else:
        {"freeze": freeze, "score": score, "evaluate": evaluate, "report": report}[
            args.phase
        ]()


if __name__ == "__main__":
    main()
