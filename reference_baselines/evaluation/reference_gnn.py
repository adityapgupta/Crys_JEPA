"""Score frozen reference-utility models once on the reserved partition.

Candidate scores:
  learned     sum_i w_i * distance_i / sum_i w_i   (lower is better)
  uniform     equal reference weights
  random      matched random reference pool

Baselines use the same candidate rows, the same StructureMatcher duplicate
rule, and the same fixed budgets. Reserved labels are opened only after all
checkpoints exist and pass a frozen-checkpoint check.

Run:
  python -m reference_baselines.evaluation.reference_gnn --prepare --out-dir OUT
  python -m reference_baselines.evaluation.reference_gnn --evaluate --out-dir OUT

Use ``--smoke`` only for engineering checks. It uses the validation rows and a
separate output directory. Official evaluation uses the reserved rows once,
after all checkpoints are frozen.
"""

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reference_baselines.core.angular import raw_distances, within_angle
from reference_baselines.core.io import sha256
from reference_baselines.datasets.reference_graph import (
    EXPERIMENT,
    PoolDataset,
    collate,
    prepare_table,
    split_rows,
)
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.evaluation.metrics import load_adjacency as validation_adjacency
from reference_baselines.evaluation.reserved_duplicates import (
    adjacency as reserved_adjacency,
    build as build_duplicates,
)
from reference_baselines.models.reference_gnn import ReferenceGNN
from reference_baselines.paths import CONFIG_DIR, RESULTS_DIR, ROOT


BUDGETS = (1, 2, 5, 7)
DEFAULT_OUT = EXPERIMENT / "reserved"
DEFAULT_SMOKE_OUT = EXPERIMENT / "smoke_evaluation"


def frozen_checkpoints(cfg, graph, root, models, seeds):
    """Load every trained checkpoint, or fail before opening reserved labels."""
    loaded = {}
    for seed in seeds:
        for kind in models:
            directory = root / f"{kind}_{graph}_seed{seed}"
            completion = directory / "complete.json"
            checkpoint = directory / "best.pt"
            if not completion.exists() or not checkpoint.exists():
                raise FileNotFoundError(f"Incomplete run: {directory}")
            record = json.loads(completion.read_text())
            model = ReferenceGNN(kind, cfg["hidden"], cfg["layers"]).eval()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state["state_dict"], strict=True)
            if state["graph"] != graph or state["kind"] != kind:
                raise ValueError(f"Checkpoint mismatch: {checkpoint}")
            loaded[(kind, seed)] = {
                "model": model,
                "sha256": sha256(checkpoint),
                "epoch": record["best_epoch"],
                "validation_bce": record["best_validation_bce"],
            }
    return loaded


def learned_scores(
    loaded,
    dataset,
    cfg,
    self_only=False,
    uniform_weights=False,
    prior_weights=False,
):
    """Return one candidate score per row for each frozen model."""
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate
    )
    result = {key: [] for key in loaded}
    rows = []
    with torch.inference_mode():
        for batch in loader:
            rows.extend(batch["rows"].tolist())
            for key, entry in loaded.items():
                _, _, score = entry["model"](
                    batch,
                    self_only=self_only,
                    uniform_weights=uniform_weights,
                    prior_weights=prior_weights,
                )
                result[key].extend(score.tolist())
    rows = np.array(rows)
    np.testing.assert_array_equal(rows, dataset.rows)
    return rows, {key: np.array(values) for key, values in result.items()}


def uniform_pool_scores(dataset, cfg):
    """Return equal-weight mean distances for a prepared reference pool."""
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate
    )
    rows, values = [], []
    with torch.inference_mode():
        for batch in loader:
            rows.extend(batch["rows"].tolist())
            counts = torch.bincount(batch["owner"], minlength=len(batch["rows"]))
            sums = torch.zeros(len(batch["rows"]), dtype=batch["distance"].dtype)
            sums.index_add_(0, batch["owner"], batch["distance"])
            values.extend((sums / counts).tolist())
    rows = np.array(rows)
    np.testing.assert_array_equal(rows, dataset.rows)
    return rows, np.array(values)


def fixed_score_tables(rows):
    """Load the frozen label-free baselines on the same rows."""
    table = pd.read_csv(RESULTS_DIR / "candidate_reference_scores.csv").set_index(
        "candidate_row"
    )
    result = {
        "original_crys_jepa": table.loc[rows, "crys_jepa"].to_numpy(float),
        "knn_k5": table.loc[rows, "knn_k5"].to_numpy(float),
        "intersection_fallback_k5": table.loc[rows, "intersection_fallback_k5"].to_numpy(float),
    }
    # Fixed angle rule: the same 1-degree gate, then the original score.
    cosine = np.load(RESULTS_DIR / "candidate_cosine_distances.npy", mmap_mode="r")[
        rows, :10
    ]
    angular = within_angle(cosine, 1)
    gated = angular.any(1)
    scores = np.full(len(rows), np.nan)
    reference = np.asarray(
        np.load(ROOT / "embeddings/data/embeddings.npy"), dtype=np.float64
    )
    query = np.load(ROOT / "embeddings/data/candidate_embeddings.npy", mmap_mode="r")
    indices = np.load(RESULTS_DIR / "candidate_cosine_indices.npy", mmap_mode="r")[
        rows, :10
    ]
    for start in range(0, len(rows), 256):
        stop = min(start + 256, len(rows))
        distance = raw_distances(
            np.asarray(query[rows[start:stop]], dtype=np.float64)[:, None, :],
            reference[indices[start:stop]],
        )
        mask = angular[start:stop]
        count = mask.sum(1)
        mean = np.divide(
            np.where(mask, distance, 0).sum(1),
            count,
            out=np.full(stop - start, np.nan),
            where=count > 0,
        )
        scores[start:stop] = mean
    result["angle_gate_1deg"] = np.where(gated, scores, table.loc[rows, "crys_jepa"].to_numpy(float))
    return result


def rank(rows, scores, adjacency):
    finite = np.isfinite(scores)
    order = rows[finite][np.lexsort((rows[finite], scores[finite]))]
    return unique_order(order, adjacency)


def evaluate(rows, adjacency, labels, manifest, scores):
    """Fixed-budget unique pass counts. No per-budget tuning."""
    targets = pd.Series(
        labels.e_above_hull_proxy.le(0.1).to_numpy(int), index=labels.index
    )
    records, orders = [], {}
    for method, values in scores.items():
        order = rank(rows, values, adjacency)
        orders[method] = order
        for percent in BUDGETS:
            budget = len(rows) * percent // 100
            if len(order) < budget:
                raise ValueError(f"{method} cannot fill unique {percent}% budget")
            chosen = order[:budget]
            records.append(
                {
                    "method": method,
                    "budget_percent": percent,
                    "selected": budget,
                    "passes": int(targets.loc[chosen].sum()),
                    "yield": float(targets.loc[chosen].sum() / budget),
                    "submitted": int(budget),
                    "chemical_systems": int(
                        manifest.loc[chosen, "chemical_system"].nunique()
                    ),
                }
            )
    return pd.DataFrame(records), orders


def comparisons(rows, adjacency, labels, manifest, scores, seed=20261023):
    """Paired chemical-system intervals against the original method."""
    targets = pd.Series(
        labels.e_above_hull_proxy.le(0.1).to_numpy(int), index=labels.index
    )
    metadata = manifest.loc[rows]
    orders = {method: rank(rows, values, adjacency) for method, values in scores.items()}
    records = []
    for method, order in orders.items():
        if method == "original_crys_jepa":
            continue
        for percent in (5,):
            budget = len(rows) * percent // 100
            selected, baseline = order[:budget], orders["original_crys_jepa"][:budget]
            interval = paired_interval(
                selected,
                baseline,
                metadata,
                targets,
                seed=seed,
                draws=1000,
            )
            records.append(
                {
                    "method": method,
                    "budget_percent": percent,
                    "passes": int(targets.loc[selected].sum()),
                    "original_passes": int(targets.loc[baseline].sum()),
                    "gain": int(targets.loc[selected].sum() - targets.loc[baseline].sum()),
                    "gain_percentage_points": float(
                        100 * (targets.loc[selected].mean() - targets.loc[baseline].mean())
                    ),
                    "bootstrap_95_percent": json.dumps(interval),
                }
            )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "reference_gnn.toml"))
    parser.add_argument("--prepare", action="store_true", help="Build reserved pools only")
    parser.add_argument("--evaluate", action="store_true", help="Open reserved labels and score")
    parser.add_argument("--run-dir", default=str(EXPERIMENT / "runs"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only this many rows. Smoke mode defaults to 300.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Smoke mode defaults to a separate directory.",
    )
    parser.add_argument(
        "--reuse-duplicates",
        action="store_true",
        help="Use the precomputed reserved duplicate table instead of rebuilding it.",
    )
    args = parser.parse_args()
    cfg = tomllib.loads(open(args.config).read())
    if not args.prepare and not args.evaluate:
        raise SystemExit("Select --prepare or --evaluate")
    if args.smoke and not args.out_dir:
        args.out_dir = str(DEFAULT_SMOKE_OUT)
    out = Path(args.out_dir or DEFAULT_OUT)
    if args.smoke and not (args.evaluate or args.prepare):
        raise SystemExit("Smoke mode still requires --prepare or --evaluate")
    if args.smoke and out == DEFAULT_OUT:
        raise SystemExit("Smoke output must not overwrite the official reserved directory")
    split = "validation" if args.smoke else "reserved"
    limit = args.limit if args.limit is not None else (300 if args.smoke else 0)
    out.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        tables = []
        for graph in ["physical", *cfg["graph_ablations"]]:
            tables.append(
                str(
                    prepare_table(
                        split, graph=graph, k=cfg["retrieval_k"], limit=limit
                    ).relative_to(RESULTS_DIR.parent)
                )
            )
        tables.append(
            str(
                prepare_table(
                    split,
                    graph="physical",
                    k=cfg["retrieval_k"],
                    pool="random",
                    seed=cfg["seeds"][0],
                    limit=limit,
                ).relative_to(RESULTS_DIR.parent)
            )
        )
        if args.smoke:
            duplicate_record = {
                "status": "skipped",
                "reason": "Smoke mode uses the frozen validation duplicate table.",
                "labels_read": False,
            }
        elif args.reuse_duplicates:
            table = EXPERIMENT / "reserved_duplicate_pairs.csv"
            if not table.exists():
                raise FileNotFoundError(f"Missing duplicate table: {table}")
            duplicate_record = {
                "status": "reused",
                "table_sha256": sha256(table),
                "labels_read": False,
            }
        else:
            _, duplicate_record = build_duplicates(workers=cfg["threads"])
        record = {
            "status": "prepared",
            "split": split,
            "pool_tables": tables,
            "reserved_duplicates": duplicate_record,
            "labels_read": False,
        }
        (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return

    if not (out / "prepare.json").exists():
        raise FileNotFoundError("Run --prepare before --evaluate")
    # Freeze check first. Candidate labels are opened only after every
    # checkpoint exists and matches the frozen protocol.
    frozen = {}
    for graph in ["physical", *cfg["graph_ablations"]]:
        frozen[graph] = frozen_checkpoints(
            cfg, graph, Path(args.run_dir), cfg["models"], cfg["seeds"]
        )
    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv", keep_default_na=False)
    labels = labels.set_index("candidate_row")
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv", keep_default_na=False)
    manifest = manifest.set_index("candidate_row")
    rows = split_rows(split)
    if limit and limit < len(rows):
        rows = np.sort(
            np.random.default_rng(cfg["seeds"][0]).choice(
                rows, limit, replace=False
            )
        )
    if not labels.loc[rows, "hull_status"].eq("ok").all():
        raise ValueError(f"{split.title()} rows lack usable hull labels")
    adjacency = validation_adjacency() if args.smoke else reserved_adjacency()

    scores = fixed_score_tables(rows)
    model_scores = {}
    records = []
    uniform_natural = None
    for graph in ["physical", *cfg["graph_ablations"]]:
        loaded = frozen[graph]
        dataset = PoolDataset(
            split,
            graph=graph,
            k=cfg["retrieval_k"],
            limit=limit,
            seed=cfg["seeds"][0],
        )
        _, values = learned_scores(loaded, dataset, cfg)
        for (kind, seed), score in values.items():
            model_scores[f"{kind}_{graph}_seed{seed}"] = score
            records.append(
                {
                    "model": kind,
                    "graph": graph,
                    "seed": seed,
                    "checkpoint_sha256": loaded[(kind, seed)]["sha256"],
                    "epoch": loaded[(kind, seed)]["epoch"],
                    "validation_bce": loaded[(kind, seed)]["validation_bce"],
                }
            )
        if graph == "physical":
            uniform_rows, uniform_natural = uniform_pool_scores(dataset, cfg)
            np.testing.assert_array_equal(uniform_rows, rows)
            _, self_only = learned_scores(loaded, dataset, cfg, self_only=True)
            for (kind, seed), score in self_only.items():
                if kind != "no_graph":
                    model_scores[f"{kind}_self_only_seed{seed}"] = score
    scores.update(model_scores)
    scores["uniform_natural_pool"] = uniform_natural
    natural_dataset = PoolDataset(
        split,
        graph="physical",
        k=cfg["retrieval_k"],
        limit=limit,
        seed=cfg["seeds"][0],
    )
    # Formula-prior control: use the primary-graph checkpoints. The prior-only
    # path ignores the learned logits, so this control has no model dependence.
    _, prior_scores = learned_scores(
        frozen[cfg["graph_primary"]],
        natural_dataset,
        cfg,
        prior_weights=True,
    )
    first = next(iter(prior_scores.values()))
    scores["formula_prior_natural_pool"] = first

    # Random-reference control: same score rule and pool size, uniform weights.
    random_dataset = PoolDataset(
        split,
        graph="physical",
        k=cfg["retrieval_k"],
        pool="random",
        seed=cfg["seeds"][0],
        limit=limit,
    )
    random_rows, random_scores = uniform_pool_scores(random_dataset, cfg)
    np.testing.assert_array_equal(random_rows, rows)
    scores["random_references"] = random_scores

    selected_labels = labels.loc[rows]
    metrics, orders = evaluate(rows, adjacency, selected_labels, manifest, scores)
    table = comparisons(rows, adjacency, selected_labels, manifest, scores)
    metrics.to_csv(out / "metrics.csv", index=False)
    table.to_csv(out / "comparisons.csv", index=False)
    pd.DataFrame(records).to_csv(out / "checkpoints.csv", index=False)
    selections = []
    for method, order in orders.items():
        for percent in BUDGETS:
            budget = len(rows) * percent // 100
            selections.extend(
                {
                    "method": method,
                    "budget_percent": percent,
                    "rank": position,
                    "candidate_row": int(row),
                }
                for position, row in enumerate(order[:budget], 1)
            )
    pd.DataFrame(selections).to_csv(out / "selections.csv", index=False)
    report = {
        "status": "passed",
        "split": split,
        "rows": int(len(rows)),
        "budgets": list(BUDGETS),
        "methods": list(scores),
        "labels_opened_after_checkpoints_frozen": True,
        "smoke": args.smoke,
        "legacy_test_used": False,
        "checkpoint_rule": cfg["checkpoint_rule"],
        "limit": "MatterSim proxy screen against the fixed MP2023 hull; not DFT confirmation.",
        "source_sha256": {
            "evaluation": sha256(Path(__file__)),
            "duplicates": (
                sha256(EXPERIMENT / "reserved_duplicate_pairs.csv")
                if not args.smoke
                else sha256(RESULTS_DIR / "validation_duplicate_pairs.csv")
            ),
        },
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(metrics.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
