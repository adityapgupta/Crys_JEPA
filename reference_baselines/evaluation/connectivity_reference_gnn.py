"""Evaluate the connectivity experiment: does any graph wiring beat the readout?

Frozen reference-utility checkpoints are scored once on the reserved partition.
The decisive, pre-registered comparison is each topology's GNN minus the
``no_graph`` readout control (no message passing), at the primary budget, with a
paired chemical-system bootstrap. We also contrast each alternative topology's GNN
against the ``physical`` GNN.

Node features, pools, target, checkpoint rule and screening score are identical to
the base reference GNN; only the graph topology differs between runs. Candidate
stability labels are opened only after every checkpoint exists and is frozen.

Run:
  python -m reference_baselines.evaluation.connectivity_reference_gnn --config C --prepare
  python -m reference_baselines.evaluation.connectivity_reference_gnn --config C --evaluate
"""

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import torch

from reference_baselines.core.io import sha256
from reference_baselines.datasets.reference_graph import (
    PoolDataset,
    prepare_table,
    split_rows,
)
from reference_baselines.evaluation.metrics import load_adjacency as validation_adjacency
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.evaluation.reference_gnn import (
    evaluate,
    fixed_score_tables,
    frozen_checkpoints,
    learned_scores,
    rank,
)
from reference_baselines.evaluation.reserved_duplicates import adjacency as reserved_adjacency
from reference_baselines.paths import CONFIG_DIR, RESULTS_DIR


EXPERIMENT = RESULTS_DIR / "connectivity_reference_gnn"
GNN_KINDS = ("gcn", "gat", "sage")
READOUT_KIND = "no_graph"


def topologies(cfg):
    """physical first, then the alternative wirings, de-duplicated and ordered."""
    ordered = [cfg["graph_primary"], *cfg["graph_ablations"]]
    seen, result = set(), []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def score_all(cfg, run_dir, split, limit):
    """Return reserved rows and a score table keyed by method name."""
    scores = {}
    checkpoint_records = []
    rows_reference = None
    # Readout control: no_graph trained under the physical graph (edges unused).
    readout = frozen_checkpoints(cfg, cfg["graph_primary"], run_dir, [READOUT_KIND], cfg["seeds"])
    physical_dataset = PoolDataset(
        split, graph=cfg["graph_primary"], k=cfg["retrieval_k"], limit=limit, seed=cfg["seeds"][0]
    )
    rows, readout_scores = learned_scores(readout, physical_dataset, cfg)
    rows_reference = rows
    for (kind, seed), score in readout_scores.items():
        scores[f"{kind}_readout_seed{seed}"] = score
        checkpoint_records.append(
            {"model": kind, "graph": "readout", "seed": seed,
             "checkpoint_sha256": readout[(kind, seed)]["sha256"],
             "validation_bce": readout[(kind, seed)]["validation_bce"]}
        )
    # Message-passing GNNs on each topology.
    for graph in topologies(cfg):
        loaded = frozen_checkpoints(cfg, graph, run_dir, list(GNN_KINDS), cfg["seeds"])
        dataset = PoolDataset(split, graph=graph, k=cfg["retrieval_k"], limit=limit, seed=cfg["seeds"][0])
        grows, values = learned_scores(loaded, dataset, cfg)
        np.testing.assert_array_equal(grows, rows_reference)
        for (kind, seed), score in values.items():
            scores[f"{kind}_{graph}_seed{seed}"] = score
            checkpoint_records.append(
                {"model": kind, "graph": graph, "seed": seed,
                 "checkpoint_sha256": loaded[(kind, seed)]["sha256"],
                 "validation_bce": loaded[(kind, seed)]["validation_bce"]}
            )
    return rows_reference, scores, pd.DataFrame(checkpoint_records)


def controlled_comparisons(rows, adjacency, labels, manifest, scores, cfg):
    """Each topology's GNN minus readout, and minus the physical GNN, at 5%."""
    targets = pd.Series(labels.e_above_hull_proxy.le(0.1).to_numpy(int), index=labels.index)
    metadata = manifest.loc[rows]
    orders = {name: rank(rows, value, adjacency) for name, value in scores.items()}
    percent = cfg["primary_budget_percent"]
    budget = len(rows) * percent // 100
    if budget < 1:
        raise ValueError("Evaluation set is too small for the primary budget")
    records = []

    def add_contrast(treatment, control, contrast, kind, seed):
        if treatment not in orders or control not in orders:
            return
        a, b = orders[treatment][:budget], orders[control][:budget]
        low, high = paired_interval(a, b, metadata, targets, seed=seed, draws=cfg["bootstrap_draws"])
        records.append(
            {"contrast": contrast, "model": kind, "seed": seed, "budget_percent": percent,
             "treatment_passes": int(targets.loc[a].sum()),
             "control_passes": int(targets.loc[b].sum()),
             "delta_yield": float(targets.loc[a].mean() - targets.loc[b].mean()),
             "ci_low": low, "ci_high": high}
        )

    graphs = topologies(cfg)
    for seed in cfg["seeds"]:
        readout = f"{READOUT_KIND}_readout_seed{seed}"
        for kind in GNN_KINDS:
            for graph in graphs:
                add_contrast(f"{kind}_{graph}_seed{seed}", readout,
                             f"{graph}_{kind}_minus_readout", kind, seed)
            for graph in graphs:
                if graph == cfg["graph_primary"]:
                    continue
                add_contrast(f"{kind}_{graph}_seed{seed}", f"{kind}_{cfg['graph_primary']}_seed{seed}",
                             f"{graph}_{kind}_minus_physical", kind, seed)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "connectivity_reference_gnn.toml"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    if args.prepare == args.evaluate:
        parser.error("Select exactly one of --prepare or --evaluate")
    cfg = tomllib.loads(Path(args.config).read_text())
    split = "validation" if args.smoke else "reserved"
    limit = args.limit if args.limit is not None else (300 if args.smoke else 0)
    run_dir = Path(args.run_dir) if args.run_dir else EXPERIMENT / ("smoke" if args.smoke else "runs")
    out = Path(args.out_dir) if args.out_dir else EXPERIMENT / ("smoke_evaluation" if args.smoke else "reserved")
    out.mkdir(parents=True, exist_ok=True)

    if args.prepare:
        tables = [
            str(prepare_table(split, graph=graph, k=cfg["retrieval_k"], limit=limit).relative_to(RESULTS_DIR.parent))
            for graph in topologies(cfg)
        ]
        duplicate = (
            RESULTS_DIR / "validation_duplicate_pairs.csv" if args.smoke
            else RESULTS_DIR / "reference_gnn" / "reserved_duplicate_pairs.csv"
        )
        if not duplicate.exists():
            raise FileNotFoundError(f"Missing frozen duplicate table: {duplicate}")
        record = {"status": "prepared", "split": split, "pool_tables": tables,
                  "duplicate_table_sha256": sha256(duplicate), "labels_read": False}
        (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return

    if not (out / "prepare.json").exists():
        raise FileNotFoundError("Run --prepare before --evaluate")
    if (out / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use a new output directory")

    rows, model_scores, checkpoint_records = score_all(cfg, run_dir, split, limit)
    reserved = split_rows(split)
    if limit and limit < len(reserved):
        reserved = np.sort(np.random.default_rng(cfg["seeds"][0]).choice(reserved, limit, replace=False))
    np.testing.assert_array_equal(rows, reserved)

    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv", keep_default_na=False).set_index("candidate_row")
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv", keep_default_na=False).set_index("candidate_row")
    if not labels.loc[rows, "hull_status"].eq("ok").all():
        raise ValueError(f"{split.title()} rows lack usable hull labels")
    adjacency = validation_adjacency() if args.smoke else reserved_adjacency()

    scores = fixed_score_tables(rows)
    scores.update(model_scores)
    selected_labels = labels.loc[rows]
    metrics, _ = evaluate(rows, adjacency, selected_labels, manifest, scores)
    metrics.to_csv(out / "metrics.csv", index=False)
    controlled_comparisons(rows, adjacency, selected_labels, manifest, model_scores, cfg).to_csv(
        out / "controlled_comparisons.csv", index=False
    )
    checkpoint_records.to_csv(out / "checkpoints.csv", index=False)
    report = {
        "status": "complete", "split": split, "rows": int(len(rows)),
        "topologies": topologies(cfg), "readout_control": READOUT_KIND,
        "primary_budget_percent": cfg["primary_budget_percent"],
        "config_sha256": sha256(Path(args.config)),
        "labels_opened_after_checkpoints_frozen": True,
        "limit": cfg["claim_limit"],
    }
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
