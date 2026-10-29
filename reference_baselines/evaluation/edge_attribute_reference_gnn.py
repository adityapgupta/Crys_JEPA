"""Evaluate frozen edge-attribute checkpoints on validation smoke or reserved rows."""

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reference_baselines.core.io import sha256
from reference_baselines.datasets.reference_graph import prepare_table, split_rows
from reference_baselines.datasets.edge_attribute_reference_graph import EdgeAttributePoolDataset, edge_attribute_collate
from reference_baselines.evaluation.metrics import load_adjacency as validation_adjacency
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.evaluation.reference_gnn import evaluate, fixed_score_tables, rank
from reference_baselines.evaluation.reserved_duplicates import adjacency as reserved_adjacency
from reference_baselines.models.edge_attribute_reference_gnn import build_model
from reference_baselines.paths import CONFIG_DIR, RESULTS_DIR
from reference_baselines.training.train_edge_attribute_reference_gnn import EXPERIMENT, run_specs


def frozen_checkpoints(cfg, root):
    loaded = {}
    for seed in cfg["seeds"]:
        for variant, kind in run_specs(cfg):
            directory = root / f"{variant}_{kind}_physical_seed{seed}"
            completion, checkpoint = directory / "complete.json", directory / "best.pt"
            if not completion.exists() or not checkpoint.exists():
                raise FileNotFoundError(f"Incomplete run: {directory}")
            record = json.loads(completion.read_text())
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                state["variant"] != variant
                or state["kind"] != kind
                or state["graph"] != "physical"
                or state["config"] != cfg
            ):
                raise ValueError(f"Checkpoint mismatch: {checkpoint}")
            model = build_model(variant, kind, cfg["hidden"], cfg["layers"]).eval()
            model.load_state_dict(state["state_dict"], strict=True)
            loaded[(variant, kind, seed)] = {
                "model": model, "sha256": sha256(checkpoint),
                "epoch": record["best_epoch"], "validation_bce": record["best_validation_bce"],
            }
    return loaded


def model_scores(loaded, datasets, cfg):
    grouped = {}
    for key, record in loaded.items():
        edge_mode = "shuffled" if key[0] == "shuffled" else "rich"
        grouped.setdefault(edge_mode, []).append((key, record))
    result, rows = {}, None
    with torch.inference_mode():
        for edge_mode, records in grouped.items():
            loader = DataLoader(
                datasets[edge_mode], batch_size=cfg["batch_size"], shuffle=False,
                collate_fn=edge_attribute_collate, num_workers=0,
            )
            values = {key: [] for key, _ in records}
            current_rows = []
            for batch in loader:
                current_rows.extend(batch["rows"].tolist())
                for key, record in records:
                    _, _, score = record["model"](batch)
                    values[key].extend(score.tolist())
            current_rows = np.asarray(current_rows)
            np.testing.assert_array_equal(current_rows, datasets[edge_mode].rows)
            if rows is None:
                rows = current_rows
            else:
                np.testing.assert_array_equal(rows, current_rows)
            result.update({key: np.asarray(value) for key, value in values.items()})
    return rows, result


def controlled_comparisons(rows, adjacency, labels, manifest, scores, cfg):
    targets = pd.Series(labels.e_above_hull_proxy.le(0.1).to_numpy(int), index=labels.index)
    orders = {name: rank(rows, value, adjacency) for name, value in scores.items()}
    percent = cfg["primary_budget_percent"]
    budget = len(rows) * percent // 100
    if budget < 1:
        raise ValueError("Evaluation set is too small for the primary budget")
    records = []

    def add_contrast(treatment, control, contrast, kind, seed):
        a, b = orders[treatment][:budget], orders[control][:budget]
        low, high = paired_interval(
            a,
            b,
            manifest.loc[rows],
            targets,
            seed=seed,
            draws=cfg["bootstrap_draws"],
        )
        records.append(
            {
                "contrast": contrast,
                "model": kind,
                "seed": seed,
                "budget_percent": percent,
                "treatment_passes": int(targets.loc[a].sum()),
                "control_passes": int(targets.loc[b].sum()),
                "delta_yield": float(targets.loc[a].mean() - targets.loc[b].mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )

    for seed in cfg["seeds"]:
        readout = f"readout_readout_seed{seed}"
        for kind in cfg["models"]:
            scalar = f"scalar_{kind}_seed{seed}"
            rich = f"rich_{kind}_seed{seed}"
            shuffled = f"shuffled_{kind}_seed{seed}"
            for treatment, control, contrast in [
                (rich, scalar, "rich_minus_scalar"),
                (rich, shuffled, "rich_minus_shuffled"),
                (rich, readout, "rich_minus_readout"),
                (readout, scalar, "readout_minus_scalar"),
            ]:
                add_contrast(treatment, control, contrast, kind, seed)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "edge_attribute_reference_gnn.toml"))
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
        table = prepare_table(split, graph="physical", k=cfg["retrieval_k"], limit=limit)
        duplicate = (
            RESULTS_DIR / "validation_duplicate_pairs.csv"
            if args.smoke else RESULTS_DIR / "reference_gnn/reserved_duplicate_pairs.csv"
        )
        if not duplicate.exists():
            raise FileNotFoundError(f"Missing frozen duplicate table: {duplicate}")
        record = {
            "status": "prepared", "split": split,
            "pool_table": str(table.relative_to(RESULTS_DIR.parent)),
            "duplicate_table_sha256": sha256(duplicate), "labels_read": False,
        }
        (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return
    if not (out / "prepare.json").exists():
        raise FileNotFoundError("Run --prepare before --evaluate")
    if (out / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use a new output directory")

    # Freeze every checkpoint before opening candidate outcomes.
    loaded = frozen_checkpoints(cfg, run_dir)
    rows = split_rows(split)
    if limit and limit < len(rows):
        rows = np.sort(np.random.default_rng(cfg["seeds"][0]).choice(rows, limit, replace=False))
    datasets = {
        mode: EdgeAttributePoolDataset(
            split, graph="physical", k=cfg["retrieval_k"], limit=limit,
            seed=cfg["seeds"][0], edge_mode=mode, edge_seed=cfg["edge_seed"],
        )
        for mode in ("rich", "shuffled")
    }
    scored_rows, values = model_scores(loaded, datasets, cfg)
    np.testing.assert_array_equal(scored_rows, rows)
    labels = pd.read_csv(RESULTS_DIR / "candidate_labels.csv", keep_default_na=False).set_index("candidate_row")
    manifest = pd.read_csv(RESULTS_DIR / "split_manifest.csv", keep_default_na=False).set_index("candidate_row")
    if not labels.loc[rows, "hull_status"].eq("ok").all():
        raise ValueError(f"{split.title()} rows lack usable hull labels")
    adjacency = validation_adjacency() if args.smoke else reserved_adjacency()
    scores = fixed_score_tables(rows)
    checkpoint_records = []
    for key, value in values.items():
        variant, kind, seed = key
        name = f"{variant}_{kind}_seed{seed}"
        scores[name] = value
        checkpoint_records.append(
            {
                "method": name, "checkpoint_sha256": loaded[key]["sha256"],
                "epoch": loaded[key]["epoch"], "validation_bce": loaded[key]["validation_bce"],
            }
        )
    metrics, _ = evaluate(rows, adjacency, labels, manifest, scores)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"candidate_row": rows, **scores}).to_csv(out / "scores.csv", index=False)
    controlled_comparisons(rows, adjacency, labels, manifest, scores, cfg).to_csv(
        out / "controlled_comparisons.csv", index=False
    )
    record = {
        "status": "complete", "split": split, "rows": len(rows),
        "edge_features": cfg["edge_features"], "topology": cfg["topology"],
        "config_sha256": sha256(Path(args.config)),
        "checkpoints": checkpoint_records,
        "limits": cfg["claim_limit"],
    }
    (out / "evaluation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
