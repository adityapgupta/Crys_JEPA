"""Train reference-utility GNNs on fit rows only.

The loss is binary cross entropy on reference decomposition membership.
Checkpoint choice uses validation reference BCE only. Candidate stability
labels are never read. The reserved partition is never opened here.

Run:
  python -m reference_baselines.training.train_reference_gnn --smoke
  python -m reference_baselines.training.train_reference_gnn
"""

import argparse
import json
import resource
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from reference_baselines.core.io import sha256
from reference_baselines.datasets.reference_graph import (
    EXPERIMENT,
    PoolDataset,
    collate,
)
from reference_baselines.models.reference_gnn import ReferenceGNN, reference_loss
from reference_baselines.paths import CONFIG_DIR, PACKAGE_DIR, RESULTS_DIR


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def reference_metrics(model, loader, device):
    """Validation reference BCE and rank quality. No candidate labels."""
    model.eval()
    loss_sum, count = 0.0, 0
    logits_all, target_all = [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            logits, _, _ = model(batch)
            loss = reference_loss(logits, batch)
            loss_sum += float(loss) * len(logits)
            count += len(logits)
            logits_all.append(logits.cpu())
            target_all.append(batch["target"].cpu())
    logits = torch.cat(logits_all)
    target = torch.cat(target_all)
    probability = logits.sigmoid()
    return {
        "bce": loss_sum / max(count, 1),
        "brier": float(torch.square(probability - target).mean()),
        "positives": int(target.sum()),
        "references": int(len(target)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "reference_gnn.toml"))
    parser.add_argument(
        "--model", choices=["gcn", "gat", "sage", "no_graph"], default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--graph", default=None, help="Override the graph ablation")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Checkpoint root. Defaults to results/reference_gnn/runs.",
    )
    args = parser.parse_args()
    cfg = tomllib.loads(open(args.config).read())
    torch.set_num_threads(min(cfg["threads"], torch.get_num_threads()))
    device = torch.device(args.device or cfg["device"])
    limit = 512 if args.smoke else cfg["train_limit"]
    validation_limit = 256 if args.smoke else cfg["validation_limit"]
    epochs = 1 if args.smoke else cfg["epochs"]

    models = [args.model] if args.model else cfg["models"]
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    graph = args.graph or cfg["graph_primary"]
    root = (
        Path(args.run_dir)
        if args.run_dir
        else EXPERIMENT / ("smoke" if args.smoke else "runs")
    )

    protocol = {
        "settings": cfg,
        "graph": graph,
        "smoke": args.smoke,
        "train_limit": limit,
        "validation_limit": validation_limit,
        "reserved_opened": False,
        "legacy_test_opened": False,
        "candidate_stability_targets": False,
        "features": "512 frozen JEPA dimensions, 100 atomic fractions, log distance, chemical flag, cosine distance",
        "source_sha256": {
            str(path.relative_to(PACKAGE_DIR.parent)): sha256(path)
            for path in [
                PACKAGE_DIR / "datasets/reference_graph.py",
                PACKAGE_DIR / "datasets/decomposition.py",
                PACKAGE_DIR / "models/reference_gnn.py",
                PACKAGE_DIR / "training/train_reference_gnn.py",
                PACKAGE_DIR / "configs/reference_gnn.toml",
            ]
        },
        "input_sha256": {
            str(path.relative_to(PACKAGE_DIR.parent)): sha256(path)
            for path in [
                RESULTS_DIR / "gnn_comparison/train_rows.npy",
                RESULTS_DIR / "gnn_comparison/validation_rows.npy",
                RESULTS_DIR / "gnn_comparison/reserved_rows.npy",
                RESULTS_DIR / "candidate_decompositions.csv",
                RESULTS_DIR / "chemical_reference_indptr.npy",
                RESULTS_DIR / "chemical_reference_indices.npy",
                RESULTS_DIR / "candidate_cosine_indices.npy",
                EXPERIMENT / "graphs" / f"{graph}.npz",
            ]
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / f"train_protocol_{graph}.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        frozen = ("settings", "graph", "train_limit", "validation_limit")
        if any(previous[key] != protocol[key] for key in frozen):
            raise ValueError(
                f"{protocol_path} changed. Use a new results directory for a new protocol."
            )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    train = PoolDataset(
        "train", graph=graph, k=cfg["retrieval_k"], limit=limit, seed=cfg["seeds"][0]
    )
    validation = PoolDataset(
        "validation",
        graph=graph,
        k=cfg["retrieval_k"],
        limit=validation_limit,
        seed=cfg["seeds"][0],
    )
    print(
        f"fit rows {len(train)} | validation rows {len(validation)} | graph {graph}",
        flush=True,
    )
    train_loader = DataLoader(
        train, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=0
    )
    validation_loader = DataLoader(
        validation, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate, num_workers=0
    )

    completed = []
    for seed in seeds:
        for kind in models:
            directory = root / f"{kind}_{graph}_seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                print("Already complete:", directory.name, flush=True)
                completed.append(directory.name)
                continue
            seed_everything(seed)
            model = ReferenceGNN(kind, cfg["hidden"], cfg["layers"]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
            best_epoch, best_bce, stale = -1, float("inf"), 0
            history = []
            started = time.perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                running, seen = 0.0, 0
                for batch in train_loader:
                    batch = {
                        key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    optimizer.zero_grad(set_to_none=True)
                    logits, _, _ = model(batch)
                    loss = reference_loss(logits, batch)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    running += loss.item() * len(logits)
                    seen += len(logits)
                train_bce = running / max(seen, 1)
                metrics = reference_metrics(model, validation_loader, device)
                record = {
                    "epoch": epoch,
                    "train_bce": train_bce,
                    "validation_bce": metrics["bce"],
                    "validation_brier": metrics["brier"],
                    "seconds": time.perf_counter() - started,
                }
                history.append(record)
                print(json.dumps(record), flush=True)
                # Rule is frozen: lowest validation reference BCE only.
                if metrics["bce"] < best_bce - 1e-6:
                    best_epoch, best_bce, stale = epoch, metrics["bce"], 0
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "kind": kind,
                            "hidden": cfg["hidden"],
                            "layers": cfg["layers"],
                            "epoch": epoch,
                            "validation_bce": best_bce,
                            "graph": graph,
                            "config": cfg,
                        },
                        directory / "best.pt",
                    )
                else:
                    stale += 1
                (directory / "history.json").write_text(
                    json.dumps(history, indent=2) + "\n"
                )
                if stale >= cfg["early_stop_patience"]:
                    break
            checkpoint = directory / "best.pt"
            if not checkpoint.exists():
                raise ValueError("No checkpoint was written")
            (directory / "complete.json").write_text(
                json.dumps(
                    {
                        "kind": kind,
                        "graph": graph,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_validation_bce": best_bce,
                        "selection_rule": "Lowest validation reference BCE",
                        "reserved_opened": False,
                        "wall_seconds": time.perf_counter() - started,
                        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / (2**30 if sys.platform == "darwin" else 2**20),
                    },
                    indent=2,
                )
                + "\n"
            )
            completed.append(directory.name)
    print("Runs complete:", completed)


if __name__ == "__main__":
    main()
