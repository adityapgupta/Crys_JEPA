"""Train the fixed-topology edge-attribute reference GNN experiment.

Candidate stability labels and reserved rows are never opened here.  All
variants use the original reference pools, decomposition targets and physical
graph; only their use of edge attributes differs.
"""

import argparse
import json
from pathlib import Path
import time
import tomllib

import torch
from torch.utils.data import DataLoader

from reference_baselines.core.io import sha256
from reference_baselines.datasets.edge_attribute_reference_graph import (
    EDGE_FEATURES,
    EdgeAttributePoolDataset,
    edge_attribute_collate,
)
from reference_baselines.models.reference_gnn import reference_loss
from reference_baselines.models.edge_attribute_reference_gnn import VARIANTS, build_model
from reference_baselines.paths import CONFIG_DIR, PACKAGE_DIR, RESULTS_DIR, ROOT
from reference_baselines.training.train_reference_gnn import reference_metrics, seed_everything


EXPERIMENT = RESULTS_DIR / "edge_attribute_reference_gnn"


def run_specs(cfg):
    specs = []
    for variant in cfg["variants"]:
        kinds = ["readout"] if variant == "readout" else cfg["models"]
        specs.extend((variant, kind) for kind in kinds)
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_DIR / "edge_attribute_reference_gnn.toml"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "readout"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    if cfg["graph"] != "physical":
        parser.error("This experiment fixes the physical graph topology")
    if tuple(cfg["edge_features"]) != EDGE_FEATURES:
        parser.error("Config edge features do not match the implementation")
    specs = run_specs(cfg)
    if args.variant:
        specs = [spec for spec in specs if spec[0] == args.variant]
    if args.model:
        specs = [spec for spec in specs if spec[1] == args.model]
    if not specs:
        parser.error("The requested variant/model combination is not defined")
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    torch.set_num_threads(min(cfg["threads"], torch.get_num_threads()))
    device = torch.device(args.device or cfg["device"])
    train_limit = 512 if args.smoke else cfg["train_limit"]
    validation_limit = 256 if args.smoke else cfg["validation_limit"]
    epochs = 1 if args.smoke else cfg["epochs"]
    root = Path(args.run_dir) if args.run_dir else EXPERIMENT / ("smoke" if args.smoke else "runs")

    protocol = {
        "settings": cfg,
        "smoke": args.smoke,
        "train_limit": train_limit,
        "validation_limit": validation_limit,
        "reserved_opened": False,
        "candidate_stability_targets": False,
        "fixed_components": "splits, pools, nodes, query features, physical topology, decomposition targets",
        "source_sha256": {
            str(path.relative_to(PACKAGE_DIR.parent)): sha256(path)
            for path in [
                PACKAGE_DIR / "datasets/reference_graph.py",
                PACKAGE_DIR / "datasets/edge_attribute_reference_graph.py",
                PACKAGE_DIR / "models/reference.py",
                PACKAGE_DIR / "models/reference_gnn.py",
                PACKAGE_DIR / "models/edge_attribute_reference_gnn.py",
                PACKAGE_DIR / "training/train_reference_gnn.py",
                PACKAGE_DIR / "training/train_edge_attribute_reference_gnn.py",
                Path(args.config).resolve(),
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
                RESULTS_DIR / "reference_composition.npy",
                RESULTS_DIR / "reference_gnn/graphs/physical.npz",
                ROOT / "embeddings/data/metadata.csv",
            ]
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "train_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Training protocol changed; use a new run directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    loaders = {}
    edge_modes = {"shuffled" if variant == "shuffled" else "rich" for variant, _ in specs}
    for edge_mode in edge_modes:
        train = EdgeAttributePoolDataset(
            "train", graph="physical", k=cfg["retrieval_k"], limit=train_limit,
            seed=cfg["seeds"][0], edge_mode=edge_mode, edge_seed=cfg["edge_seed"],
        )
        validation = EdgeAttributePoolDataset(
            "validation", graph="physical", k=cfg["retrieval_k"], limit=validation_limit,
            seed=cfg["seeds"][0], edge_mode=edge_mode, edge_seed=cfg["edge_seed"],
        )
        loaders[(edge_mode, "train")] = DataLoader(
            train, batch_size=cfg["batch_size"], shuffle=True,
            collate_fn=edge_attribute_collate, num_workers=0,
        )
        loaders[(edge_mode, "validation")] = DataLoader(
            validation, batch_size=cfg["batch_size"], shuffle=False,
            collate_fn=edge_attribute_collate, num_workers=0,
        )

    completed = []
    for seed in seeds:
        for variant, kind in specs:
            name = f"{variant}_{kind}_physical_seed{seed}"
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                completed.append(name)
                continue
            seed_everything(seed)
            model = build_model(variant, kind, cfg["hidden"], cfg["layers"]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
            )
            edge_mode = "shuffled" if variant == "shuffled" else "rich"
            train_loader = loaders[(edge_mode, "train")]
            validation_loader = loaders[(edge_mode, "validation")]
            best_epoch, best_bce, stale = -1, float("inf"), 0
            history = []
            started = time.perf_counter()
            for epoch in range(1, epochs + 1):
                model.train()
                running, seen = 0.0, 0
                for batch in train_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
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
                metrics = reference_metrics(model, validation_loader, device)
                record = {
                    "epoch": epoch,
                    "train_bce": running / max(seen, 1),
                    "validation_bce": metrics["bce"],
                    "validation_brier": metrics["brier"],
                    "seconds": time.perf_counter() - started,
                }
                history.append(record)
                print(name, json.dumps(record), flush=True)
                if metrics["bce"] < best_bce - 1e-6:
                    best_epoch, best_bce, stale = epoch, metrics["bce"], 0
                    torch.save(
                        {
                            "state_dict": model.state_dict(), "variant": variant, "kind": kind,
                            "hidden": cfg["hidden"], "layers": cfg["layers"], "epoch": epoch,
                            "validation_bce": best_bce, "graph": "physical", "config": cfg,
                        },
                        directory / "best.pt",
                    )
                else:
                    stale += 1
                (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                if stale >= cfg["early_stop_patience"]:
                    break
            if not (directory / "best.pt").exists():
                raise ValueError("No checkpoint was written")
            (directory / "complete.json").write_text(
                json.dumps(
                    {
                        "variant": variant, "kind": kind, "seed": seed,
                        "best_epoch": best_epoch, "best_validation_bce": best_bce,
                        "selection_rule": cfg["checkpoint_rule"], "reserved_opened": False,
                        "wall_seconds": time.perf_counter() - started,
                    },
                    indent=2,
                ) + "\n"
            )
            completed.append(name)
    print("Runs complete:", completed)


if __name__ == "__main__":
    main()
