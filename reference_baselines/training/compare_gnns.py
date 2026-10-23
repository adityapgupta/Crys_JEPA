"""Matched, bounded CPU pilot: GCN, MPNN, and a no-message reference control."""

import argparse
import copy
import json
import resource
import sys
import time
import tomllib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reference_baselines.core.io import sha256
from reference_baselines.datasets.gnn_comparison import (
    OUT,
    CosineReferenceDataset,
    collate,
)
from reference_baselines.evaluation.metrics import load_adjacency, unique_order
from reference_baselines.models.message_passing import ReferenceGNN
from reference_baselines.paths import CONFIG_DIR, RESULTS_DIR, PACKAGE_DIR, ROOT


def evaluate_scores(rows, targets, scores, adjacency, probabilities=False):
    good = np.isfinite(scores)
    order = rows[good][np.lexsort((rows[good], -scores[good]))]
    unique = unique_order(order, adjacency)
    labels = pd.Series(targets, index=rows)
    result = {"count": len(rows), "score_coverage": float(good.mean())}
    for percent in [1, 2, 5, 7]:
        budget = len(rows) * percent // 100
        if budget < 1 or len(unique) < budget:
            raise ValueError("Cannot fill unique validation budget")
        passes = int(labels.loc[unique[:budget]].sum())
        result[f"passes_{percent}"] = passes
        result[f"yield_{percent}"] = passes / budget
    if probabilities:
        p = np.clip(scores, 1e-7, 1 - 1e-7)
        result["bce"] = float(
            -(targets * np.log(p) + (1 - targets) * np.log1p(-p)).mean()
        )
        result["brier"] = float(np.square(p - targets).mean())
    return result


def predict(model, loader, self_only=False):
    model.eval()
    rows, targets, probabilities = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch, self_only=self_only)
            rows.extend(batch["rows"].tolist())
            targets.extend(batch["target"].tolist())
            probabilities.extend(logits.sigmoid().tolist())
    return np.array(rows), np.array(targets), np.array(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_DIR / "gnn_comparison.toml")
    )
    args = parser.parse_args()
    cfg = tomllib.loads(open(args.config).read())
    torch.set_num_threads(cfg["threads"])
    split = json.loads((OUT / "split_protocol.json").read_text())
    start = time.perf_counter()
    train = CosineReferenceDataset("train", cfg["train_limit"], cfg["sample_seed"])
    validation = CosineReferenceDataset(
        "validation", cfg["validation_limit"], cfg["sample_seed"]
    )
    if np.intersect1d(train.rows, np.load(OUT / "reserved_rows.npy")).size:
        raise ValueError("Reserved rows entered fit set")
    paths = [
        PACKAGE_DIR / "models/message_passing.py",
        PACKAGE_DIR / "datasets/gnn_comparison.py",
        PACKAGE_DIR / "training/compare_gnns.py",
        CONFIG_DIR / "gnn_comparison.toml",
    ]
    paths += [
        ROOT / "embeddings/data" / name
        for name in ["embeddings.npy", "candidate_embeddings.npy"]
    ]
    paths += [
        RESULTS_DIR / name
        for name in [
            "candidate_composition.npy",
            "reference_composition.npy",
            "training_targets.npy",
            "candidate_cosine_indices.npy",
            "reference_cosine_graph_k10.npz",
            "chemical_reference_indptr.npy",
            "chemical_reference_indices.npy",
        ]
    ]
    protocol = {
        "settings": cfg,
        "split_counts": split["counts"],
        "reserved_evaluated": False,
        "sampled_train": len(train),
        "sampled_validation": len(validation),
        "train_rows": train.rows.tolist(),
        "validation_rows": validation.rows.tolist(),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
    }
    frozen = OUT / "pilot_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Pilot protocol changed; use a separate results directory")
    frozen.write_text(json.dumps(protocol, indent=2) + "\n")
    # Cache only this bounded pilot, not all 250k candidate graphs.
    train_items, validation_items = list(train), list(validation)
    all_items = train_items + validation_items
    stats = {
        "cache_seconds": time.perf_counter() - start,
        "mean_references_per_query": float(np.mean([len(i["x"]) for i in all_items])),
        "mean_nonself_edges": float(
            np.mean([(i["edge"][0] != i["edge"][1]).sum().item() for i in all_items])
        ),
        "queries_with_nonself_edges_fraction": float(
            np.mean([bool((i["edge"][0] != i["edge"][1]).any()) for i in all_items])
        ),
    }
    (OUT / "pilot_graph_diagnostics.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    print("Cached pilot graphs", json.dumps(stats), flush=True)
    validation_loader = DataLoader(
        validation_items, batch_size=cfg["batch_size"], collate_fn=collate
    )
    adjacency = load_adjacency()
    targets = np.array([i["target"] for i in validation_items])
    baseline_rows = []
    source = np.load(RESULTS_DIR / "mean_ablation/validation_scores.npz")
    balanced = np.load(RESULTS_DIR / "pre_gnn/validation_formula_scores.npz")
    for name, data, method in [
        ("original_mean", source, "chemical_mean"),
        ("angle_gate", source, "gate_then_chemical_mean"),
        ("formula_balanced", balanced, "formula_balanced_mean"),
    ]:
        values = pd.Series(
            data["scores"][:, data["methods"].tolist().index(method)],
            index=data["candidate_rows"],
        )
        metric = evaluate_scores(
            validation.rows, targets, -values.loc[validation.rows].to_numpy(), adjacency
        )
        baseline_rows.append({"model": name, **metric})
    pd.DataFrame(baseline_rows).to_csv(OUT / "pilot_baselines.csv", index=False)
    for seed in cfg["seeds"]:
        for kind in cfg["models"]:
            directory = OUT / "runs" / f"{kind}_seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / "complete.json").exists():
                print("Already complete:", kind, seed, flush=True)
                continue
            torch.manual_seed(seed)
            model = ReferenceGNN(kind, cfg["hidden"], cfg["layers"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                train_items,
                batch_size=cfg["batch_size"],
                shuffle=True,
                generator=generator,
                collate_fn=collate,
            )
            best, best_state, history = (-1, -np.inf), None, []
            for epoch in range(1, cfg["epochs"] + 1):
                tick = time.perf_counter()
                model.train()
                total, count = 0.0, 0
                for batch in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(batch)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, batch["target"]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 5, error_if_nonfinite=True
                    )
                    optimizer.step()
                    total += float(loss.detach()) * len(logits)
                    count += len(logits)
                rows, y, p = predict(model, validation_loader)
                metrics = evaluate_scores(rows, y, p, adjacency, probabilities=True)
                record = {
                    "epoch": epoch,
                    "train_bce": total / count,
                    "seconds": time.perf_counter() - tick,
                    **metrics,
                }
                history.append(record)
                (directory / "history.json").write_text(
                    json.dumps(history, indent=2) + "\n"
                )
                key = (metrics["yield_5"], -metrics["bce"])
                if key > best:
                    best, best_state = key, copy.deepcopy(model.state_dict())
                    torch.save(
                        {
                            "state_dict": best_state,
                            "kind": kind,
                            "settings": cfg,
                            "seed": seed,
                            "epoch": epoch,
                        },
                        directory / "best.pt",
                    )
                    pd.DataFrame(
                        {"candidate_row": rows, "probability": p, "target": y}
                    ).to_csv(directory / "validation_predictions.csv", index=False)
                print(
                    kind,
                    seed,
                    "epoch",
                    epoch,
                    "train BCE",
                    round(total / count, 4),
                    "5% yield",
                    round(metrics["yield_5"], 4),
                    "seconds",
                    round(record["seconds"], 2),
                    flush=True,
                )
            model.load_state_dict(best_state)
            rows, y, p = predict(model, validation_loader)
            final = evaluate_scores(rows, y, p, adjacency, probabilities=True)
            _, _, self_prob = predict(model, validation_loader, self_only=True)
            without_edges = evaluate_scores(
                rows, y, self_prob, adjacency, probabilities=True
            )
            summary = {
                "model": kind,
                "seed": seed,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": sum(r["seconds"] for r in history),
                "best_epoch": torch.load(directory / "best.pt", weights_only=True)[
                    "epoch"
                ],
                "reserved_evaluated": False,
                "metrics": final,
                "self_edges_only_inference": without_edges,
                "mean_absolute_edge_ablation_probability_change": float(
                    np.abs(p - self_prob).mean()
                ),
            }
            (directory / "complete.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    runtime = {
        "wall_seconds": time.perf_counter() - start,
        "peak_process_RSS_GiB": rss / (2**30 if sys.platform == "darwin" else 2**20),
        "device": "cpu",
        "threads": cfg["threads"],
        "reserved_evaluated": False,
        "note": "Pilot timing; not a measured full-data training time",
    }
    (OUT / "pilot_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(runtime), flush=True)


if __name__ == "__main__":
    main()
