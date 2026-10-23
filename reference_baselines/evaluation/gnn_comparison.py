"""Evaluate fixed pilot checkpoints on full validation; never open reserved rows."""

import json
import time
import tomllib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.datasets.gnn_comparison import (
    OUT,
    CosineReferenceDataset,
    collate,
)
from reference_baselines.evaluation.metrics import load_adjacency, unique_order
from reference_baselines.evaluation.nonparametric import paired_interval
from reference_baselines.models.message_passing import ReferenceGNN
from reference_baselines.paths import CONFIG_DIR, RESULTS_DIR
from reference_baselines.training.compare_gnns import evaluate_scores


def main():
    cfg = tomllib.loads((CONFIG_DIR / "gnn_comparison.toml").read_text())
    torch.set_num_threads(cfg["threads"])
    models, records = {}, []
    for seed in cfg["seeds"]:
        for kind in cfg["models"]:
            path = OUT / "runs" / f"{kind}_seed{seed}" / "best.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = ReferenceGNN(kind, cfg["hidden"], cfg["layers"]).eval()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            models[(kind, seed)] = model
            records.append(
                {
                    "model": kind,
                    "seed": seed,
                    "epoch": checkpoint["epoch"],
                    "sha256": sha256(path),
                }
            )
    dataset = CosineReferenceDataset("validation")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], collate_fn=collate)
    predictions = {(kind, seed, "full_graph"): [] for kind, seed in models}
    predictions.update(
        {
            (kind, seed, "self_edges_only"): []
            for kind, seed in models
            if kind != "no_graph"
        }
    )
    rows, targets = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            rows.extend(batch["rows"].tolist())
            targets.extend(batch["target"].tolist())
            for (kind, seed), model in models.items():
                logits, _ = model(batch)
                predictions[(kind, seed, "full_graph")].extend(
                    logits.sigmoid().tolist()
                )
                if kind != "no_graph":
                    logits, _ = model(batch, self_only=True)
                    predictions[(kind, seed, "self_edges_only")].extend(
                        logits.sigmoid().tolist()
                    )
            if (i + 1) % 200 == 0:
                print("Full validation:", len(rows), "/", len(dataset), flush=True)
    rows, targets = np.array(rows), np.array(targets)
    np.testing.assert_array_equal(rows, dataset.rows)
    adjacency = load_adjacency()
    metrics, selections = [], {}
    for (kind, seed, edges), probabilities in predictions.items():
        p = np.array(probabilities)
        record = {
            "model": kind,
            "seed": seed,
            "edges": edges,
            **evaluate_scores(rows, targets, p, adjacency, probabilities=True),
        }
        metrics.append(record)
        pd.DataFrame(
            {"candidate_row": rows, "probability": p, "target": targets}
        ).to_csv(OUT / f"full_validation_{kind}_seed{seed}_{edges}.csv", index=False)
        order = rows[np.lexsort((rows, -p))]
        selections[(kind, seed, edges)] = unique_order(order, adjacency)[
            : len(rows) * 5 // 100
        ]
    original = np.load(RESULTS_DIR / "mean_ablation/validation_scores.npz")
    balanced = np.load(RESULTS_DIR / "pre_gnn/validation_formula_scores.npz")
    baselines = []
    for name, data, method in [
        ("original_mean", original, "chemical_mean"),
        ("angle_gate", original, "gate_then_chemical_mean"),
        ("formula_balanced", balanced, "formula_balanced_mean"),
    ]:
        source = pd.Series(
            data["scores"][:, data["methods"].tolist().index(method)],
            index=data["candidate_rows"],
        )
        baselines.append(
            {
                "model": name,
                **evaluate_scores(
                    rows, targets, -source.loc[rows].to_numpy(), adjacency
                ),
            }
        )
    pd.DataFrame(metrics).to_csv(OUT / "full_validation_metrics.csv", index=False)
    pd.DataFrame(baselines).to_csv(OUT / "full_validation_baselines.csv", index=False)
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    y = np.zeros(len(manifest), dtype=int)
    y[rows] = targets.astype(int)
    intervals = []
    for seed in cfg["seeds"]:
        for kind in ["gcn", "mpnn"]:
            selected = selections[(kind, seed, "full_graph")]
            control = selections[("no_graph", seed, "full_graph")]
            low, high = paired_interval(
                selected, control, manifest, y, seed=20261022, draws=1000
            )
            intervals.append(
                {
                    "model": kind,
                    "seed": seed,
                    "comparator": "no_graph",
                    "extra_passes_5": int(y[selected].sum() - y[control].sum()),
                    "gain_pp": float(100 * (y[selected].mean() - y[control].mean())),
                    "ci_low_pp": low * 100,
                    "ci_high_pp": high * 100,
                }
            )
    pd.DataFrame(intervals).to_csv(OUT / "full_validation_intervals.csv", index=False)
    pairs = read_csv(RESULTS_DIR / "pre_gnn/same_formula_pairs.csv")
    pairs = pairs[pairs.split.eq("validation") & pairs.different_stability_labels]
    pair_metrics = []
    for (kind, seed, edges), probabilities in predictions.items():
        if edges != "full_graph":
            continue
        scores = pd.Series(probabilities, index=rows)
        delta = (
            scores.loc[pairs.candidate_a].to_numpy()
            - scores.loc[pairs.candidate_b].to_numpy()
        )
        energy_delta = pairs.hull_a.to_numpy() - pairs.hull_b.to_numpy()
        available = np.isfinite(delta) & (delta != 0)
        accuracy = delta[available] * energy_delta[available] < 0
        pair_metrics.append(
            {
                "model": kind,
                "seed": seed,
                "pairs": int(available.sum()),
                "accuracy": float(accuracy.mean()),
            }
        )
    pd.DataFrame(pair_metrics).to_csv(
        OUT / "full_validation_same_formula.csv", index=False
    )
    report = {
        "validation_rows": len(rows),
        "train_sample": cfg["train_limit"],
        "pilot_validation_sample": cfg["validation_limit"],
        "full_validation_seconds": time.perf_counter() - started,
        "further_optimizer_steps": 0,
        "reserved_evaluated": False,
        "legacy_test_evaluated": False,
        "selection": "Checkpoints already chosen on pilot validation; no additional fitting or checkpoint changes",
        "inference_ablation": "Self-edges-only changes a trained checkpoint's graph; it is not a separately trained control",
        "limits": "Small training sample, five epochs; development validation only; bootstrap intervals unadjusted",
        "checkpoints": records,
    }
    (OUT / "full_validation_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        pd.DataFrame(metrics)
        .query("edges == 'full_graph'")[["model", "seed", "passes_5", "yield_5", "bce"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
