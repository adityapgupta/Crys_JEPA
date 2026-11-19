"""Can the Crys-JEPA node embeddings recover the relational edge features?

This probe tests the premise behind the relational-edge experiment: each edge feature
(SOAP distance, co-occurrence PMI, hull compatibility) is *sourced* independently of the
node embeddings, but that does not prove the embeddings cannot already reproduce it. Here
we try to predict each edge feature from the two endpoint embeddings. Low recoverability
(R^2 / AUC near the trivial baseline) upgrades the claim from "independent by
construction" to "verified non-redundant"; high recoverability means the feature is
redundant with information the model already had, which would itself explain the null.

Design choices that keep the probe honest:
  * **Symmetric inputs.** The features are symmetric in (i, j), so the probe input is
    ``[emb_i + emb_j, |emb_i - emb_j|]`` -- it cannot exploit edge direction.
  * **Reference-disjoint split.** References are split into train / test *first*; a probe
    edge is a test edge only when BOTH endpoints are test references. No test reference's
    embedding is ever seen in training, so R^2 measures genuine generalisation, not
    memorisation of a reference seen from its other edges.
  * **Undirected, no self-loops.** Each physical edge is counted once (i < j); self-loops
    are dropped (degenerate symmetric input).

Torch-free: numpy + scipy only, so it runs on a login node or in any plain venv. Only the
feature inputs that are present on disk are probed (hull alone runs without the SOAP /
frequency precompute).

Run:  python -I -m reference_baselines.evaluation.embedding_recoverability_probe
"""

import argparse
import json

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from reference_baselines.paths import RESULTS_DIR, ROOT

REFERENCE_COUNT = 27136
PMI_CLIP = 5.0

EMBEDDINGS_PATH = ROOT / "embeddings/data/embeddings.npy"
PHYSICAL_PATH = RESULTS_DIR / "reference_gnn/graphs/physical.npz"
SOAP_PATH = RESULTS_DIR / "soap_density_normalized.npy"
FREQUENCY_PATH = RESULTS_DIR / "relational_edge/reference_decomp_frequency.npy"
NORMALISER_PATH = RESULTS_DIR / "relational_edge/cooccurrence_normaliser.npy"
HULL_PATH = RESULTS_DIR / "relational_edge/hull_compatibility.npz"


def undirected_edges():
    """Unique undirected physical edges (i < j) with their co-occurrence counts."""
    graph = sparse.load_npz(PHYSICAL_PATH).tocoo()
    i, j, weight = graph.row.astype(np.int64), graph.col.astype(np.int64), graph.data
    keep = i < j  # one direction per pair, drops self-loops
    return i[keep], j[keep], np.expm1(weight[keep])  # physical weight is log1p(count)


def soap_targets(i, j):
    soap = np.nan_to_num(np.load(SOAP_PATH).astype(np.float64), nan=0.0)
    norm = np.maximum(np.linalg.norm(soap, axis=1, keepdims=True), 1e-12)
    soap = soap / norm
    dot = np.sum(soap[i] * soap[j], axis=1)
    return 1.0 - np.clip(dot, -1.0, 1.0)


def pmi_targets(i, j, counts):
    frequency = np.load(FREQUENCY_PATH).astype(np.float64)
    normaliser = float(np.load(NORMALISER_PATH))
    ci = np.maximum(frequency[i], 1.0)
    cj = np.maximum(frequency[j], 1.0)
    ratio = (normaliser * counts) / (ci * cj)
    pmi = np.log(np.maximum(ratio, 1e-6))
    return np.clip(pmi, -PMI_CLIP, PMI_CLIP) / PMI_CLIP


def hull_targets(i, j):
    hull = sparse.load_npz(HULL_PATH).tocsr()
    return np.asarray(hull[i, j]).ravel().astype(np.float64)


def symmetric_inputs(embeddings, i, j):
    ei, ej = embeddings[i], embeddings[j]
    return np.concatenate([ei + ej, np.abs(ei - ej)], axis=1)


def ridge_fit_predict(x_train, y_train, x_test, lam):
    """Closed-form ridge on standardised inputs; returns test predictions."""
    mean, std = x_train.mean(0), x_train.std(0)
    std = np.where(std < 1e-8, 1.0, std)
    xt = (x_train - mean) / std
    xv = (x_test - mean) / std
    ybar = y_train.mean()
    yc = y_train - ybar
    gram = xt.T @ xt + lam * np.eye(xt.shape[1])
    w = np.linalg.solve(gram, xt.T @ yc)
    return xv @ w + ybar


def r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def auc(y_true, score):
    pos = y_true > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def reference_disjoint_split(i, j, test_fraction, rng):
    refs = np.unique(np.concatenate([i, j]))
    rng.shuffle(refs)
    cut = int(len(refs) * (1.0 - test_fraction))
    is_test = np.zeros(REFERENCE_COUNT, dtype=bool)
    is_test[refs[cut:]] = True
    test = is_test[i] & is_test[j]
    train = (~is_test[i]) & (~is_test[j])  # cross-split edges are discarded
    return train, test


def probe_feature(name, inputs, target, i, j, binary, splits, test_fraction, lam, seed):
    scores = []
    sizes = []
    for s in range(splits):
        rng = np.random.default_rng(seed + s)
        train, test = reference_disjoint_split(i, j, test_fraction, rng)
        if train.sum() < 50 or test.sum() < 50:
            continue
        pred = ridge_fit_predict(inputs[train], target[train], inputs[test], lam)
        scores.append(auc(target[test], pred) if binary else r2(target[test], pred))
        sizes.append((int(train.sum()), int(test.sum())))
    metric = "auc" if binary else "r2"
    record = {
        "feature": name,
        "metric": metric,
        f"{metric}_mean": float(np.mean(scores)) if scores else float("nan"),
        f"{metric}_std": float(np.std(scores)) if scores else float("nan"),
        "splits_used": len(scores),
        "train_edges": sizes[0][0] if sizes else 0,
        "test_edges": sizes[0][1] if sizes else 0,
    }
    if binary:
        record["positive_rate"] = float(target.mean())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default=str(RESULTS_DIR / "relational_edge_reference_gnn/recoverability_probe.json"))
    args = parser.parse_args()

    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r").astype(np.float64)
    if len(embeddings) != REFERENCE_COUNT:
        raise ValueError(f"Embeddings have {len(embeddings)} rows, expected {REFERENCE_COUNT}")
    i, j, counts = undirected_edges()
    inputs = symmetric_inputs(embeddings, i, j)

    plan = []
    if SOAP_PATH.exists():
        plan.append(("soap_distance", soap_targets(i, j), False))
    if FREQUENCY_PATH.exists() and NORMALISER_PATH.exists():
        plan.append(("cooccurrence_pmi", pmi_targets(i, j, counts), False))
    if HULL_PATH.exists():
        plan.append(("hull_compatibility", hull_targets(i, j), True))
    if not plan:
        raise FileNotFoundError("No edge-feature arrays found to probe")

    records = [
        probe_feature(name, inputs, target, i, j, binary,
                      args.splits, args.test_fraction, args.ridge_lambda, args.seed)
        for name, target, binary in plan
    ]
    summary = {
        "undirected_edges": int(len(i)),
        "embedding_dim": int(embeddings.shape[1]),
        "input_dim": int(inputs.shape[1]),
        "test_fraction": args.test_fraction,
        "ridge_lambda": args.ridge_lambda,
        "split": "reference-disjoint (both endpoints held out)",
        "features": records,
        "interpretation": "Higher R^2/AUC => feature more recoverable from embeddings "
                           "=> more redundant. Near-zero R^2 (or AUC near positive_rate-"
                           "equivalent 0.5) => genuinely non-redundant.",
    }
    out = args.out
    import pathlib
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'feature':20s} {'metric':6s} {'mean':>8s} {'std':>7s}  edges(tr/te)")
    for r in records:
        m = r["metric"]
        print(f"{r['feature']:20s} {m:6s} {r[f'{m}_mean']:8.3f} {r[f'{m}_std']:7.3f}  "
              f"{r['train_edges']}/{r['test_edges']}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
