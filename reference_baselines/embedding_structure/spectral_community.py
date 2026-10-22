"""Descriptive community alignment with symmetry and chemical-system labels."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import OUTPUT_DIR, provenance


def banner(msg):
    print(f"\n{msg}", flush=True)


def _json_default(o):
    """Make numpy types JSON-serialisable (the builder's manifest holds arrays)."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def spectral_gap(W, n_eig=30):
    """Optional normalized-Laplacian diagnostic; no cluster-count inference."""
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    n = W.shape[0]
    degree = np.asarray(W.sum(1)).ravel()
    if n < 3 or np.any(degree <= 0):
        return None, None, 'Spectrum requires at least three nonisolated nodes'
    L = sp.eye(n, format='csr') - sp.diags(1 / np.sqrt(degree)) @ W @ sp.diags(1 / np.sqrt(degree))
    try:
        rng = np.random.default_rng(42)
        top, vectors = eigsh(2 * sp.eye(n) - L, k=min(n_eig, n-2), which='LA',
                             maxiter=3000, tol=1e-12, v0=rng.normal(size=n))
    except ArpackNoConvergence:
        return None, None, 'Not converged; spectrum omitted'
    vals = 2 - top
    order = np.argsort(vals)
    vals, vectors = vals[order], vectors[:, order]
    residuals = np.linalg.norm(L @ vectors - vectors * vals, axis=0)
    orthogonality = float(np.linalg.norm(vectors.T @ vectors - np.eye(len(vals))))
    diagnostics = {'residual_norms': residuals.tolist(), 'orthogonality_error': orthogonality}
    if residuals.max() > 1e-9 or orthogonality > 1e-7 or vals.min() < -1e-9:
        return None, diagnostics, 'Residual/orthogonality check failed; spectrum omitted'
    return vals, diagnostics, 'Residual-checked truncated spectrum; small gaps still require sensitivity analysis'


def louvain_labels(W, seed, resolution=1.0):
    """Louvain community labels (one int per node) on the weighted graph."""
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    G = nx.from_scipy_sparse_array(W, edge_attribute="weight")
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    labels = np.empty(W.shape[0], dtype=np.int64)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            labels[v] = cid
    return labels


def alignment_row(comm, target, rng, n_shuffle=20):
    """AMI/NMI/ARI of communities vs a label, with a shuffled-label chance control."""
    from sklearn.metrics import (adjusted_mutual_info_score,
                                  adjusted_rand_score,
                                  normalized_mutual_info_score)
    obs = {
        "AMI": adjusted_mutual_info_score(comm, target),
        "NMI": normalized_mutual_info_score(comm, target),
        "ARI": adjusted_rand_score(comm, target),
    }
    sh = {"AMI": [], "NMI": [], "ARI": []}
    for _ in range(n_shuffle):
        perm = rng.permutation(target)
        sh["AMI"].append(adjusted_mutual_info_score(comm, perm))
        sh["NMI"].append(normalized_mutual_info_score(comm, perm))
        sh["ARI"].append(adjusted_rand_score(comm, perm))
    return {m: {"observed": float(obs[m]),
                "shuffled_mean": float(np.mean(sh[m])),
                "shuffled_std": float(np.std(sh[m]))} for m in obs}


def describe_communities(labels, sg, chem, top=10):
    """For the largest communities: size + dominant space group / chemistry + purity."""
    out = []
    ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1][:top]
    for i in order:
        cid = ids[i]
        m = labels == cid
        row = {"community": int(cid), "size": int(counts[i])}
        for name, arr in [("spacegroup", sg), ("chemical_system", chem)]:
            vals = arr[m]
            vals = vals[vals != None]
            if len(vals):
                u, c = np.unique(vals.astype(str), return_counts=True)
                j = int(np.argmax(c))
                row[f"top_{name}"] = u[j]
                row[f"top_{name}_share"] = round(float(c[j]) / len(vals), 3)
            else:
                row[f"top_{name}"] = "n/a"
                row[f"top_{name}_share"] = 0.0
        out.append(row)
    return out


def run_variant(name, normalize, emb, sg, chem, labeled, k, seed, out_dir, rng, spectrum=False):
    """Full pipeline for one graph variant (euclidean / cosine)."""
    from .build_graph import build_knn_graph

    banner(f"VARIANT: {name}  (normalize={normalize})")
    t0 = time.perf_counter()
    W, manifest = build_knn_graph(emb, k=k, normalize=normalize, symmetrize="union")
    manifest = {kk: vv for kk, vv in manifest.items() if kk != "node_index"}
    print("graph:", json.dumps(manifest, indent=2, default=_json_default), flush=True)
    sp.save_npz(os.path.join(out_dir, f"graph_{name}_k{k}.npz"), W)

    vals, diagnostics, note = spectral_gap(W) if spectrum else (None, None, 'Not requested; no spectral claim')
    print(f"\n[spectral] {note}")
    if vals is not None:
        print(f"[spectral] smallest eigenvalues: "
              f"{np.array2string(vals[:12], precision=4, floatmode='fixed')}")

    seeds = [seed, seed + 1, seed + 2]
    runs = [louvain_labels(W, s) for s in seeds]
    labels = runs[0]
    n_comm = len(np.unique(labels))
    sizes = np.sort(np.bincount(labels))[::-1]
    from sklearn.metrics import adjusted_mutual_info_score
    stab = [adjusted_mutual_info_score(runs[0], runs[i]) for i in (1, 2)]
    print(f"\n[louvain] communities (seed {seed}): {n_comm}")
    print(f"[louvain] seed stability AMI vs seeds {seeds[1:]}: "
          f"{[round(s, 3) for s in stab]}")
    print(f"[louvain] community count across seeds: "
          f"{[len(np.unique(r)) for r in runs]}")
    print(f"[louvain] top 10 community sizes: {sizes[:10].tolist()}")
    print(f"[louvain] top community = {sizes[0] / len(labels):.1%} of all nodes")

    desc = describe_communities(labels, sg, chem, top=10)
    banner("Largest communities: dominant structure vs chemistry")
    print(f"{'comm':>5} {'size':>7} {'top space grp':>14} {'share':>6} "
          f"{'top chem system':>20} {'share':>6}")
    for r in desc:
        print(f"{r['community']:>5} {r['size']:>7} {r['top_spacegroup']:>14} "
              f"{r['top_spacegroup_share']:>6} {r['top_chemical_system']:>20} "
              f"{r['top_chemical_system_share']:>6}")

    comm_lab = labels[labeled]
    sg_lab = sg[labeled].astype(str)
    chem_lab = chem[labeled].astype(str)
    align = {
        "spacegroup": alignment_row(comm_lab, sg_lab, rng),
        "chemical_system": alignment_row(comm_lab, chem_lab, rng),
    }
    banner("ALIGNMENT: communities vs structure vs chemistry  (observed | chance)")
    print(f"{'metric':>5}  {'vs space group':>28}  {'vs chemical system':>28}")
    for m in ("AMI", "NMI", "ARI"):
        a = align["spacegroup"][m]
        b = align["chemical_system"][m]
        print(f"{m:>5}  {a['observed']:>10.3f} (chance {a['shuffled_mean']:>6.3f})  "
              f"{b['observed']:>10.3f} (chance {b['shuffled_mean']:>6.3f})")
    winner = ("STRUCTURE (space group)"
              if align["spacegroup"]["AMI"]["observed"] > align["chemical_system"]["AMI"]["observed"]
              else "CHEMISTRY (chemical system)")
    print(f"\n==> communities align more with: {winner}   (by AMI)")

    if vals is not None:
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(vals)), vals, "o-", ms=3)
        plt.xlabel("index"); plt.ylabel("eigenvalue")
        plt.title(f"Normalised-Laplacian spectrum ({name}, k={k})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"spectrum_{name}.png"), dpi=140)
        plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(sizes[:30])), sizes[:30])
    plt.xlabel("community (rank by size)"); plt.ylabel("number of crystals")
    plt.title(f"Community sizes ({name}) — {n_comm} communities")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"community_sizes_{name}.png"), dpi=140)
    plt.close()

    result = {
        "variant": name, "normalize": normalize, "k": k,
        "graph": manifest,
        "spectral": {"note": note,
                     "eigenvalues": None if vals is None else vals.tolist(),
                     "diagnostics": diagnostics},
        "louvain": {"seed": seed, "n_communities": int(n_comm),
                    "seed_stability_AMI": [float(s) for s in stab],
                    "n_communities_per_seed": [int(len(np.unique(r))) for r in runs],
                    "top_sizes": sizes[:10].tolist(),
                    "largest_fraction": float(sizes[0] / len(labels))},
        "communities": desc,
        "alignment": align,
        "alignment_winner_by_AMI": winner,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"\n[{name}] done in {result['seconds']}s")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dir with embeddings.npy + metadata.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--spectrum", action="store_true", help="Optional residual-checked spectral diagnostic")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from .build_graph import load_embeddings
    banner("STEP 0 — load embeddings + metadata")
    emb, meta = load_embeddings(data_dir=args.data)
    print("embeddings:", emb.shape, emb.dtype)
    if meta is None:
        sys.exit("metadata.csv not found / pandas missing — cannot run alignment.")
    for col in ("spacegroup.number", "chemical_system"):
        if col not in meta.columns:
            sys.exit(f"metadata is missing required column '{col}'. "
                     f"columns present: {list(meta.columns)}")

    sg = meta["spacegroup.number"].to_numpy()
    chem = meta["chemical_system"].to_numpy()
    labeled = meta["spacegroup.number"].notna().to_numpy() & meta["chemical_system"].notna().to_numpy()
    print(f"nodes: {len(meta)}  |  labelled (both present): {int(labeled.sum())}")
    print(f"distinct space groups : {meta['spacegroup.number'].nunique()}")
    print(f"distinct chem systems : {meta['chemical_system'].nunique()}")
    print("NOTE: chemical_system has FAR more categories than space group, so raw")
    print("      NMI can be inflated by its many categories. AMI corrects for this; we also")
    print("      report each metric against a shuffled-label chance baseline.")

    results = {}
    for name, normalize in [("euclidean", False), ("cosine", True)]:
        results[name] = run_variant(name, normalize, emb, sg, chem, labeled,
                                     args.k, args.seed, args.out, rng, args.spectrum)

    banner("HEADLINE — AMI of communities vs structure / chemistry")
    print(f"{'variant':>10}  {'vs space group':>16}  {'vs chem system':>16}  {'winner':>26}")
    for name in results:
        a = results[name]["alignment"]["spacegroup"]["AMI"]["observed"]
        b = results[name]["alignment"]["chemical_system"]["AMI"]["observed"]
        print(f"{name:>10}  {a:>16.3f}  {b:>16.3f}  {results[name]['alignment_winner_by_AMI']:>26}")

    results['provenance'] = provenance([os.path.join(args.data, 'embeddings.npy'),
                                        os.path.join(args.data, 'metadata.csv')],
                                       k=args.k, seed=args.seed, spectrum=args.spectrum)
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    print(f"\nSaved JSON + figures + graphs to: {args.out}")


if __name__ == "__main__":
    main()
