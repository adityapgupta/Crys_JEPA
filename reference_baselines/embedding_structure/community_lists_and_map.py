"""Per-crystal community labels and descriptive PCA/t-SNE visualizations."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import arguments, provenance, save_json, digest
from .build_graph import load_embeddings, build_knn_graph


def labels_from_communities(comms, n):
    lab = np.empty(n, dtype=np.int64)
    for cid, nodes in enumerate(comms):
        for v in nodes:
            lab[v] = cid

    order = pd.Series(lab).value_counts().index.to_list()
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[x] for x in lab], dtype=np.int64)


def summarise(labels, sg, chem, formula):
    rows = []
    for cid in np.unique(labels):
        m = labels == cid
        sgv, sgc = np.unique(sg[m].astype(str), return_counts=True)
        chv, chc = np.unique(chem[m].astype(str), return_counts=True)
        i, j = int(sgc.argmax()), int(chc.argmax())
        rows.append({
            "label": int(cid), "size": int(m.sum()),
            "top_spacegroup": sgv[i], "top_spacegroup_share": round(sgc[i] / m.sum(), 3),
            "top_chemical_system": chv[j], "top_chemical_system_share": round(chc[j] / m.sum(), 3),
            "example_formulas": ", ".join(pd.Series(formula[m]).head(5).astype(str)),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def coloured_map(xy, labels, title, path, top=12):
    """Scatter the 2D map. Colour the `top` biggest label-groups; grey the rest."""
    counts = pd.Series(labels).value_counts()
    big = counts.index[:top].to_list()
    plt.figure(figsize=(9, 8))
    rest = ~np.isin(labels, big)
    plt.scatter(xy[rest, 0], xy[rest, 1], s=2, c="lightgrey", alpha=0.4, linewidths=0)
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(big):
        m = labels == lab
        plt.scatter(xy[m, 0], xy[m, 1], s=4, color=cmap(i % 20),
                    label=str(lab), alpha=0.8, linewidths=0)
    plt.legend(title=title.split("(")[0].strip(), markerscale=3, fontsize=8,
               loc="upper right", ncol=2)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main():
    parser = arguments(__doc__, data=True)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    DATA, OUT, SEED = args.data, args.out, args.seed
    OUT.mkdir(parents=True, exist_ok=True)
    print('Building graph and loading metadata...', flush=True)
    emb, meta = load_embeddings(DATA)
    W, manifest = build_knn_graph(emb, k=args.k)
    n = len(meta)
    sg = meta["spacegroup.number"].to_numpy()
    chem = meta["chemical_system"].to_numpy()
    formula = meta["pretty_formula"].to_numpy()

    G = nx.from_scipy_sparse_array(W, edge_attribute="weight")

    print("louvain (fine, resolution=1.0) ...", flush=True)
    fine = labels_from_communities(
        louvain_communities(G, weight="weight", resolution=1.0, seed=SEED), n)
    print(f"  fine communities: {fine.max() + 1}", flush=True)

    print("louvain (coarse, resolution=0.1) -> 'regions' ...", flush=True)
    region = labels_from_communities(
        louvain_communities(G, weight="weight", resolution=0.1, seed=SEED), n)
    print(f"  coarse regions: {region.max() + 1}", flush=True)

    out = pd.DataFrame({
        "source_row": np.arange(n),
        "num_id": meta.get("num_id", pd.Series(range(n))),
        "material_id": meta["material_id"],
        "pretty_formula": formula,
        "spacegroup_number": sg,
        "chemical_system": chem,
        "community_fine": fine,
        "region_coarse": region,
    }).sort_values(["community_fine", "num_id"])
    out.to_csv(os.path.join(OUT, "crystals_by_community.csv"), index=False)
    print(f"wrote crystals_by_community.csv ({len(out)} rows)", flush=True)

    summarise(fine, sg, chem, formula).to_csv(
        os.path.join(OUT, "community_summary.csv"), index=False)
    summarise(region, sg, chem, formula).to_csv(
        os.path.join(OUT, "region_summary.csv"), index=False)
    print("wrote community_summary.csv + region_summary.csv", flush=True)

    print("projecting to 2D (PCA-50 -> t-SNE) ... this is the slow step", flush=True)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x50 = PCA(n_components=min(50, emb.shape[1], n), random_state=SEED).fit_transform(emb)
    xy = TSNE(n_components=2, init="pca", perplexity=min(30, n-1), random_state=SEED,
              verbose=1).fit_transform(x50)
    np.save(os.path.join(OUT, "tsne_xy.npy"), xy)
    save_json(OUT/'map_provenance.json', {
        'metadata_sha256': digest(DATA/'metadata.csv'), 'xy_sha256': digest(OUT/'tsne_xy.npy'),
        'provenance': provenance([DATA/'embeddings.npy', DATA/'metadata.csv'], seed=SEED, k=args.k),
        'fine_communities': int(fine.max()+1), 'lower_resolution_communities': int(region.max()+1),
        'graph': {key: value for key, value in manifest.items() if key != 'node_index'},
        'row_order': 'metadata.csv row order', 'limits': 'PCA/t-SNE visualization; not a test of structural independence.'})

    coloured_map(xy, sg.astype(np.int64),
                 "2D embedding map, coloured by space group (top 12 shown)",
                 os.path.join(OUT, "map_by_spacegroup.png"))
    coloured_map(xy, fine,
                 "2D embedding map, coloured by community (12 largest shown)",
                 os.path.join(OUT, "map_by_community.png"))
    print(f"\nDONE. Everything in: {OUT}", flush=True)


if __name__ == "__main__":
    main()
