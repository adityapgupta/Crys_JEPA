"""Descriptive energy variance by space group and embedding community."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import arguments, read_metadata, aligned_communities, provenance, save_json, digest
import json


def eta_squared(values, labels):
    """Fraction of variance in `values` explained by group `labels` (0..1)."""
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    ss_between = 0.0
    for g in np.unique(labels):
        v = values[labels == g]
        ss_between += len(v) * (v.mean() - grand) ** 2
    return ss_between / ss_total if ss_total > 0 else 0.0


def eta_with_chance(values, labels, rng, n_shuffle=10):
    obs = eta_squared(values, labels)
    chance = np.mean([eta_squared(values, rng.permutation(labels)) for _ in range(n_shuffle)])
    return obs, float(chance)


def main():
    args = arguments(__doc__, data=True).parse_args()
    OUT = MAP_OUT = args.out
    DATA = args.data
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = read_metadata(os.path.join(DATA, "metadata.csv"))
    n = len(meta)
    energy = meta["formation_energy_per_atom"].to_numpy(dtype=float)
    sg = meta["spacegroup.number"].to_numpy()

    cbc = aligned_communities(meta, read_metadata(os.path.join(MAP_OUT, "crystals_by_community.csv")))
    community = cbc["community_fine"].to_numpy()

    map_info = json.loads((args.out/'map_provenance.json').read_text())
    if map_info['metadata_sha256'] != digest(args.data/'metadata.csv') or map_info['xy_sha256'] != digest(args.out/'tsne_xy.npy'):
        raise ValueError('Map provenance mismatch; regenerate community map')
    xy = np.load(os.path.join(MAP_OUT, "tsne_xy.npy"))
    if xy.shape != (n, 2):
        raise ValueError('Map must match metadata row order and count')
    ok = np.isfinite(energy)
    print(f"crystals: {n} | with energy: {int(ok.sum())}", flush=True)

    q = np.quantile(energy[ok], [0, .2, .4, .6, .8, 1.0])
    bins = np.clip(np.digitize(energy, q[1:-1]), 0, 4)
    labels_txt = [f"[{q[i]:.2f}, {q[i+1]:.2f}) eV/atom" for i in range(5)]
    plt.figure(figsize=(9, 8))
    cmap = plt.get_cmap("viridis")
    for b in range(5):
        m = ok & (bins == b)
        plt.scatter(xy[m, 0], xy[m, 1], s=3, color=cmap(b / 4), label=labels_txt[b],
                    alpha=0.7, linewidths=0)
    plt.legend(title="formation energy", markerscale=3, fontsize=8, loc="upper right")
    plt.title("Our t-SNE map, coloured by formation energy")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "map_by_energy.png"), dpi=140)
    plt.close()
    print("wrote map_by_energy.png", flush=True)

    e = energy[ok]
    sg_ok, comm_ok = sg[ok], community[ok]
    print("\n== How much of formation-energy spread is explained? (observed | chance) ==", flush=True)
    statistics = {}
    for name, lab in [("space group", sg_ok), ("community (fine)", comm_ok)]:
        obs, ch = eta_with_chance(e, lab, rng)
        statistics[name] = {"eta_squared": obs, "shuffled_mean": ch}
        print(name, statistics[name], flush=True)
    print("  global energy std: %.3f eV/atom | range: [%.2f, %.2f]"
          % (e.std(), e.min(), e.max()), flush=True)

    save_json(args.out/'energy_summary.json', {
        'provenance': provenance([args.data/'metadata.csv', args.out/'crystals_by_community.csv'], seed=args.seed, shuffles=10),
        'statistics': statistics, 'interpretation': 'Descriptive variance decomposition; not independence from energy.'})

    df = pd.DataFrame({"community": comm_ok, "energy": e, "sg": sg_ok})
    g = df.groupby("community")
    top = g.size().sort_values(ascending=False).head(15).index
    rows = []
    for c in top:
        sub = df[df.community == c]
        sgv, sgc = np.unique(sub.sg.astype(str), return_counts=True)
        rows.append({
            "community": int(c), "size": len(sub),
            "top_spacegroup": sgv[sgc.argmax()],
            "sg_share": round(sgc.max() / len(sub), 2),
            "energy_mean": round(sub.energy.mean(), 2),
            "energy_std": round(sub.energy.std(), 2),
            "energy_min": round(sub.energy.min(), 2),
            "energy_max": round(sub.energy.max(), 2),
        })
    tbl = pd.DataFrame(rows)
    tbl["covers_%_of_global_range"] = (
        100 * (tbl.energy_max - tbl.energy_min) / (e.max() - e.min())).round(0)
    tbl.to_csv(os.path.join(OUT, "per_community_energy.csv"), index=False)
    print("\n== Energy spread inside the 15 biggest communities ==", flush=True)
    print(tbl.to_string(index=False), flush=True)
    print("\nglobal std %.2f -> median within-community std %.2f (ratio %.2f)"
          % (e.std(), tbl.energy_std.median(), tbl.energy_std.median() / e.std()), flush=True)
    print(f"\nDONE. Outputs in: {OUT}", flush=True)


if __name__ == "__main__":
    main()
