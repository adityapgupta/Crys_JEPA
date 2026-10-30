"""Space-group concentration with a size-matched permutation null."""
import numpy as np
import pandas as pd
from .common import arguments, read_metadata, provenance, save_json


def crystal_system(g):
    return ('triclinic' if g <= 2 else 'monoclinic' if g <= 15 else
            'orthorhombic' if g <= 74 else 'tetragonal' if g <= 142 else
            'trigonal' if g <= 167 else 'hexagonal' if g <= 194 else 'cubic')


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def permutation_indices(size, rng, blocks=None):
    if blocks is None:
        return rng.permutation(size)
    order = np.arange(size)
    for rows in blocks:
        order[rows] = rng.permutation(rows)
    return order


def concentration(sg, communities, draws=1000, seed=2026, minimum=50, strata=None):
    if draws < 1 or minimum < 1:
        raise ValueError('draws and minimum must be positive')
    sg = np.asarray(sg)
    communities = np.asarray(communities)
    if sg.ndim != 1 or len(sg) == 0 or sg.shape != communities.shape:
        raise ValueError('Nonempty aligned one-dimensional label arrays required')
    blocks = None
    if strata is not None:
        strata = np.asarray(strata)
        if strata.shape != sg.shape:
            raise ValueError('Strata must align with labels')
        blocks = [np.flatnonzero(strata == b) for b in np.unique(strata)]
    groups, si = np.unique(sg, return_inverse=True)
    _, ci = np.unique(communities, return_inverse=True)
    nc = int(ci.max()) + 1
    sizes = np.bincount(si)
    def counts(c):
        return np.bincount(si * nc + c, minlength=len(groups)*nc).reshape(len(groups), nc)
    def top_share(ct):
        return np.sort(ct, axis=1)[:, -min(5, nc):].sum(axis=1) / sizes
    observed_counts = counts(ci)
    observed = top_share(observed_counts)
    rng = np.random.default_rng(seed)
    # Permuting community assignments is equivalent to permuting space-group labels.
    null = np.array([top_share(counts(ci[permutation_indices(len(ci), rng, blocks)]))
                     for _ in range(draws)])
    keep = sizes >= minimum
    if not keep.any():
        raise ValueError('No space groups meet the minimum sample size')
    mean = null.mean(axis=0)
    p = (1 + (null >= observed).sum(axis=0)) / (draws + 1)
    df = pd.DataFrame({
        'space_group': groups[keep],
        'crystal_system': [crystal_system(g) for g in groups[keep]],
        'members': sizes[keep], 'top5_share': observed[keep],
        'null_top5_mean': mean[keep],
        'null_top5_q025': np.quantile(null, .025, axis=0)[keep],
        'null_top5_q975': np.quantile(null, .975, axis=0)[keep],
        'enrichment': (observed / mean)[keep],
        'permutation_p': p[keep], 'bh_q': bh_adjust(p[keep]),
        'eff_communities': (1 / ((observed_counts / sizes[:, None])**2).sum(axis=1))[keep],
    })
    return df.sort_values('enrichment', ascending=False).reset_index(drop=True)


def main():
    parser = arguments(__doc__)
    parser.set_defaults(seed=2026)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--minimum', type=int, default=50)
    args = parser.parse_args()
    source = args.out / 'crystals_by_community.csv'
    cbc = read_metadata(source)
    df = concentration(cbc.spacegroup_number, cbc.community_fine,
                       args.draws, args.seed, args.minimum)
    df.to_csv(args.out / 'spacegroup_capture.csv', index=False)
    save_json(args.out / 'spacegroup_capture_summary.json', {
        'provenance': provenance([source], seed=args.seed, draws=args.draws, minimum=args.minimum),
        'groups_tested': len(df), 'bh_q_at_most_005': int((df.bh_q <= .05).sum()),
        'interpretation': 'Concentration relative to label permutation; no prototype identification or energy conditioning. Null intervals are not confidence intervals for the effect.',
    })
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
