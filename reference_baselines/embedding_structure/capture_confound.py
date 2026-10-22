"""Exploratory association of concentration with stoichiometric diversity."""
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from scipy.stats import spearmanr
from .common import arguments, read_metadata, provenance, save_json


def signature(formula):
    composition = Composition(formula, strict=True)
    fractions = [Fraction(str(v)).limit_denominator(1000000)
                 for v in composition.get_el_amt_dict().values()]
    if not fractions or any(x <= 0 for x in fractions):
        raise ValueError(f'Invalid positive composition: {formula}')
    denominator = lcm(*(x.denominator for x in fractions))
    counts = [int(x * denominator) for x in fractions]
    divisor = reduce(gcd, counts)
    return tuple(sorted(x // divisor for x in counts))


def main():
    args = arguments(__doc__).parse_args()
    cap_path = args.out / 'spacegroup_capture.csv'
    cbc_path = args.out / 'crystals_by_community.csv'
    cap, cbc = read_metadata(cap_path), read_metadata(cbc_path)
    # Fail on an invalid formula rather than silently omit it.
    cbc['sig'] = cbc.pretty_formula.apply(signature)
    rows = []
    for group, frame in cbc.groupby('spacegroup_number'):
        if group not in set(cap.space_group):
            continue
        counts = frame.sig.value_counts()
        probabilities = counts.to_numpy() / len(frame)
        rows.append({'space_group': group, 'n_stoich_types': len(counts),
                     'top_type_share': float(probabilities.max()),
                     'eff_stoich_types': float(np.exp(-np.sum(probabilities*np.log(probabilities))))})
    merged = cap.merge(pd.DataFrame(rows), on='space_group', validate='one_to_one')
    associations = {}
    for field in ['eff_stoich_types', 'top_type_share', 'members']:
        rho, p = spearmanr(merged.enrichment, merged[field])
        associations[field] = {'rho': float(rho) if np.isfinite(rho) else None,
                               'p_unadjusted': float(p) if np.isfinite(p) else None}
    merged.to_csv(args.out / 'capture_vs_homogeneity.csv', index=False)
    save_json(args.out / 'capture_associations.json', {
        'provenance': provenance([cap_path, cbc_path]),
        'groups': len(merged), 'associations': associations,
        'interpretation': 'Exploratory marginal correlations; no causal conclusion, energy adjustment, or identification of structural prototypes.',
    })
    systems = ['cubic', 'hexagonal', 'trigonal', 'tetragonal', 'orthorhombic', 'monoclinic', 'triclinic']
    colors = dict(zip(systems, plt.cm.tab10(range(len(systems)))))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    d = cap.sort_values('enrichment', ascending=False)
    ax[0].bar(range(len(d)), d.enrichment, color=[colors[s] for s in d.crystal_system])
    ax[0].axhline(1, ls='--', c='gray', lw=1)
    ax[0].set(xlabel='Space groups ranked by concentration', ylabel='Observed / permutation-mean top-five share',
              title='Space-group concentration (size-matched null)')
    ax[0].legend([plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in systems], systems, fontsize=8, ncol=2)
    ax[1].scatter(merged.eff_stoich_types, merged.enrichment,
                  c=[colors[s] for s in merged.crystal_system], s=35)
    for row in merged.itertuples():
        if row.space_group in (225, 216, 194, 221, 62, 12, 166):
            ax[1].annotate(str(row.space_group), (row.eff_stoich_types, row.enrichment), fontsize=8)
    rho = associations['eff_stoich_types']['rho']
    ax[1].axhline(1, ls='--', c='gray', lw=1)
    ax[1].set(xlabel='Effective number of stoichiometry types', ylabel='Observed / permutation-mean top-five share',
              title=f'Exploratory diversity association (Spearman {rho:+.2f})' if rho is not None else 'Diversity association undefined')
    fig.tight_layout()
    fig.savefig(args.out / 'capture_figure.png', dpi=130)
    plt.close(fig)
    print(associations)


if __name__ == '__main__':
    main()
