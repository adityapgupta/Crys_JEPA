"""Space-group concentration under energy-stratified permutations."""
import numpy as np
import pandas as pd

from .common import arguments, read_metadata, aligned_communities, provenance, save_json
from .spacegroup_capture import concentration


def energy_strata(energy, width):
    energy = np.asarray(energy, dtype=float)
    if energy.ndim != 1 or not len(energy) or not np.isfinite(energy).all():
        raise ValueError('All crystals require finite formation energies')
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Energy bin width must be positive and finite')
    # Fixed zero origin; bins are [b * width, (b + 1) * width).
    return np.floor(energy / width).astype(np.int64)


def mixing_diagnostics(energy, sg, community, bins):
    rows = []
    movable = np.zeros(len(energy), dtype=bool)
    for b in np.unique(bins):
        mask = bins == b
        labels = np.unique(sg[mask]).size
        communities = np.unique(community[mask]).size
        movable[mask] = labels > 1 and communities > 1
        rows.append({'bin': int(b), 'members': int(mask.sum()),
                     'energy_min': float(energy[mask].min()),
                     'energy_max': float(energy[mask].max()),
                     'space_groups': labels, 'communities': communities})
    return pd.DataFrame(rows), movable


def main():
    parser = arguments(__doc__, data=True)
    parser.set_defaults(seed=2026)
    parser.add_argument('--widths', type=float, nargs='+', default=[.01, .025, .05],
                        help='Energy bin widths in eV/atom; 0.01 is the primary analysis')
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--minimum', type=int, default=50)
    args = parser.parse_args()
    if len(set(args.widths)) != len(args.widths):
        parser.error('Use distinct bin widths')
    source = args.out / 'crystals_by_community.csv'
    metadata = args.data / 'metadata.csv'
    meta = read_metadata(metadata)
    cbc = aligned_communities(meta, read_metadata(source))
    energy = meta.formation_energy_per_atom.to_numpy(dtype=float)
    sg = cbc.spacegroup_number.to_numpy()
    community = cbc.community_fine.to_numpy()
    strata = [(w, energy_strata(energy, w)) for w in args.widths]
    out = args.out / 'energy_capture'
    out.mkdir(parents=True, exist_ok=True)
    print('Computing unrestricted baseline...', flush=True)
    baseline = concentration(sg, community, args.draws, args.seed, args.minimum)
    baseline.to_csv(out / 'unrestricted.csv', index=False)
    summaries = []
    for width, bins in strata:
        print(f'Energy width {width:g} eV/atom...', flush=True)
        result = concentration(sg, community, args.draws, args.seed, args.minimum, strata=bins)
        diagnostics, movable = mixing_diagnostics(energy, sg, community, bins)
        result['fraction_in_mixed_bins'] = [float(movable[sg == g].mean()) for g in result.space_group]
        comparison = baseline[['space_group', 'enrichment', 'bh_q']].rename(
            columns={'enrichment': 'unrestricted_enrichment', 'bh_q': 'unrestricted_bh_q'})
        result = result.merge(comparison, on='space_group', validate='one_to_one')
        name = f'width_{width:g}'
        result.to_csv(out / f'{name}.csv', index=False)
        diagnostics.to_csv(out / f'{name}_bins.csv', index=False)
        summaries.append({'width_ev_per_atom': width, 'groups_tested': len(result),
                          'bh_q_at_most_005': int((result.bh_q <= .05).sum()),
                          'fraction_in_mixed_bins': float(movable.mean()),
                          'bins': len(diagnostics)})
        print(summaries[-1], flush=True)
    save_json(out / 'summary.json', {
        'provenance': provenance([source, metadata], seed=args.seed, draws=args.draws,
                                 minimum=args.minimum, widths=args.widths, bin_origin=0),
        'unrestricted_significant_groups': int((baseline.bh_q <= .05).sum()),
        'analyses': summaries,
        'interpretation': 'Conditional concentration beyond a binned-energy null, not complete energy independence. BH adjustment is separate within each width; widths are sensitivity checks, not opportunities to select significance. Mixed-bin fractions describe possible mixing, not power. Null intervals are not effect confidence intervals.',
    })


if __name__ == '__main__':
    main()
