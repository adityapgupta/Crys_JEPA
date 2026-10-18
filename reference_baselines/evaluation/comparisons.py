"""Paired chemical-system bootstrap for fixed validation selections, relative to Crys-JEPA."""

import json

import numpy as np
import pandas as pd

from reference_baselines.evaluation.metrics import unique_order, load_adjacency
from reference_baselines.paths import RESULTS_DIR


OUT = RESULTS_DIR


def main():
    scores = pd.read_csv(OUT/'candidate_reference_scores.csv')
    labels = pd.read_csv(OUT/'candidate_labels.csv')
    use = scores.vun & scores.split.eq('validation') & labels.hull_status.eq('ok')
    frame = scores[use].copy()
    frame['success'] = labels.loc[use, 'near_stable_proxy'].astype(int)
    metrics = pd.read_csv(OUT/'validation_baseline_metrics.csv')
    methods = [m for m in metrics.method.unique() if m in frame]
    adjacency = load_adjacency()
    results = []
    for selection_policy in ['raw', 'unique_refill']:
      for fraction in [.01, .05]:
        budget = int(len(frame)*fraction)
        original = frame[np.isfinite(frame.crys_jepa)].sort_values(['crys_jepa','candidate_row'])
        if selection_policy == 'unique_refill':
            original = frame.loc[unique_order(original.index, adjacency)]
        original = original.head(budget)
        original_groups = original.groupby('chemical_system').success.agg(['sum','count'])
        for method in methods:
            ranked = frame[np.isfinite(frame[method])].sort_values([method,'candidate_row'])
            if selection_policy == 'unique_refill':
                ranked = frame.loc[unique_order(ranked.index, adjacency)]
            ranked = ranked.head(budget)
            current = ranked.groupby('chemical_system').success.agg(['sum','count'])
            systems = original_groups.index.union(current.index)
            a = original_groups.reindex(systems, fill_value=0).to_numpy()
            b = current.reindex(systems, fill_value=0).to_numpy()
            rng = np.random.default_rng(2026)
            bootstrap = rng.integers(0, len(systems), (2000, len(systems)))
            aa, bb = a[bootstrap].sum(1), b[bootstrap].sum(1)
            keep = (aa[:,1]>0)&(bb[:,1]>0)
            difference = bb[keep,0]/bb[keep,1] - aa[keep,0]/aa[keep,1]
            results.append({'method':method, 'selection_fraction':fraction, 'selection_policy':selection_policy,
                            'precision_difference_vs_crys_jepa':float(ranked.success.mean()-original.success.mean()),
                            'paired_cluster_95_percent_interval':json.dumps(np.quantile(difference,[.025,.975]).tolist()),
                            'bootstrap_replicates':int(keep.sum())})
    pd.DataFrame(results).to_csv(OUT/'validation_paired_comparisons.csv',index=False)
    print('Paired validation comparisons saved. Intervals are descriptive, unadjusted for multiple comparisons.')


if __name__ == '__main__':
    main()
