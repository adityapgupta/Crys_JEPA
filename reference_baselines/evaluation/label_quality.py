"""Flag extreme proxy energies and test sensitivity without changing the primary labels."""

import json

import numpy as np
import pandas as pd

from reference_baselines.evaluation.metrics import load_adjacency, unique_order
from reference_baselines.paths import RESULTS_DIR


OUT = RESULTS_DIR


def main():
    labels = pd.read_csv(OUT/'candidate_labels.csv',keep_default_na=False)
    scores = pd.read_csv(OUT/'candidate_reference_scores.csv')
    use = scores.vun & scores.split.isin(['train','validation'])
    working = labels[use].copy()
    working['split'] = scores.loc[use,'split']
    flagged = working[working.e_above_hull_proxy < -.5]
    flagged.to_csv(OUT/'extreme_proxy_energy_flags.csv',index=False)
    frame = scores[scores.vun & scores.split.eq('validation')].copy()
    frame['hull'] = labels.loc[frame.index,'e_above_hull_proxy'].astype(float)
    targets = labels.near_stable_proxy.astype(int).to_numpy()
    methods = json.loads((OUT/'reference_experiment_provenance.json').read_text())['score_methods']
    adjacency = load_adjacency()
    rows = []
    for lower_bound in [-np.inf,-1.,-.5,-.25]:
        subset = frame[frame.hull >= lower_bound]
        for method in methods:
            ranked = subset[np.isfinite(subset[method])].sort_values([method,'candidate_row']).candidate_row
            unique = unique_order(ranked,adjacency)
            for fraction in [.01,.05]:
                budget = int(len(subset)*fraction)
                selection = unique[:budget]
                rows.append({'method':method,'selection_fraction':fraction,'proxy_hull_lower_bound':lower_bound,
                             'validation_rows':len(subset),'excluded_validation_rows':len(frame)-len(subset),
                             'budget':budget,'unique_selected_count':len(selection),
                             'unique_proxy_yield_per_slot':float(targets[selection].sum()/budget)})
    pd.DataFrame(rows).to_csv(OUT/'validation_extreme_label_sensitivity.csv',index=False)
    report = {'status':'passed','scope':'VUN train and validation only; no test performance examined',
              'primary_labels_changed':False,'flag_rule':'proxy hull energy < -0.5 eV/atom; diagnostic, not proof of a bad label',
              'flagged_count':len(flagged),'flagged_by_split':flagged.split.value_counts().to_dict(),
              'below_minus_1_eV_count':int((working.e_above_hull_proxy < -1.).sum()),
              'proxy_hull_quantiles':working.e_above_hull_proxy.quantile([0,.001,.01,.5,.99,.999,1]).to_dict(),
              'interpretation':'Large negative distances may reflect MLFF error, correction mismatch, or sparse fixed-hull coverage. Negative values alone are not an invalidity test. Keep primary labels; report exclusion sensitivity separately.',
              'external_sources':['https://arxiv.org/abs/1802.07605','https://doi.org/10.1038/s41524-025-01602-9']}
    (OUT/'label_quality_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
