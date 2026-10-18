"""Build the experiment report; require all checks before declaring training readiness."""

from datetime import datetime, timezone
import importlib.metadata
import json

import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.paths import ROOT, PACKAGE_DIR, RESULTS_DIR, CONFIG_DIR, DOCS_DIR, SOURCE_PACKAGES


HERE = PACKAGE_DIR
OUT = RESULTS_DIR


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    assert read('verification.json')['passed']
    for name in ['energy_comparison.json','structure_fingerprint_provenance.json','structure_comparison.json']:
        assert read(name)['verification']['passed']
    for name in ['outcome_verification.json','training_smoke_check.json','validation_global_uniqueness.json']:
        assert read(name)['status'] == 'passed'
    smoke = read('training_smoke_check.json')
    assert len(smoke['models']) == 6 and smoke['optimizer_steps'] == 0 and smoke['test_predictions'] == 0
    for name, digest in smoke['sha256'].items():
        assert sha256(HERE/name) == digest, f'Stale training check: {name}'
    audit = read('training_data_audit.json')
    assert audit['split_overlap_chemical_systems'] == 0 and not audit['test_metrics_computed']
    assert audit['excluded_vun_rows_without_label'] == 0
    assert not read('validation_evaluation.json')['test_evaluation_performed']
    assert read('candidate_label_provenance.json')['hull_status'] == {'ok':324622}
    quality = read('label_quality_audit.json')
    assert quality['status'] == 'passed' and not quality['primary_labels_changed']
    unique = pd.read_csv(OUT/'validation_unique_baseline_metrics.csv')
    raw = pd.read_csv(OUT/'validation_baseline_metrics.csv')
    paired = pd.read_csv(OUT/'validation_paired_comparisons.csv')
    assert set(paired.selection_policy) == {'raw','unique_refill'}
    names = {'crys_jepa':'Crys-JEPA','knn_k10':'kNN, k=10','union_k10':'Union, k=10',
             'intersection_fallback_k5':'Intersection, k=5, with fallback',
             'chemical_softmax_t1':'Chemical softmax, temperature=1','random_candidates':'Random candidates (100 repeats)'}
    lines = ['# Reference experiments and training readiness', '',
             'Ready for model training with MatterSim proxy labels. No full model training or new DFT calculation has been run.', '',
             '## Validation results', '',
             'A positive label means corrected MatterSim energy at most 0.1 eV/atom above the fixed MP2023 DFT hull. These are proxy results. The budget is 1% or 5% of 37,438 validation candidates.', '',
             'For unique results, remove repeated structures across batches in score order. Refill from the same list to give each method the same number of unique crystals. Test performance remains unused.', '',
             '| Method | Raw positive rate, 1% | Unique positive rate, 1% | Unique positive rate, 5% |',
             '| --- | ---: | ---: | ---: |']
    for method, name in names.items():
        a = raw[(raw.method == method)&raw.selection_fraction.eq(.01)].iloc[0]
        b = unique[unique.method == method].set_index('selection_fraction')
        lines.append(f'| {name} | {100*a.near_stable_proxy_fraction:.1f}% | {100*b.loc[.01,"unique_yield_per_requested_dft_slot"]:.1f}% | {100*b.loc[.05,"unique_yield_per_requested_dft_slot"]:.1f}% |')
    comparison = paired[(paired.method=='intersection_fallback_k5')&paired.selection_fraction.eq(.05)&paired.selection_policy.eq('unique_refill')].iloc[0]
    ci = json.loads(comparison.paired_cluster_95_percent_interval)
    lines += ['',f'The paired chemical-system bootstrap interval for intersection-fallback minus Crys-JEPA at the unique 5% budget is [{100*ci[0]:.1f}, {100*ci[1]:.1f}] percentage points. Intervals are descriptive, with no multiple-comparison correction. These runs do not establish a reliable winner.', '',
              'The pure k=5 intersection covers only 4.3% of validation candidates. The fallback uses the original chemical-pool mean when the intersection is empty. A larger union with a fixed mean does not give a clear small-budget gain.', '',
              '## Reference quality', '',
              '- All 27,136 known CIFs pass ID, formula, and descriptor checks. JEPA neighbours have closer geometry-only SOAP descriptors than energy-gap-matched controls. This supports a geometry signal, not screening success.',
              '- The k=10 reference graph is connected. Formation energy is smooth on the graph. The eigensolver did not converge, so no spectral-gap result is claimed.',
              '- On average, 45.4% of full-hull decomposition weight is present in the small reference bank. At least one bank phase exists for 27,980 validation queries. Recall is assessed only for phases available in this bank.',
              '- Original chemical pools and unions retain all available decomposition phases by construction. Unrestricted k=10 neighbours recover about 1.0% on average among queries with an available phase. Cross-system references can provide context, but cannot supply decomposition phases with absent atoms.',
              '- Distance, formation energy, and attention weights are not physical correctness labels for a reference.', '',
              '## Data and checks', '',
              '- References: 27,136 x 512. Candidates: 324,622 x 512. Every candidate source ID, chemical system, and VUN flag agrees with the supplied arrays. Folder 84 is absent in both sources; it is not regenerated.',
              '- All 324,622 candidates receive usable labels. Candidate corrections follow the repository path. Evaluation uses the pinned full MP2023 hull, rather than the smaller MP20 screening hull.',
              '- Training: 176,458 VUN rows; validation: 37,438; test: 36,994. Exact chemical systems do not cross these splits. This holds out supervised candidate labels, not encoder pretraining chemistry. Known reference nodes remain available to every split.',
              '- Independent checks pass: 56 hull solutions against linear programming, decomposition balance, 128 stored-energy correction checks, and 768 direct reference-score checks. CSV parsing preserves the valid formula `NaN`.',
              '- Global validation uniqueness checks cover 54,338 same-formula pairs and find 12,998 matching pairs. Baselines and training use the same greedy StructureMatcher rule.',
              '- Six models pass finite forward/backward, target isolation, batch isolation, reference-order invariance, empty-pool, and attention-normalization checks. Checkpoint loading and reference export pass with a temporary untrained fixture. No optimizer step is taken.', '',
              '## Training definition', '',
              'Models: candidate-only MLP, reference attention, weighted GCN variant, weighted GraphSAGE mean variant, edge-aware MPNN, and sparse single-head graph transformer. They are small task-specific implementations, not exact reproductions. Each graph node is one whole reference crystal, not an atom.', '',
              'Primary pool: original chemical references union top 10 JEPA neighbours. Each candidate has its own induced reference graph. Reference models share the same query-to-reference attention readout. Features are frozen JEPA vectors, composition, vector norm, distances, and chemical eligibility. Energy labels never enter the inputs.', '',
              'The target is the MatterSim near-stable label. Use binary cross entropy, two learning rates, three seeds, and the fixed settings in `../configs/training.toml`. Stop on validation unique positive rate at the 5% budget. Pool and k ablations follow the primary comparison. Attention ranks context; it does not predict physical decomposition probabilities.', '',
              'After training, compare learned and uniform readout. Require gains over candidate-only and reference-attention models before crediting graph layers. Freeze all settings before opening test performance.', '',
              '## Limits and next stage', '',
              'MatterSim error and estimated compatibility corrections affect these labels. Final force convergence is not recorded. Release VUN flags do not establish novelty against MP2023. Training rows can repeat within their own split; chemical-system grouping prevents such structures from crossing downstream splits.', '',
              'The label audit flags 165 train/validation rows below -0.5 eV/atom relative to the fixed hull (38 validation rows). These extreme values can reflect force-field errors, correction mismatch, or sparse hull coverage. They are not treated as proven discoveries or silently removed. `validation_extreme_label_sensitivity.csv` reports results after excluding values below -1, -0.5, and -0.25 eV/atom. These cutoffs are diagnostic, not physical validity rules.', '',
              'Independent DFT validation remains required for stability claims. After model selection, fix a balanced subset from Crys-JEPA, fixed alternatives, learned models, and random controls. Use the same DFT settings, corrections, hull, and failure policy. No gain in generation or synthesis is established by these offline results.', '',
              '## Key files', '',
              '- `../results/validation_unique_baseline_metrics.csv`: primary unique comparison, with and without refill.',
              '- `../results/validation_baseline_metrics.csv`: 25 reference methods, two reference-free controls, random selection, and the label oracle.',
              '- `../results/validation_per_turn_summary.csv`, `validation_threshold_sensitivity.csv`: turn and threshold checks.',
              '- `../results/validation_paired_comparisons.csv`: uncertainty for fixed selections.',
              '- `../results/validation_decomposition_summary.csv`: phase recall and bank coverage.',
              '- `../results/training_data_audit.json`, `training_smoke_check.json`, `training_readiness.json`: readiness evidence.',
              '- `../results/label_quality_audit.json`, `extreme_proxy_energy_flags.csv`: proxy-label concerns and audit.',
              '- `../README.md`: reproduction commands. `../configs/training.toml`: exact training grid.', '',
              '## Sources', '',
              '- [Pinned Crys-JEPA data source](https://huggingface.co/datasets/liun-online/Crys_JEPA): structures, energies, reference hull.',
              '- [Materials Project phase diagrams](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds): composition-constrained phase competition.',
              '- [Matbench Discovery](https://arxiv.org/abs/2308.14920): fixed DFT hull and discovery selection metrics.',
              '- [Active learning for crystal prediction](https://arxiv.org/abs/1802.07605) and [robust potential training](https://doi.org/10.1038/s41524-025-01602-9): why predicted low-energy structures still need DFT checks and transferability tests.',
              '- [SOAP across structural and alchemical space](https://arxiv.org/abs/1601.04077), [DScribe](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html): independent geometry comparison.',
              '- [GCN](https://arxiv.org/abs/1609.02907), [GraphSAGE](https://arxiv.org/abs/1706.02216), [MPNN](https://arxiv.org/abs/1704.01212), [graph transformers](https://arxiv.org/abs/2012.09699): model families for controlled alternatives.', '']
    (DOCS_DIR/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines))
    files = list(HERE.glob('*.py')) + [f for directory in SOURCE_PACKAGES for f in (HERE/directory).glob('*.py')] + [CONFIG_DIR/'training.toml',PACKAGE_DIR/'requirements/structure.txt']
    ready = {'ready':True,'scope':'reference-model training with MatterSim proxy labels',
             'created_utc':datetime.now(timezone.utc).isoformat(),'full_model_training_started':False,
             'independent_dft_validation_complete':False,'test_performance_evaluated':False,
             'extreme_proxy_energy_flags_train_validation':quality['flagged_count'],
             'training_counts':audit['counts'],'models_checked':[m['model'] for m in smoke['models']],
             'primary_metric':'global-unique proxy positive rate at 5% candidate-count budget, with refill',
             'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','torch','pymatgen','dscribe','lmdb']},
             'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in files},
             'evidence':['outcome_verification.json','training_data_audit.json','training_smoke_check.json',
                         'validation_global_uniqueness.json','validation_unique_baseline_metrics.csv']}
    (OUT/'training_readiness.json').write_text(json.dumps(ready,indent=2)+'\n')
    print('Ready for MatterSim proxy model training. Report:',DOCS_DIR/'EXPERIMENT_REPORT.md')


if __name__ == '__main__':
    main()
