"""Audit every validation duplicate pair and evaluate fixed-budget unique selections."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json

import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.evaluation.metrics import unique_order, load_adjacency
from reference_baselines.evaluation.selected_duplicates import load_folder, compare_group
from reference_baselines.paths import ROOT, RESULTS_DIR


OUT = RESULTS_DIR


def main():
    metadata = pd.read_csv(OUT/'split_manifest.csv')
    labels = pd.read_csv(OUT/'candidate_labels.csv', keep_default_na=False)
    chosen = metadata[metadata.vun & metadata.split.eq('validation') & labels.hull_status.eq('ok')]
    structures = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(load_folder, list(chosen.groupby('eval_folder'))), 1):
            structures.extend(result)
            if i % 50 == 0:
                print('Validation folders read:', i, flush=True)
    groups = {}
    for row, structure in structures:
        groups.setdefault(structure.composition.reduced_formula, []).append((row, structure))
    tasks = [(formula, sorted(records, key=lambda x: x[0])) for formula, records in groups.items() if len(records)>1]
    pairs = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(compare_group, tasks, chunksize=8),1):
            pairs.extend(result)
            if i % 1000 == 0:
                print('Validation formula groups checked:',i,'/',len(tasks),flush=True)
    pd.DataFrame(pairs,columns=['candidate_row_a','candidate_row_b']).to_csv(OUT/'validation_duplicate_pairs.csv',index=False)
    adjacency = load_adjacency()
    scores = pd.read_csv(OUT/'candidate_reference_scores.csv').set_index('candidate_row')
    frame = scores.loc[chosen.candidate_row].copy()
    frame['matter_sim_hull_oracle'] = labels.set_index('candidate_row').loc[frame.index, 'e_above_hull_proxy'].astype(float)
    embeddings = np.load(ROOT/'embeddings/data/candidate_embeddings.npy',mmap_mode='r')
    ref = np.load(ROOT/'embeddings/data/embeddings.npy',mmap_mode='r')
    q = np.asarray(embeddings[frame.index],dtype=np.float64)
    frame['embedding_norm_control'] = np.square(q).sum(1)
    frame['global_centroid_control'] = np.square(q-np.asarray(ref,dtype=np.float64).mean(0)).sum(1)
    targets = labels.near_stable_proxy.astype(int).to_numpy()
    methods = json.loads((OUT/'reference_experiment_provenance.json').read_text())['score_methods']
    methods += ['embedding_norm_control','global_centroid_control','matter_sim_hull_oracle']
    records = []
    for method in methods:
        ranked = frame[np.isfinite(frame[method])].sort_values([method,'candidate_row']).index.to_numpy()
        unique = unique_order(ranked,adjacency)
        for fraction in [.01,.05,.1,.2]:
            budget = int(len(frame)*fraction)
            raw = ranked[:budget]
            kept = unique_order(raw,adjacency)
            refill = unique[:budget]
            records.append({'method':method,'selection_fraction':fraction,'budget':budget,
                            'selected_before_dedup':len(raw),'unique_without_refill':len(kept),
                            'near_stable_unique_without_refill':int(targets[kept].sum()),
                            'unique_yield_per_original_slot':float(targets[kept].sum()/budget),
                            'unique_after_refill':len(refill),'near_stable_unique_after_refill':int(targets[refill].sum()),
                            'unique_yield_per_requested_dft_slot':float(targets[refill].sum()/budget)})
    rng = np.random.default_rng(2026)
    for fraction in [.01,.05,.1,.2]:
        budget = int(len(frame)*fraction)
        raw_values, refill_values = [], []
        for _ in range(100):
            ranked = rng.permutation(frame.index)
            kept = unique_order(ranked[:budget],adjacency)
            refill = unique_order(ranked,adjacency)[:budget]
            raw_values.append(targets[kept].sum()/budget)
            refill_values.append(targets[refill].sum()/budget)
        records.append({'method':'random_candidates','selection_fraction':fraction,'budget':budget,
                        'unique_yield_per_original_slot':float(np.mean(raw_values)),
                        'unique_yield_per_requested_dft_slot':float(np.mean(refill_values)),
                        'random_refill_repeat_95_percent_interval':json.dumps(np.quantile(refill_values,[.025,.975]).tolist())})
    pd.DataFrame(records).to_csv(OUT/'validation_unique_baseline_metrics.csv',index=False)
    report = {'status':'passed','validation_rows':len(chosen),'formula_groups':len(groups),
              'structure_pairs_checked':sum(len(g)*(len(g)-1)//2 for g in groups.values()),
              'matching_pairs':len(pairs),'matcher':'pymatgen StructureMatcher() defaults, as in repo evaluator',
              'selection_policy':'Greedy uniqueness in score order. Report both no refill and refill to fixed unique DFT budget.',
              'test_structures_examined':0,'novelty_against_mp2023_verified':False,
              'sha256':{p.name:sha256(p) for p in [Path(__file__).resolve(),OUT/'validation_duplicate_pairs.csv']}}
    (OUT/'validation_global_uniqueness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
