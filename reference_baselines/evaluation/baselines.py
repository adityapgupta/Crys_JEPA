"""Evaluate fixed reference-selection baselines on validation rows only."""

import json
import re

from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from reference_baselines.paths import ROOT, RESULTS_DIR


def cluster_interval(table, seed=2026):
    groups = table.groupby('chemical_system').near_stable_proxy.agg(['sum', 'count'])
    if len(groups) < 2:
        return [None, None]
    values = groups.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (1000, len(values)))].sum(1)
    return np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist()


def main():
    out = RESULTS_DIR
    scores = pd.read_csv(out / 'candidate_reference_scores.csv')
    labels = pd.read_csv(out / 'candidate_labels.csv')
    assert len(scores) == len(labels)
    np.testing.assert_array_equal(scores.candidate_row, labels.candidate_row)
    for col in ['eval_folder', 'idx', 'chemical_system', 'vun']:
        np.testing.assert_array_equal(scores[col], labels[col])
    # Never evaluate the locked test split here.
    use = scores.vun & (scores.split == 'validation') & (labels.hull_status == 'ok')
    frame = scores.loc[use].copy()
    assert labels.loc[use, 'near_stable_proxy'].isin([True, False]).all()
    frame['near_stable_proxy'] = labels.loc[use, 'near_stable_proxy'].astype(bool).to_numpy()
    frame['e_above_hull_proxy'] = labels.loc[use, 'e_above_hull_proxy'].to_numpy()
    frame['matter_sim_hull_oracle'] = frame.e_above_hull_proxy

    embeddings = np.load(ROOT / 'embeddings/data/candidate_embeddings.npy', mmap_mode='r')
    references = np.load(ROOT / 'embeddings/data/embeddings.npy', mmap_mode='r')
    q = np.asarray(embeddings[frame.candidate_row], dtype=np.float64)
    centroid = np.asarray(references, dtype=np.float64).mean(0)
    frame['embedding_norm_control'] = np.square(q).sum(1)
    frame['global_centroid_control'] = np.square(q-centroid).sum(1)
    methods = list(json.loads((out / 'reference_experiment_provenance.json').read_text())['score_methods'])
    methods += ['embedding_norm_control', 'global_centroid_control', 'matter_sim_hull_oracle']
    base_rate = float(frame.near_stable_proxy.mean())
    rows = []
    sensitivity = []
    per_turn = []
    selection_records = []
    for method in methods:
        available = frame[np.isfinite(frame[method])].copy()
        ranked = available.sort_values([method, 'candidate_row'], kind='stable')
        rank_correlation = float(spearmanr(available[method], available.e_above_hull_proxy).statistic)
        for fraction in [.01, .05, .10, .20]:
            requested = int(np.floor(fraction * len(frame)))
            selected = ranked.head(requested)
            precision = float(selected.near_stable_proxy.mean()) if len(selected) else None
            positives = int(selected.near_stable_proxy.sum())
            row = {'method': method, 'selection_fraction': fraction, 'budget': requested,
                   'selected_count': len(selected), 'coverage_fraction': len(available) / len(frame),
                   'near_stable_proxy_fraction': precision, 'near_stable_count': positives,
                   'yield_per_requested_slot': positives / requested,
                   'random_expected_fraction': base_rate,
                   'enrichment_over_random': precision / base_rate if precision is not None else None,
                   'chemical_systems_selected': selected.chemical_system.nunique(),
                   'median_hull_proxy': float(selected.e_above_hull_proxy.median()),
                   'score_hull_spearman': rank_correlation,
                   'mean_hull_proxy': float(selected.e_above_hull_proxy.mean())}
            if fraction in [.01, .05]:
                row['cluster_bootstrap_95_percent_interval'] = json.dumps(cluster_interval(selected))
            rows.append(row)
            for threshold in [0., .05, .1]:
                sensitivity.append({'method': method, 'selection_fraction': fraction,
                                    'threshold_eV_per_atom': threshold, 'selected_count': len(selected),
                                    'yield_per_requested_slot': int((selected.e_above_hull_proxy <= threshold).sum()) / requested})
            if fraction == .01:
                selection_records.extend({'method': method, 'candidate_row': int(i)} for i in selected.candidate_row)
        for turn, group in frame.groupby(frame.eval_folder // 10):
            available_group = group[np.isfinite(group[method])].sort_values([method, 'candidate_row'], kind='stable')
            for fraction in [.01, .05, .1, .2]:
                budget = int(np.floor(len(group) * fraction))
                selected = available_group.head(budget)
                per_turn.append({'method': method, 'turn': int(turn), 'selection_fraction': fraction,
                                 'budget': budget, 'selected_count': len(selected),
                                 'near_stable_count': int(selected.near_stable_proxy.sum())})
    # Random candidate selection: expected rate and repeat-to-repeat variation at fixed budget.
    rng = np.random.default_rng(2026)
    for fraction in [.01, .05, .10, .20]:
        budget = int(np.floor(fraction * len(frame)))
        values = [float(frame.near_stable_proxy.to_numpy()[rng.choice(len(frame), budget, replace=False)].mean()) for _ in range(100)]
        rows.append({'method': 'random_candidates', 'selection_fraction': fraction, 'budget': budget,
                     'selected_count': budget, 'coverage_fraction': 1.,
                     'near_stable_proxy_fraction': float(np.mean(values)), 'random_expected_fraction': base_rate,
                     'random_repeat_95_percent_interval': json.dumps(np.quantile(values, [.025, .975]).tolist())})
    result = pd.DataFrame(rows)
    result.to_csv(out / 'validation_baseline_metrics.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'validation_threshold_sensitivity.csv', index=False)
    per_turn = pd.DataFrame(per_turn)
    per_turn.to_csv(out / 'validation_per_turn_metrics.csv', index=False)
    per_turn.groupby(['method', 'selection_fraction'])[['budget', 'selected_count', 'near_stable_count']].sum().assign(
        yield_per_requested_slot=lambda x: x.near_stable_count / x.budget).to_csv(out / 'validation_per_turn_summary.csv')
    pd.DataFrame(selection_records).to_csv(out / 'validation_top1_selections.csv', index=False)

    # Physical reference relevance: recall of actual hull phases present in our permitted bank.
    rm = pd.read_csv(out / 'reference_rows.csv')
    material_to_row = {s: i for i, s in enumerate(rm.material_id)}
    knn = np.load(out / 'candidate_knn_indices.npy')
    chemical_knn = np.load(out / 'chemical_knn_indices.npy')
    indptr = np.load(out / 'chemical_reference_indptr.npy')
    ids = np.load(out / 'chemical_reference_indices.npy')
    physical = []
    for record in labels.loc[use].itertuples():
        targets = {}
        for product in json.loads(record.decomposition):
            match = re.match(r'^(mp-\d+)', product['entry_id'])
            if match and match.group(1) in material_to_row:
                ref_row = material_to_row[match.group(1)]
                targets[ref_row] = targets.get(ref_row, 0.) + product['fraction']
        available_weight = sum(targets.values())
        row = record.candidate_row
        original = set(ids[indptr[row]:indptr[row + 1]])
        for k in [5, 10, 20, 50]:
            nearest = set(knn[row, :k]); intersection = original & nearest
            pools = {'crys_jepa': original, f'knn_k{k}': nearest, f'union_k{k}': original | nearest,
                     f'intersection_k{k}': intersection, f'chemical_knn_k{k}': set(chemical_knn[row, :k]) - {-1}}
            for name, pool in pools.items():
                if name == 'crys_jepa' and k != 5:
                    continue
                recovered = sum(weight for target, weight in targets.items() if target in pool)
                physical.append({'candidate_row': row, 'method': name, 'available_decomposition_weight': available_weight,
                                 'recovered_weight': recovered, 'recall_when_available': recovered / available_weight if available_weight else np.nan})
    pd.DataFrame(physical).to_csv(out / 'validation_decomposition_recall.csv', index=False)
    p = pd.DataFrame(physical)
    summaries = p.groupby('method').agg(queries=('candidate_row', 'size'),
        queries_with_bank_phase=('recall_when_available', 'count'),
        mean_available_decomposition_weight=('available_decomposition_weight', 'mean'),
        mean_recovered_weight=('recovered_weight', 'mean'),
        mean_recall_when_available=('recall_when_available', 'mean')).reset_index()
    summaries.to_csv(out / 'validation_decomposition_summary.csv', index=False)
    report = {'population': 'VUN validation candidates with usable proxy labels', 'candidate_count': len(frame),
              'near_stable_proxy_base_rate': base_rate, 'metrics_file': 'validation_baseline_metrics.csv',
              'physical_reference_relevance': 'fractional recall of decomposition phases present in the fixed MP-20 reference bank; full-hull phases absent from the bank are reported separately',
              'threshold': .1, 'test_evaluation_performed': False,
              'oracle_note': 'MatterSim hull ranking uses the same signal as the proxy labels; it is a diagnostic ceiling, not independent validation.',
              'abstention_policy': 'NaN scores abstain; selected_count may be below requested budget',
              'limits': ['No new DFT validation has been run.', 'Released VUN flags are batch-local and refer to MP-20.',
                         'Cluster intervals describe validation selection uncertainty, not model training seed variation.']}
    (out / 'validation_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(result[result.selection_fraction == .01][['method', 'near_stable_proxy_fraction', 'coverage_fraction']].to_string(index=False))


if __name__ == '__main__':
    main()
