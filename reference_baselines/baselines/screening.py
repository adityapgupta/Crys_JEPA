"""
V_screen.py, with the EMBEDDING GENERATION removed.

Instead of loading the JEPA model and embedding candidates + references on the fly,
this version loads PRE-COMPUTED embeddings and keeps the rest
of V_screen's logic identical: subsystem reference gathering, the D_C score
(mean squared embedding distance), and the valid+unique+novel ranking.

Requires that candidates and references were embedded with the SAME model + scaler
(so they live in the same space). File counts alone cannot verify this provenance.

Run: python V_screen_precomputed.py --data-dir embeddings/data --out-dir screen_out
"""

from itertools import combinations
from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd

from reference_baselines.paths import ROOT


def expand_into_subsystems(chemical_system: str) -> list:
    elements = chemical_system.split('-')
    list_combinations = []
    for n in range(1, len(elements) + 1):
        list_combinations += list(combinations(elements, n))   # C_{elements}^n
    return list_combinations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'embeddings/data')
    parser.add_argument('--out-dir', type=Path, default=Path('screen_out'))
    args = parser.parse_args()
    CAND_EMB = args.data_dir / 'candidate_embeddings.npy'
    CAND_META = args.data_dir / 'candidate_metadata.csv'
    REF_EMB = args.data_dir / 'embeddings.npy'
    REF_META = args.data_dir / 'metadata.csv'
    NUM_TURNS = 40                           # mp_20: 40 turns x 10 folders = eval folders 0..399
    OUT_DIR = args.out_dir


    # ---- load pre-computed embeddings (this block REPLACES the model + get_emb code) ----
    cand_emb  = np.load(CAND_EMB).astype(np.float32)
    cand_meta = pd.read_csv(CAND_META)
    ref_emb   = np.load(REF_EMB).astype(np.float32)
    ref_meta  = pd.read_csv(REF_META)
    assert len(cand_emb) == len(cand_meta), "candidate embeddings and metadata must be row-aligned"
    assert len(ref_emb)  == len(ref_meta),  "reference embeddings and metadata must be row-aligned"
    assert cand_emb.ndim == ref_emb.ndim == 2 and cand_emb.shape[1] == ref_emb.shape[1]
    assert np.isfinite(cand_emb).all() and np.isfinite(ref_emb).all(), "embeddings must be finite"
    assert cand_meta['vun'].isin([True, False]).all(), "vun must contain Boolean values"
    assert not cand_meta.duplicated(['eval_folder', 'idx']).any(), "duplicate candidate IDs"
    assert cand_meta['eval_folder'].between(0, NUM_TURNS * 10 - 1).all(), "unexpected eval folder"
    assert cand_meta['chemical_system'].notna().all() and ref_meta['chemical_system'].notna().all()

    # group reference embeddings by exact chemical system, so subsystem lookup is a dict hit
    ref_by_sys = {sys: ref_emb[np.asarray(idx)]
                  for sys, idx in ref_meta.groupby('chemical_system').groups.items()}

    os.makedirs(OUT_DIR, exist_ok=True)


    # ---- screening loop: SAME structure as V_screen, embedding swapped for a lookup ----
    all_rows = []
    for turn in range(NUM_TURNS):
        folders = [turn * 10 + p for p in range(10)]
        mask = cand_meta['eval_folder'].isin(folders).values
        sub_meta = cand_meta[mask].reset_index(drop=True)
        sub_emb  = cand_emb[mask]
        if len(sub_meta) == 0:
            continue

        emb_dis = []
        for i in range(len(sub_meta)):
            gen_emb = sub_emb[i]                                  # <-- was: get_emb(deal_structures([candidate]))
            cs_gen  = sub_meta.loc[i, 'chemical_system']

            # gather references: the candidate's chemical system + all its subsystems
            subsys = expand_into_subsystems(cs_gen)
            refs = []
            for s in subsys:
                for key in ['-'.join(sorted(s))]:
                    if key in ref_by_sys:
                        refs.append(ref_by_sys[key])             # <-- was: embed the matching ref CIFs

            if len(refs) > 0:
                ref_arr = np.vstack(refs)                        # (num_refs, 512)
                emb_dis.append(float(((gen_emb[None, :] - ref_arr) ** 2).sum(-1).mean()))   # D_C
            else:
                emb_dis.append(1e9)

        emb_dis = np.array(emb_dis)
        np.save(os.path.join(OUT_DIR, f'dis_{turn}.npy'), emb_dis)   # per-turn D_C (like dis_<turn>.pt)
        print(f'turn {turn}: {len(sub_meta)} candidates scored', flush=True)

        # ---- valid+unique+novel ranking (same as V_screen) ----
        vun = sub_meta['vun'].values.astype(bool)
        index = np.arange(len(emb_dis))
        vun_dis = emb_dis[vun]
        vun_index = index[vun]
        vun_order = np.argsort(vun_dis)
        vun_index_order = vun_index[vun_order]

        for rank, j in enumerate(vun_index_order):
            all_rows.append({
                'turn': turn,
                'eval_folder': int(sub_meta.loc[j, 'eval_folder']),
                'idx': int(sub_meta.loc[j, 'idx']),
                'chemical_system': sub_meta.loc[j, 'chemical_system'],
                'D_C': float(emb_dis[j]),
                'rank_in_turn': rank,
            })

    # the screened + ranked v.u.n candidates (replaces the ft_<turn>.csv output)
    ranked = pd.DataFrame(all_rows)
    ranked.to_csv(os.path.join(OUT_DIR, 'screened_ranked.csv'), index=False)
    scoreable = ranked['D_C'] < 1e9
    summary = {
        'reference_count': len(ref_meta),
        'candidate_count': len(cand_meta),
        'embedding_dimension': cand_emb.shape[1],
        'vun_count': len(ranked),
        'scoreable_vun_count': int(scoreable.sum()),
        'unscoreable_vun_count': int((~scoreable).sum()),
        'missing_eval_folders': sorted(set(range(NUM_TURNS * 10)) - set(cand_meta['eval_folder'])),
        'score_quantiles_scoreable_vun': {
            str(q): float(ranked.loc[scoreable, 'D_C'].quantile(q))
            for q in [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
        },
        'ranking': 'ascending D_C within each group of ten eval folders; zero-based ranks',
        'unscoreable_policy': 'retain with original sentinel score 1e9',
        'comparison_limit': 'Original saved score arrays and rankings are unavailable; compare PROGRESS.MD summaries only.',
    }
    with open(OUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"done -> {OUT_DIR}/dis_*.npy (per-turn D_C) and {OUT_DIR}/screened_ranked.csv "
          f"({len(all_rows)} valid+unique+novel candidates ranked by D_C)")


if __name__ == '__main__':
    main()
