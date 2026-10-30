"""Equal-budget selection, global structure uniqueness, and fixed random controls."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.evaluation.metrics import unique_order
from reference_baselines.evaluation.selected_duplicates import compare_group
from .common import database, fingerprint, finish, receipt


def find_duplicates(metadata, structure_database):
    from pymatgen.core import Structure
    adjacency = {}
    pairs = []
    with database(structure_database) as db:
        for _, group in metadata.groupby("formula", sort=True):
            if len(group) < 2:
                continue
            structures = [(int(row.candidate_row), Structure.from_dict(json.loads(db.execute(
                "SELECT structure_json FROM structures WHERE id=?", (row.id,)).fetchone()[0])))
                for row in group.itertuples()]
            for a, b in compare_group((group.iloc[0].formula, structures)):
                pairs.append((a, b))
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
    return adjacency, pairs


def make_selections(scores, methods, fractions, adjacency, seed=2026):
    """Return selected rows plus every requested budget, including empty selections."""
    selections, requests = [], []
    scopes = [("turn", int(key), group) for key, group in scores.groupby("turn", sort=True)]
    scopes += [("batch", int(key), group) for key, group in scores.groupby("eval_folder", sort=True)]
    # Pooled reports aggregate the primary per-turn selections, rather than re-ranking them.
    for scope, group_id, frame in scopes:
        random_seed = int(fingerprint([seed, scope, group_id])[:16], 16)
        rng = np.random.default_rng(random_seed)
        random_order = rng.permutation(frame.candidate_row.to_numpy())
        for method in methods + ["random_candidates"]:
            if method == "random_candidates":
                ranked = random_order
            else:
                values = pd.to_numeric(frame[method], errors="coerce")
                ranked = frame.loc[np.isfinite(values)].assign(_score=values).sort_values(
                    ["_score", "candidate_row"], kind="stable").candidate_row.to_numpy()
            unique = unique_order(ranked, adjacency)
            for fraction in fractions:
                budget = int(np.floor(fraction * len(frame)))
                for policy, ordering in (("raw", ranked), ("unique_refill", unique)):
                    chosen = ordering[:budget]
                    raw_unique = len(unique_order(ranked[:budget], adjacency))
                    request = {"scope": scope, "group": group_id, "method": method,
                               "policy": policy, "fraction": fraction, "population": len(frame),
                               "budget": budget, "selected_count": len(chosen),
                               "scoreable_count": len(ranked), "raw_unique_without_refill": raw_unique,
                               "raw_selected_count": min(budget, len(ranked))}
                    requests.append(request)
                    selections.extend({**request, "candidate_row": int(row), "rank": rank}
                                      for rank, row in enumerate(chosen, 1))
    return pd.DataFrame(selections), pd.DataFrame(requests)


def select_candidates(config):
    output = Path(config["output"])
    identity = fingerprint({"scores": sha256(output / "score.json"), "seed": config["seed"],
                            "fractions": config["fractions"], "source": sha256(Path(__file__))})
    marker = output / "select.json"
    if receipt(marker, identity):
        return
    scores = pd.read_csv(output / "scores.csv", keep_default_na=False)
    methods = ["chemical"] + [f"{kind}_{k}" for kind in ("knn", "union") for k in config["neighbours"]]
    adjacency, pairs = find_duplicates(scores, output / "structures.sqlite")
    selected, requests = make_selections(scores, methods, config["fractions"], adjacency, config["seed"])
    if selected.empty:
        raise ValueError("No selections: increase the batch range or selection fraction")
    selected.to_csv(output / "selections.csv", index=False)
    requests.to_csv(output / "selection_requests.csv", index=False)
    pd.DataFrame(pairs, columns=["candidate_row_a", "candidate_row_b"]).to_csv(output / "duplicate_pairs.csv", index=False)
    wanted = sorted(set(selected.candidate_row))
    scores[scores.candidate_row.isin(wanted)].to_csv(output / "evaluation_candidates.csv", index=False)
    finish(marker, identity, [output / name for name in ("selections.csv", "selection_requests.csv", "duplicate_pairs.csv", "evaluation_candidates.csv")],
           evaluation_union_count=len(wanted), matching_pairs=len(pairs),
           primary_scope="turn", pooled_policy="aggregate turn selections; batch selections are diagnostic")
