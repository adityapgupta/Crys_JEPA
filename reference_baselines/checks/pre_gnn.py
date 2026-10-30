"""Audit all saved counts and exact group quotas independently of the evaluator."""

import json
import numpy as np

from reference_baselines.baselines.formula_balance import OUT
from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.heldout import protocol
from reference_baselines.paths import RESULTS_DIR, ROOT, PACKAGE_DIR


def main():
    protocol()  # Original fixed artifacts must remain unchanged.
    report = json.loads((OUT / "evaluation.json").read_text())
    for name, digest in report["sha256"].items():
        assert sha256(RESULTS_DIR / name) == digest, name
    labels = read_csv(RESULTS_DIR / "candidate_labels.csv").set_index("candidate_row")
    manifest = read_csv(RESULTS_DIR / "split_manifest.csv").set_index("candidate_row")
    selected = read_csv(OUT / "selections.csv")
    metrics = read_csv(OUT / "metrics.csv")
    keys = ["split", "budget_percent", "mode", "matched_group", "quota_anchor"]
    audited = 0
    for setting, frame in selected.groupby(keys, dropna=False):
        allocations = []
        for method, f in frame.groupby("method"):
            rows = f.candidate_row.to_numpy()
            assert len(rows) == len(np.unique(rows))
            match = metrics
            for key, value in zip(keys, setting):
                match = match[match[key].eq(value)]
            m = match[match.method.eq(method)].iloc[0]
            assert len(rows) == m.selected
            assert labels.loc[rows, "e_above_hull_proxy"].le(0.1).sum() == m.passing
            assert manifest.loc[rows, "split"].eq(setting[0]).all()
            if setting[2] == "matched":
                series = (
                    manifest.chemical_system
                    if setting[3] == "chemical_system"
                    else labels.formula
                )
                allocations.append(series.loc[rows].value_counts().sort_index())
            audited += 1
        for allocation in allocations[1:]:
            assert allocation.equals(allocations[0])
    result = {
        "status": "passed",
        "selection_groups_checked": audited,
        "exact_counts_and_stability_labels": True,
        "matched_quotas_identical": True,
        "original_frozen_artifacts_unchanged": True,
        "encoder_reproduction_complete": json.loads(
            (OUT / "encoder_check.json").read_text()
        )["saved_embeddings_reproduced"],
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256(p)
            for p in [
                PACKAGE_DIR / "baselines/formula_balance.py",
                PACKAGE_DIR / "evaluation/pre_gnn.py",
                PACKAGE_DIR / "evaluation/matched_pairs.py",
                PACKAGE_DIR / "checks/encoder_invariance.py",
                PACKAGE_DIR / "configs/fresh_evaluation.toml",
            ]
        },
    }
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
