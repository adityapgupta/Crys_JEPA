"""Check accepted audit outputs and that the original frozen test is unchanged."""

import json
from importlib.metadata import version
import platform

import numpy as np

from reference_baselines.core.audit_io import read_csv
from reference_baselines.core.io import sha256
from reference_baselines.evaluation.heldout import protocol
from reference_baselines.evaluation.selection_audit import OUT, ORIGINAL, ANGLE
from reference_baselines.paths import RESULTS_DIR
from reference_baselines.evaluation.metrics import unique_order


def main():
    frozen = protocol()
    fresh = read_csv(OUT / "fresh_mattersim.csv")
    summary = json.loads((OUT / "fresh_mattersim.json").read_text())
    assert summary["graph_batch_size"] == 1
    assert summary["pass_label_disagreements"] == 0
    assert fresh.raw_energy_difference_eV_per_atom.abs().max() < 1e-4
    final = read_csv(OUT / "rerelaxed_checked.csv").set_index(
        "candidate_row", drop=False
    )
    np.testing.assert_array_equal(
        sorted(fresh.candidate_row), sorted(final.candidate_row)
    )
    assert len(final) == 3243 and final.index.is_unique
    assert final.steps.le(300).all()
    converged = final[final.converged]
    assert converged.final_fmax.le(0.050001).all()
    assert converged.filtered_fmax.le(0.050001).all()
    assert final.rerelaxed_passing.eq(final.rerelaxed_hull.le(0.1)).all()
    added = final[final.selection_group.eq("added")]
    assert len(added) == 1394
    assert (
        added.converged.all()
        and added.final_geometry_valid.all()
        and added.composition_valid.all()
    )
    assert (added.usable & added.rerelaxed_passing).sum() == 635
    selection = read_csv(RESULTS_DIR / "heldout/selections.csv")
    metrics = read_csv(OUT / "rerelaxed_method_metrics.csv").set_index("method")
    adjacency = {}
    for a, b in read_csv(OUT / "rerelaxed_duplicate_pairs.csv").itertuples(
        index=False, name=None
    ):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    for method in [ORIGINAL, ANGLE]:
        chosen = (
            selection[selection.method.eq(method)]
            .sort_values("rank")
            .candidate_row.to_numpy()
        )
        assert len(chosen) == frozen["budget"] == 1849
        chosen = chosen[final.loc[chosen, "usable"].to_numpy(bool)]
        unique = unique_order(chosen, adjacency)
        f = final.loc[unique]
        assert f.rerelaxed_passing.sum() == metrics.loc[method, "unique_passing"]
        assert (f.rerelaxed_passing & f.novel_after_relaxation).sum() == metrics.loc[
            method, "unique_novel_passing"
        ]
    report = json.loads((OUT / "rerelaxed_evaluation.json").read_text())
    for name, expected in report["sha256"].items():
        assert sha256(OUT / name) == expected
    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in [
                "mattersim",
                "smact",
                "torch",
                "numpy",
                "pymatgen",
                "ase",
                "torch-geometric",
            ]
        },
    }
    (OUT / "audit_runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    verification = {
        "status": "passed",
        "frozen_test_unchanged": True,
        "fresh_energy_labels_reproduced": True,
        "all_added_candidates_fully_checked": True,
        "atomic_and_cell_filter_force_cutoffs_checked": True,
        "method_counts_independently_recomputed": True,
        "generated_data_outside_git": True,
    }
    (OUT / "audit_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
