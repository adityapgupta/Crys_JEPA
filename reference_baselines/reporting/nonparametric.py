"""Write a simple-English report and a baseline/cluster readiness record."""

from datetime import datetime, timezone
import json
import os

import pandas as pd

from reference_baselines.core.io import sha256
from reference_baselines.paths import (
    BASELINE_RESULTS_DIR,
    CONFIG_DIR,
    DOCS_DIR,
    PACKAGE_DIR,
    ROOT,
)


def main():
    out = BASELINE_RESULTS_DIR
    evaluation = json.loads((out / "evaluation.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    examples = json.loads((out / "reference_example_checks.json").read_text())
    assert (
        evaluation["status"] == verification["status"] == examples["status"] == "passed"
    )
    assert (
        not evaluation["test_performance_evaluated"]
        and not evaluation["neural_models_trained"]
    )
    metrics = pd.read_csv(out / "validation_metrics.csv")
    primary = metrics[
        metrics.selection_fraction.eq(0.05) & metrics.threshold_eV_per_atom.eq(0.1)
    ].copy()
    assert len(primary) == evaluation["method_count"]
    primary = primary.sort_values(
        ["unique_yield_per_slot", "method"], ascending=[False, True]
    )
    selected = [
        ("crys_jepa", "Original Crys-JEPA"),
        ("intersection_fallback_k5", "Euclidean intersection + fallback, k=5"),
        ("cosine_knn_raw_k5", "Cosine neighbours, raw-distance score, k=5"),
        ("cosine_knn_cos_k5", "Cosine neighbours and cosine score, k=5"),
        ("cosine_intersection_fallback_k5", "Cosine intersection + fallback, k=5"),
        ("strict_chemical_mean", "Original mean with strict on-hull references"),
        ("theta1_gate_then_original_score", "Angle gate, then original score"),
    ]
    families = [
        "cosine angle cutoff",
        "Laplacian resolvent",
        "Laplacian heat",
        "conductance reference sweep",
    ]
    spectral_done = bool(primary.family.eq("spectral retrieval").any())
    if spectral_done:
        families.append("spectral retrieval")
    for family in families:
        best = primary[primary.family.eq(family)].iloc[0]
        selected.append((best.method, "Exploratory best: " + family))
    table = primary.set_index("method")
    lines = [
        "# Distance, angle, and graph baseline experiments",
        "",
        "Current phase: fixed baselines. No candidate-only model, GCN, attention model, or transformer was trained.",
        "",
        "## Data and reference sources",
        "",
        "- Reference bank: 27,136 known MP20 training crystals. Every recorded reference hull distance is below 0.08 eV/atom. These are already near-stable known crystals.",
        "- Strict comparison bank: 11,958 of those known crystals with recorded hull distance at most 1e-8 eV/atom.",
        "- Generated crystals are queries. They never become reference nodes. Evaluation uses 37,438 VUN validation queries. Test outcome labels remain unused.",
        "- Stability checks use corrected MatterSim candidate energies against the fixed MP2023 DFT hull. They are proxy labels, not new DFT results.",
        "",
        "## What was run",
        "",
        "1. Original Crys-JEPA and Euclidean neighbours, including intersection and union.",
        "2. Cosine neighbours for k=5,10,20,50. Some variants keep the original raw-distance score; others explicitly use a cosine score.",
        "3. Cosine angle cutoffs theta=1,3,5,10,20,45 degrees, applied within the nearest k references. Empty sets either abstain or use an explicitly named fallback.",
        "4. Retrieval from the stricter known-reference bank.",
        "5. Full-reference-graph Laplacian resolvent and heat diffusion. These spread query seed weights across known reference nodes. Scores are weighted mean raw JEPA distances. Candidate energy labels are not used by these methods.",
        "6. Conductance-based graph sweeps. These choose among reference prefixes of size 5,10,20,50 using graph edges, then score the chosen references.",
        "7. Three random-reference controls for each matched pool-size/fallback rule, plus gate-only controls. These test whether reference choice adds value beyond set size or the angle gate itself.",
        "",
        "The core grid is in [nonparametric.toml](../configs/nonparametric.toml). Follow-up random and gate controls are in [random_controls.py](../baselines/random_controls.py). They were added after the initial results to test the cause of the gain. No supervised fitting is used.",
        "",
        "## Results at the same selection budget",
        "",
        "Each method can select up to 1,871 unique generated crystals: 5% of the validation set, rounded down. Repeated structures are removed in score order; we continue down the list to refill the budget. A passing crystal has proxy energy at most 0.1 eV/atom above the hull.",
        "",
        "| Method | Passing unique crystals | Yield per requested slot | Score coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, label in selected:
        row = table.loc[name]
        lines.append(
            f"| {label} | {int(row.near_stable_unique_count)} | {100 * row.unique_yield_per_slot:.1f}% | {100 * row.score_coverage:.1f}% |"
        )
    theta_best = primary[primary.family.eq("cosine angle cutoff")].iloc[0]
    gate = table.loc["theta1_gate_then_original_score"]
    random_theta = primary[
        primary.method.str.startswith("random_theta1_gate_size_seed")
    ].unique_yield_per_slot.mean()
    gate_random = primary[
        primary.method.str.startswith("theta1_gate_random_order_seed")
    ].unique_yield_per_slot.mean()
    lines += [
        "",
        f"The angle-family best setting is `{theta_best.method}`. It was selected after comparing the validation grid; it is exploratory, not an unseen-test winner.",
        f"The mean yield of the three random-reference controls with the same theta=1 gate and pool size is {100 * random_theta:.1f}%. The gate with random within-group order gives {100 * gate_random:.1f}% on average.",
        f"The gate followed by the original score gives {100 * gate.unique_yield_per_slot:.1f}%. This control is important: a gain can come from the angular gate as well as from which references were kept.",
        "",
        "At this budget the simple angle gate followed by the original score matches the angle-reference rule in yield. Their selections overlap in 1,818 of 1,871 crystals (97.2%), so the lists are not identical. This supports angular screening; it does not establish that new reference weights or a learned reference model are needed.",
        "",
        "Paired chemical-system bootstrap intervals are in `../results/nonparametric/validation_paired_intervals.csv`. They are descriptive and have no correction for testing many settings. Do not treat a single best grid result as a confirmed gain.",
        "",
        "## Checks and limits",
        "",
        f"- Checked {verification['full_sort_query_checks']} exact full-bank/strict-bank neighbour queries and {verification['scalar_angular_score_checks']} scalar scores independently. Checked all four cosine graphs. All passed.",
        "- All completed diffusion filters meet their recorded numerical error tolerance. Eight query distributions also match direct sums over known reference distances.",
        "- Angle and heat/resolvent operators pass small-matrix numerical controls. Both cluster eigenmode solvers pass a small-graph test.",
        (
            "- Accepted full-graph eigenvector retrieval results are included. Eigenpairs passed the recorded residual checks."
            if spectral_done
            else "- Long full-graph eigenvector attempts did not pass the local convergence/time checks. No eigenvector-retrieval result is included. Heat and resolvent methods above operate on the full Laplacian without this eigendecomposition."
        ),
        "- Other selection budgets, energy cutoffs, per-turn results, coverage, and extreme-low-energy exclusion checks are saved with the metrics.",
        "- Reference decomposition-phase recall is saved separately. Similar embedding vectors are not evidence that a reference is a correct decomposition phase.",
        "- MatterSim error, sparse hull coverage, and multiple comparisons remain limits. Independent DFT checks remain necessary.",
        "",
        "## Files and next work",
        "",
        "- [Readable reference examples](../results/nonparametric/reference_examples.csv) show candidate IDs and the known MP20 reference IDs they use.",
        "- [All metrics](../results/nonparametric/validation_metrics.csv), [angle coverage](../results/nonparametric/validation_theta_coverage.csv), and [phase recall](../results/nonparametric/decomposition_recall.csv). Result files stay local.",
        "- [Cluster runner](../cluster/run_baselines.sh), [Slurm template](../cluster/baselines.slurm), and [input list](../cluster/cluster_inputs.txt). The runner does not submit jobs.",
        "- Repeat the fixed protocol on the cluster if needed. Review the gate and matched-random controls before choosing a baseline. Keep test outcomes sealed until settings are fixed.",
        "- GCN and graph attention are later stages. The candidate-only model is not part of the current plan.",
        "",
        "## Sources behind the graph methods",
        "",
        "- [Cosine definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).",
        "- [Spectral clustering and graph Laplacians](https://arxiv.org/abs/0711.0189).",
        "- [PageRank beyond the Web](https://epubs.siam.org/doi/10.1137/140976649).",
        "- [Diffusion maps](https://www.sciencedirect.com/science/article/pii/S1063520306000546).",
        "",
    ]
    path = DOCS_DIR / "nonparametric_experiments.md"
    path.write_text("\n".join(lines))
    files = []
    for folder in [
        "baselines",
        "core",
        "graphs",
        "evaluation",
        "checks",
        "reporting",
        "cluster",
    ]:
        files.extend(
            f
            for f in (PACKAGE_DIR / folder).iterdir()
            if f.is_file() and f.suffix in {".py", ".sh", ".slurm", ".txt"}
        )
    files += [
        PACKAGE_DIR / "paths.py",
        CONFIG_DIR / "training.toml",
        CONFIG_DIR / "nonparametric.toml",
        ROOT / "tests/test_nonparametric.py",
        ROOT / "tests/test_reference_layout.py",
        PACKAGE_DIR / "requirements/nonparametric.txt",
    ]
    ready = {
        "status": "ready",
        "scope": "non-neural reference baselines and cluster reruns",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_count": len(primary),
        "candidate_only_model_needed": False,
        "neural_training_performed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "test_performance_evaluated": False,
        "explicit_eigenvector_retrieval_completed": spectral_done,
        "full_laplacian_filters_completed": ["heat", "resolvent", "conductance"],
        "sha256": {str(f.relative_to(ROOT)): sha256(f) for f in files},
    }
    (out / "readiness.json").write_text(json.dumps(ready, indent=2) + "\n")
    print("Baseline experiment report:", path)


if __name__ == "__main__":
    main()
