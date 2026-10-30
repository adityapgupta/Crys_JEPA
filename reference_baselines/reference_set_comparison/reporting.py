"""Reports with explicit denominators and no full-population inference from selected rows."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from .common import database, fingerprint, finish, read_json, receipt, write_json


GROUP_KEYS = ["scope", "group", "method", "policy", "fraction"]


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def success_metrics(table, budget, threshold=.1):
    valid = table.hull_status.eq("ok") & np.isfinite(pd.to_numeric(table.e_above_hull_proxy, errors="coerce"))
    known = table.loc[valid]
    energy = known.e_above_hull_proxy.astype(float)
    good = (energy <= 0) if threshold == 0 else (energy < threshold)
    successes = int(good.sum())
    unknown = len(table) - len(known)
    return {"selected_count": len(table), "evaluated_count": len(known), "unknown_count": unknown,
            "stable_count": successes, "valid_label_coverage": safe_fraction(len(known), len(table)),
            "precision_evaluated": safe_fraction(successes, len(known)),
            "stable_yield_per_requested_slot": safe_fraction(successes, budget),
            "stable_yield_lower_bound": safe_fraction(successes, budget),
            "stable_yield_upper_bound": safe_fraction(successes + unknown, budget),
            "median_hull_distance": float(energy.median()) if len(energy) else None,
            "mean_hull_distance": float(energy.mean()) if len(energy) else None,
            "exact_threshold_count": int((energy == threshold).sum()),
            "below_minus_0_5_count": int((energy < -.5).sum()),
            "chemical_systems": int(table.chemical_system.nunique()),
            **{f"hull_quantile_{p}": float(energy.quantile(p)) if len(energy) else None for p in (0, .1, .25, .5, .75, .9, 1)}}


def paired_interval(original, alternative, repeats, seed):
    """Paired resampling of chemical systems; complete labels only, with coverage reported separately."""
    def aggregate(frame):
        known = frame[frame.hull_status.eq("ok")].copy()
        known["success"] = known.e_above_hull_proxy.astype(float) < .1
        return known.groupby("chemical_system").success.agg(["sum", "count"])
    a, b = aggregate(original), aggregate(alternative)
    systems = a.index.union(b.index)
    if len(systems) < 2:
        return None, None
    a = a.reindex(systems, fill_value=0).to_numpy(dtype=float)
    b = b.reindex(systems, fill_value=0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0, repeats, 100):
        draws = rng.integers(0, len(systems), (min(100, repeats-start), len(systems)))
        aa, bb = a[draws].sum(1), b[draws].sum(1)
        valid = (aa[:, 1] > 0) & (bb[:, 1] > 0)
        differences.extend((bb[valid, 0] / bb[valid, 1] - aa[valid, 0] / aa[valid, 1]).tolist())
    return tuple(np.quantile(differences, [.025, .975])) if differences else (None, None)


def pooled_requests(requests, selections):
    grouped = requests[requests.scope.eq("turn")].groupby(["method", "policy", "fraction"], as_index=False)
    pooled = grouped[["population", "budget", "selected_count", "scoreable_count", "raw_unique_without_refill", "raw_selected_count"]].sum()
    pooled["scope"], pooled["group"] = "pooled", -1
    extra = selections[selections.scope.eq("turn")].copy()
    extra["scope"], extra["group"] = "pooled", -1
    return pd.concat([requests, pooled], ignore_index=True), pd.concat([selections, extra], ignore_index=True)


def generate_reports(config, model=None):
    output = Path(config["output"])
    model_names = [model] if model is not None else list(config["models"])
    if any(name not in config["models"] for name in model_names):
        raise ValueError("Report model is not configured")
    report_dir = output / "reports" / model if model is not None else output / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    identity = fingerprint({"selection": sha256(output / "select.json"), "labels": {
        name: sha256(output / f"{name}_hull.json") for name in model_names},
        "code": sha256(Path(__file__)), "bootstrap_repeats": config["bootstrap_repeats"]})
    marker = report_dir / "complete.json"
    if marker.exists() and read_json(marker)["identity"] == identity and receipt(marker, identity):
        return
    requests = pd.read_csv(output / "selection_requests.csv")
    selections = pd.read_csv(output / "selections.csv")
    requests, selections = pooled_requests(requests, selections)
    ids = {key: set(group.candidate_row) for key, group in selections.groupby(GROUP_KEYS)}
    labels = {name: pd.read_csv(output / f"{name}_labels.csv", keep_default_na=False).set_index("candidate_row", verify_integrity=True)
              for name in model_names}
    for table in labels.values():
        table["e_above_hull_proxy"] = pd.to_numeric(table.e_above_hull_proxy, errors="coerce")
    metrics, overlaps, paired, consensus, membership = [], [], [], [], []
    for request in requests.to_dict("records"):
        key = tuple(request[k] for k in GROUP_KEYS)
        chosen = ids.get(key, set())
        baseline_key = (request["scope"], request["group"], "chemical", request["policy"], request["fraction"])
        base = ids.get(baseline_key, set())
        shared, baseline_only, alternative_only = chosen & base, base - chosen, chosen - base
        overlap = {**request, "shared": len(shared), "baseline_only": len(baseline_only),
                   "alternative_only": len(alternative_only), "selection_jaccard": safe_fraction(len(shared), len(chosen | base)),
                   "baseline_selected_alternative_not_percent": 100*len(baseline_only)/len(base) if base else None,
                   "alternative_selected_baseline_not_percent": 100*len(alternative_only)/len(chosen) if chosen else None}
        overlaps.append(overlap)
        for row in sorted(chosen | base):
            membership.append({**{k: request[k] for k in GROUP_KEYS}, "candidate_row": row,
                               "selected_baseline": row in base, "selected_alternative": row in chosen})
        for name, table in labels.items():
            selected = table.loc[sorted(chosen)]
            original = table.loc[sorted(base)]
            for threshold in (0., .05, .1):
                metrics.append({**request, "proxy": name, "threshold_eV_per_atom": threshold,
                                "threshold_rule": "<=" if threshold == 0 else "<",
                                "duplicate_fraction_raw_selection": 1 - request["raw_unique_without_refill"]/request["raw_selected_count"] if request["raw_selected_count"] else None,
                                **success_metrics(selected, request["budget"], threshold)})
            if request["method"] != "chemical":
                low, high = paired_interval(original, selected, config["bootstrap_repeats"], config["seed"])
                original_stats = success_metrics(original, len(base))
                selected_stats = success_metrics(selected, request["budget"])
                precision_a, precision_b = original_stats["precision_evaluated"], selected_stats["precision_evaluated"]
                row = {**request, "proxy": name,
                       "precision_difference_vs_chemical": precision_b-precision_a if precision_a is not None and precision_b is not None else None,
                       "paired_chemical_system_ci_low": low, "paired_chemical_system_ci_high": high}
                for cohort, members in (("shared", shared), ("baseline_only", baseline_only), ("alternative_only", alternative_only)):
                    stats = success_metrics(table.loc[sorted(members)], len(members))
                    for field in ("selected_count", "stable_count", "precision_evaluated", "unknown_count"):
                        row[f"{cohort}_{field}"] = stats[field]
                paired.append(row)
        # Consensus is descriptive; both proxies must have valid labels for an agreement claim.
        if {"megnet", "esen"}.issubset(labels):
            a, b = labels["megnet"].loc[sorted(chosen)], labels["esen"].loc[sorted(chosen)]
            known = a.hull_status.eq("ok") & b.hull_status.eq("ok")
            agood, bgood = a.e_above_hull_proxy < .1, b.e_above_hull_proxy < .1
            both = int((known & agood & bgood).sum())
            neither = int((known & ~agood & ~bgood).sum())
            consensus.append({**request, "both_evaluated": int(known.sum()), "unknown_either": int((~known).sum()),
                              "both_pass": both, "neither_pass": neither,
                              "megnet_only_pass": int((known & agood & ~bgood).sum()),
                              "esen_only_pass": int((known & ~agood & bgood).sum()),
                              "agreement_fraction": safe_fraction(both+neither, int(known.sum()))})
    products = {"metrics.csv": pd.DataFrame(metrics), "overlap.csv": pd.DataFrame(overlaps),
                "paired_comparisons.csv": pd.DataFrame(paired),
                "selection_membership.csv": pd.DataFrame(membership)}
    if consensus:
        products["proxy_consensus.csv"] = pd.DataFrame(consensus)
    metric_table = products["metrics.csv"]
    for scope in ("batch", "turn", "pooled"):
        products[f"{scope}_metrics.csv"] = metric_table[metric_table.scope.eq(scope)]
    for filename, table in products.items():
        table.to_csv(report_dir / filename, index=False)
    primary = metric_table[metric_table.scope.eq("pooled") & metric_table.fraction.eq(.01)
                           & metric_table.threshold_eV_per_atom.eq(.1)]
    # Avoid requiring tabulate for a portable Markdown report.
    columns = ["method", "policy", "proxy", "budget", "stable_count", "evaluated_count", "precision_evaluated"]
    lines = [f"# Reference-selection experiment ({', '.join(model_names)})", "", "Primary: per-turn top 1%; fixed model-specific MP2023 phase banks.", "",
             "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(row[c]) for c in columns) + " |" for row in primary.to_dict("records"))
    lines += ["", "## Interpretation", "", "- Precision uses valid proxy labels; coverage and requested-slot yield bounds are reported separately.",
              "- A missing label is unknown, not stable. Unfilled selection slots cannot count as discoveries.",
              "- Thresholds are <0.1 and <0.05 eV/atom; the on/below-hull diagnostic uses <=0.",
              "- Pooled results aggregate selections made within ten-folder groups; per-folder selections are diagnostic.",
              "- Unique refill is performed within each selection group; pooled sets can still contain duplicates across groups.",
              "- The phase bank contains only fixed reference phases, never selected candidates.",
              "- Proxies share historical materials data; agreement is not independent DFT confirmation.",
              "- No population recall, PR curves, or full-population score correlations are claimed from selected-union labels.",
              "- Intervals resample chemical systems and are descriptive, without multiple-comparison adjustment."]
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with database(output / "runtime.sqlite") as db:
        try:
            runtime = pd.read_sql_query("SELECT * FROM runtime", db)
        except Exception:
            runtime = pd.DataFrame(columns=["stage", "seconds", "timestamp"])
    runtime.to_csv(report_dir / "runtime.csv", index=False)
    write_json(report_dir / "provenance.json", {
        "config": read_json(output / "run.json"),
        "models": {name: read_json(output / f"{name}_model.json") for name in model_names},
        "report_proxies": model_names, "cross_proxy_consensus_available": len(model_names) > 1,
        "phases": read_json(output / "phases.json"), "selection": read_json(output / "select.json"),
        "interval": "paired chemical-system bootstrap, complete labels; descriptive",
        "threshold_eV_per_atom": .1, "rule": "strictly less than", "sampled_label_scope": "selected union plus random controls"})
    outputs = [report_dir / f for f in products] + [report_dir / name for name in ("REPORT.md", "runtime.csv", "provenance.json")]
    finish(marker, identity, outputs)
    print(f"Reports: {report_dir / 'REPORT.md'}", flush=True)
