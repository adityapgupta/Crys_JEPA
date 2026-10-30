"""Read-only report analysis shared by notebook views for every proxy and batch range.

No model packages or raw cluster databases are needed. Source report receipts are
verified before use; derived tables/figures go into a separate analysis folder.
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from reference_baselines.core.io import sha256
from .common import read_json, write_json


@dataclass
class ReportBundle:
    directory: Path
    tables: dict
    provenance: dict

    @property
    def proxies(self):
        return tuple(sorted(self.tables["pooled_metrics"].proxy.unique()))


def load_report(directory):
    directory = Path(directory).resolve()
    receipt = read_json(directory / "complete.json")
    verified = set()
    for record in receipt["outputs"]:
        path = (directory / record["name"]).resolve()
        if path.parent != directory or sha256(path) != record["sha256"]:
            raise ValueError(f"Incomplete/modified report transfer: {record['name']}")
        verified.add(path.name)
    names = ["metrics", "pooled_metrics", "turn_metrics", "batch_metrics",
             "paired_comparisons", "overlap", "selection_membership", "runtime"]
    if "proxy_consensus.csv" in verified:
        names.append("proxy_consensus")
    required = {name + ".csv" for name in names} | {"provenance.json"}
    if not required.issubset(verified):
        raise ValueError(f"Missing verified report products: {required - verified}")
    return ReportBundle(directory, {name: pd.read_csv(directory / (name + ".csv"))
                                    for name in names}, read_json(directory / "provenance.json"))


def metric_rows(bundle, proxy=None, scope="pooled", policy="unique_refill", threshold=.1, fraction=None):
    table = bundle.tables["metrics"]
    keep = table.scope.eq(scope) & table.policy.eq(policy) & np.isclose(table.threshold_eV_per_atom, threshold)
    if proxy is not None:
        keep &= table.proxy.eq(proxy)
    if fraction is not None:
        keep &= np.isclose(table.fraction, fraction)
    return table.loc[keep].copy()


def audit_report(bundle):
    """Validate counts/denominators without treating unknown labels as failures."""
    m = bundle.tables["metrics"]
    keys = ["scope", "group", "method", "policy", "fraction", "proxy", "threshold_eV_per_atom"]
    if m.duplicated(keys).any():
        raise ValueError("Duplicated metric request")
    counts = m[["population", "budget", "selected_count", "evaluated_count", "unknown_count", "stable_count"]]
    if not (np.isfinite(counts).all().all() and (counts >= 0).all().all()
            and (counts == np.floor(counts)).all().all()):
        raise ValueError("Invalid metric counts")
    if not ((m.selected_count <= m.budget) & (m.budget <= m.population)
            & (m.stable_count <= m.evaluated_count)
            & (m.evaluated_count + m.unknown_count == m.selected_count)).all():
        raise ValueError("Inconsistent selected/evaluated/unknown counts")
    for column, expected in {
        "precision_evaluated": m.stable_count / m.evaluated_count.replace(0, np.nan),
        "valid_label_coverage": m.evaluated_count / m.selected_count.replace(0, np.nan),
        "stable_yield_lower_bound": m.stable_count / m.budget.replace(0, np.nan),
        "stable_yield_upper_bound": (m.stable_count + m.unknown_count) / m.budget.replace(0, np.nan),
    }.items():
        np.testing.assert_allclose(m[column], expected, equal_nan=True)
    request_keys = ["scope", "group", "policy", "fraction", "threshold_eV_per_atom"]
    if not (m.groupby(request_keys)[["budget", "population"]].nunique() <= 1).all().all():
        raise ValueError("Methods/proxies have different requested budgets/populations")

    members = bundle.tables["selection_membership"].copy()
    for name in ("selected_baseline", "selected_alternative"):
        if not pd.api.types.is_bool_dtype(members[name]):
            raise ValueError(f"Membership flags must be Boolean: {name}")
    group_keys = ["scope", "group", "method", "policy", "fraction"]
    if members.duplicated(group_keys + ["candidate_row"]).any():
        raise ValueError("Duplicated candidate membership")
    members["shared"] = members.selected_baseline & members.selected_alternative
    members["baseline_only"] = members.selected_baseline & ~members.selected_alternative
    members["alternative_only"] = ~members.selected_baseline & members.selected_alternative
    observed = members.groupby(group_keys)[["shared", "baseline_only", "alternative_only"]].sum()
    reported = bundle.tables["overlap"].set_index(group_keys)
    for name in ("shared", "baseline_only", "alternative_only"):
        np.testing.assert_array_equal(reported[name], observed.reindex(reported.index, fill_value=0)[name])
    np.testing.assert_allclose(reported.selection_jaccard,
        reported.shared / (reported.shared + reported.baseline_only + reported.alternative_only).replace(0, np.nan), equal_nan=True)

    if "proxy_consensus" in bundle.tables:
        c = bundle.tables["proxy_consensus"]
        categories = ["both_pass", "neither_pass", "megnet_only_pass", "esen_only_pass"]
        np.testing.assert_array_equal(c[categories].sum(axis=1), c.both_evaluated)
        np.testing.assert_array_equal(c.both_evaluated + c.unknown_either, c.selected_count)
        np.testing.assert_allclose(c.agreement_fraction,
            (c.both_pass + c.neither_pass) / c.both_evaluated.replace(0, np.nan), equal_nan=True)
        # Reconcile with per-model metrics, using exact equal candidate selections.
        for name in ("megnet", "esen"):
            own = metric_rows(bundle, name, scope="pooled")
            joint = c[c.scope.eq("pooled") & c.policy.eq("unique_refill")]
            joined = own.merge(joint, on=group_keys, suffixes=("_metric", "_joint"), validate="one_to_one")
            assert (joined.both_evaluated <= joined.evaluated_count).all()
            common_pass = joined.both_pass + joined[name + "_only_pass"]
            assert (common_pass <= joined.stable_count).all()
            assert (joined.stable_count <= common_pass + joined.evaluated_count - joined.both_evaluated).all()

    # Pooled is an aggregation of primary turn requests, never an extra replicate.
    selected = metric_rows(bundle)
    turn = metric_rows(bundle, scope="turn")
    aggregate_keys = ["method", "policy", "fraction", "proxy"]
    aggregate_counts = ["population", "budget", "selected_count", "evaluated_count", "unknown_count", "stable_count"]
    summed = turn.groupby(aggregate_keys)[aggregate_counts].sum().sort_index()
    pooled = selected.set_index(aggregate_keys)[aggregate_counts].sort_index()
    pd.testing.assert_frame_equal(summed, pooled, check_dtype=False)
    return selected.groupby(["proxy", "fraction"]).agg(
        population=("population", "first"), budget=("budget", "first"), methods=("method", "nunique"),
        minimum_coverage=("valid_label_coverage", "min"), maximum_unknown_per_method=("unknown_count", "max"),
        minimum_selected=("selected_count", "min"))


def primary_table(bundle, proxy, fraction=.01):
    primary = metric_rows(bundle, proxy, fraction=fraction)
    ci = bundle.tables["paired_comparisons"]
    ci = ci[ci.scope.eq("pooled") & ci.policy.eq("unique_refill") & ci.proxy.eq(proxy) & np.isclose(ci.fraction, fraction)]
    table = primary.merge(ci[["method", "paired_chemical_system_ci_low", "paired_chemical_system_ci_high"]],
                          on="method", how="left", validate="one_to_one")
    baseline = table.set_index("method").loc["chemical", "precision_evaluated"]
    table["gain_vs_chemical_pp"] = 100 * (table.precision_evaluated - baseline)
    return table.sort_values(["precision_evaluated", "method"], ascending=[False, True]).reset_index(drop=True)


def budget_winners(bundle):
    rows = []
    for (proxy, fraction), table in metric_rows(bundle).groupby(["proxy", "fraction"], sort=True):
        for metric in ("precision_evaluated", "stable_yield_lower_bound"):
            best = table[np.isclose(table[metric], table[metric].max())]
            rows.append({"proxy": proxy, "fraction": fraction, "criterion": metric,
                         "method": ", ".join(sorted(best.method)), "value": best.iloc[0][metric],
                         "budget": int(best.iloc[0].budget)})
    return pd.DataFrame(rows)


def consensus_table(bundle, fraction=.01):
    c = bundle.tables["proxy_consensus"]
    c = c[c.scope.eq("pooled") & c.policy.eq("unique_refill") & np.isclose(c.fraction, fraction)].copy()
    c["both_pass_yield_lower"] = c.both_pass / c.budget
    c["both_pass_yield_upper"] = (c.both_pass + c.unknown_either) / c.budget
    c["both_pass_precision_common"] = c.both_pass / c.both_evaluated.replace(0, np.nan)
    return c.sort_values(["both_pass_yield_lower", "method"], ascending=[False, True])


def family_color(method):
    return ("#2463a6" if method.startswith("knn_") else "#c87d32" if method.startswith("union_")
            else "#40464d" if method == "chemical" else "#999999")


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=170, bbox_inches="tight")
    return fig


def plot_primary(bundle, proxy, directory):
    import matplotlib.pyplot as plt
    table = primary_table(bundle, proxy)
    fig, (ax, ci_ax) = plt.subplots(1, 2, figsize=(13, 5.2))
    y = np.arange(len(table))
    ax.barh(y, 100*table.precision_evaluated, color=[family_color(m) for m in table.method])
    for i, row in enumerate(table.itertuples()):
        ax.text(100*row.precision_evaluated+1, i,
                f"{row.stable_count}/{row.evaluated_count}" + (f" (+{row.unknown_count} unknown)" if row.unknown_count else ""), va="center", fontsize=8)
    ax.set(yticks=y, yticklabels=table.method, xlim=(0, 108), xticks=[0, 20, 40, 60, 80, 100],
           xlabel="Precision among valid labels (%)", title=f"{proxy.upper()}: top 1%, unique refill")
    ax.invert_yaxis()
    compared = table[table.method.ne("chemical")]
    for i, row in enumerate(compared.itertuples()):
        ci_ax.hlines(i, 100*row.paired_chemical_system_ci_low, 100*row.paired_chemical_system_ci_high, color="#737b85", lw=2)
        ci_ax.scatter(row.gain_vs_chemical_pp, i, color=family_color(row.method), zorder=3)
    ci_ax.axvline(0, color="#40464d", ls="--", lw=1)
    ci_ax.set(yticks=np.arange(len(compared)), yticklabels=compared.method,
              xlabel="Precision difference vs chemical (percentage points)", title="Descriptive paired chemical-system 95% intervals")
    ci_ax.invert_yaxis()
    for axis in (ax, ci_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed proxy hull distance <0.1 eV/atom; intervals use valid labels, not unknown-label bounds")
    return save_figure(fig, directory, f"{proxy}_primary_comparison.png")


def plot_budgets(bundle, proxy, directory):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    table = metric_rows(bundle, proxy)
    for method, group in table.groupby("method", sort=True):
        group = group.sort_values("fraction")
        for ax, metric in zip(axes, ("precision_evaluated", "stable_yield_lower_bound")):
            ax.plot(100*group.fraction, 100*group[metric], marker="o", label=method,
                    ls="--" if method in ("chemical", "random_candidates") else "-")
        axes[1].fill_between(100*group.fraction, 100*group.stable_yield_lower_bound,
                             100*group.stable_yield_upper_bound, alpha=.06)
    for ax, title in zip(axes, ("Precision among valid labels", "Passing crystals / requested slots")):
        ax.set(xlabel="Selected fraction (%)", ylabel="Percent", title=title, xticks=[1, 5, 10], ylim=(0, 100))
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{proxy.upper()}: budget sensitivity; shading treats unknown labels as failures/successes")
    return save_figure(fig, directory, f"{proxy}_budget_sensitivity.png")


def plot_batches(bundle, proxy, directory):
    import matplotlib.pyplot as plt
    rows = metric_rows(bundle, proxy, scope="batch", fraction=.01)
    matrix = 100*rows.pivot(index="group", columns="method", values="stable_yield_lower_bound")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set(xticks=np.arange(len(matrix.columns)), xticklabels=matrix.columns,
           yticks=np.arange(len(matrix)), yticklabels=matrix.index, ylabel="Eval folder")
    ax.tick_params(axis="x", rotation=35)
    for y, values in enumerate(matrix.to_numpy()):
        for x, value in enumerate(values):
            ax.text(x, y, f"{value:.0f}", ha="center", va="center", color="white" if value >= 70 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Proxy-stable / requested slots (%)")
    fig.suptitle(f"{proxy.upper()}: per-folder top-1% diagnostics; unknown labels contribute no confirmed successes")
    return save_figure(fig, directory, f"{proxy}_per_batch_diagnostics.png")


def plot_overall(bundle, directory):
    import matplotlib.pyplot as plt
    consensus = consensus_table(bundle).set_index("method").sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    y = np.arange(len(consensus))
    for offset, proxy, color in ((-.16, "megnet", "#2463a6"), (.16, "esen", "#b7464e")):
        t = primary_table(bundle, proxy).set_index("method").reindex(consensus.index)
        axes[0].barh(y+offset, 100*t.stable_yield_lower_bound, height=.29, color=color, label=proxy.upper())
        axes[0].hlines(y+offset, 100*t.stable_yield_lower_bound, 100*t.stable_yield_upper_bound, color=color, lw=3)
    axes[0].set(xlabel="Proxy-stable / requested slots (%)", xlim=(0, 100), title="Same selections, different proxy judgements")
    axes[0].legend(frameon=False)
    left = np.zeros(len(consensus))
    for field, label, color in (("both_pass", "Both pass", "#2a8d72"), ("megnet_only_pass", "MEGNet only", "#75a6d6"),
                                ("esen_only_pass", "eSEN only", "#d47b83"), ("neither_pass", "Neither passes", "#d7d7d7"),
                                ("unknown_either", "Unknown in either", "#e3b257")):
        width = 100*consensus[field]/consensus.budget
        axes[1].barh(y, width, left=left, label=label, color=color)
        left += width.to_numpy()
    axes[1].set(xlabel="Percent of requested slots", xlim=(0, 100), title="Cross-proxy labels, without averaging models")
    axes[1].legend(loc="lower center", bbox_to_anchor=(.5, -.29), ncol=3, fontsize=8, frameon=False)
    for ax in axes:
        ax.set(yticks=y, yticklabels=consensus.index)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Top 1%, unique refill, fixed-hull distance <0.1 eV/atom; line segments are unknown-label bounds")
    return save_figure(fig, directory, "overall_proxy_comparison.png")


def export_analysis(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    products = {"coverage_audit.csv": audit_report(bundle).reset_index(), "budget_winners.csv": budget_winners(bundle)}
    for proxy in bundle.proxies:
        products[proxy + "_primary.csv"] = primary_table(bundle, proxy)
    if "proxy_consensus" in bundle.tables:
        products["consensus_primary.csv"] = consensus_table(bundle)
    for name, table in products.items():
        table.to_csv(directory / name, index=False)
    write_json(directory / "analysis_provenance.json", {
        "report_receipt_sha256": sha256(bundle.directory / "complete.json"),
        "analysis_source_sha256": sha256(Path(__file__)), "proxies": list(bundle.proxies),
        "primary": {"scope": "pooled turn selections", "policy": "unique_refill", "fraction": .01, "threshold": .1},
        "derived_tables": {name: sha256(directory / name) for name in products}})
