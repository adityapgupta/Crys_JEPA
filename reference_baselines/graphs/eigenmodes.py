"""Compute checked Laplacian eigenmodes in a time-bounded worker."""

import argparse
import json
import multiprocessing
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh, cg, LinearOperator

from reference_baselines.core.settings import baseline_settings
from reference_baselines.paths import BASELINE_RESULTS_DIR, RESULTS_DIR


def solve(metric, modes, solver):
    try:
        cfg = baseline_settings()
        name = (
            f"reference_graph_k{cfg['graph_k']}.npz"
            if metric == "euclidean"
            else f"reference_cosine_graph_k{cfg['graph_k']}.npz"
        )
        graph = sparse.load_npz(RESULTS_DIR / name).astype(float)
        degree = np.asarray(graph.sum(1)).ravel()
        if np.any(degree <= 0):
            raise ValueError("Eigenmode calculation requires positive graph degrees.")
        scale = sparse.diags(1 / np.sqrt(degree))
        laplacian = sparse.eye(len(degree)) - scale @ graph @ scale
        components = int(
            connected_components(graph, directed=False, return_labels=False)
        )
        start = time.perf_counter()
        if solver == "shift-invert":
            values, vectors = eigsh(
                laplacian,
                k=modes + components,
                sigma=-1e-4,
                which="LM",
                tol=1e-7,
                maxiter=3000,
                v0=np.random.default_rng(cfg["seed"]).normal(size=len(degree)),
            )
        else:
            shift_value = 0.01
            shift = laplacian + shift_value * sparse.eye(len(degree))

            def inverse(vector):
                solution, info = cg(shift, vector, rtol=1e-10, atol=0.0, maxiter=1000)
                if info:
                    raise RuntimeError(f"Inner shifted solve failed: {info}")
                return solution

            operator = LinearOperator(shift.shape, matvec=inverse, dtype=np.float64)
            values, vectors = eigsh(
                laplacian,
                k=modes + components,
                sigma=-shift_value,
                which="LM",
                OPinv=operator,
                tol=1e-7,
                maxiter=3000,
                v0=np.random.default_rng(cfg["seed"]).normal(size=len(degree)),
            )
        order = np.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
        residual = np.linalg.norm(laplacian @ vectors - vectors * values, axis=0)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(len(values)), atol=1e-6)
        if np.max(residual) > 1e-6 or values.min() < -1e-7:
            raise ValueError("Unreliable eigenpair residual or negative eigenvalue.")
        np.savez_compressed(
            BASELINE_RESULTS_DIR / f"{metric}_eigenmodes.npz",
            values=values,
            vectors=vectors,
            degree=degree,
        )
        report = {
            "status": "passed",
            "graph_metric": metric,
            "components": components,
            "eigenvalues": values.tolist(),
            "residual_norms": residual.tolist(),
            "seconds": time.perf_counter() - start,
            "solver": solver + "; full known-reference normalized Laplacian",
            "reference_count": len(degree),
            "second_eigenvalue": float(values[1]),
            "graph_uses_candidate_nodes": False,
        }
    except Exception as exc:
        report = {
            "status": "failed",
            "graph_metric": metric,
            "error": f"{type(exc).__name__}: {exc}",
        }
    (BASELINE_RESULTS_DIR / f"{metric}_eigenmodes.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph-metric", choices=["euclidean", "cosine"], required=True
    )
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument(
        "--solver",
        choices=["iterative-shift", "shift-invert"],
        default="iterative-shift",
    )
    args = parser.parse_args()
    BASELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=solve,
        args=(args.graph_metric, baseline_settings()["spectral_modes"], args.solver),
    )
    process.start()
    process.join(args.max_seconds)
    path = BASELINE_RESULTS_DIR / f"{args.graph_metric}_eigenmodes.json"
    if process.is_alive():
        process.terminate()
        process.join(10)
        path.write_text(
            json.dumps(
                {
                    "status": "time_limit",
                    "graph_metric": args.graph_metric,
                    "limit_seconds": args.max_seconds,
                    "claim": "no full spectral result from this attempt",
                },
                indent=2,
            )
            + "\n"
        )
        raise SystemExit(
            "Eigenmode calculation exceeded the local time limit. Cluster command supports a longer limit."
        )
    if process.exitcode != 0:
        path.write_text(
            json.dumps(
                {"status": "worker_failed", "exitcode": process.exitcode}, indent=2
            )
            + "\n"
        )
        raise SystemExit("Eigenmode worker failed; no spectral result claimed.")
    result = json.loads(path.read_text())
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
