"""Staged SOC workflow. No proxy packages are imported for unrelated stages.

python -m reference_baselines.reference_set_comparison.pipeline --help
"""

import argparse
from pathlib import Path

from .common import fingerprint, read_json, receipt, timed, write_json
from .configuration import load_config, run_identity


DEPENDENCIES = {"embed": ["prepare"], "score": ["embed"], "select": ["score"],
                "smoke-model": ["prepare"],
                "phases": ["embed"], "predict": ["select", "phases"], "hull": ["select", "phases"],
                "megnet_hull": ["select", "phases"], "esen_hull": ["select", "phases"],
                "report": ["select", "megnet_hull", "esen_hull"]}


def check_stage(output, stage):
    marker = output / f"{stage}.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Run prerequisite stage first: {stage}")
    saved = read_json(marker)
    receipt(marker, saved["identity"])


def execute(config, stage, model=None, max_items=None, retry_errors=False):
    from reference_baselines.core.io import sha256
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(run_identity(config))
    run_file = output / "run.json"
    if run_file.exists():
        if read_json(run_file)["identity"] != identity:
            raise ValueError("Configuration changed: use a new output directory (never mix runs)")
    else:
        write_json(run_file, {"identity": identity, "config": run_identity(config), "schema": 1})
    def check_dependencies(current):
        required_stages = (["select", f"{model}_hull"] if current == "report" and model
                           else DEPENDENCIES.get(current, []))
        for required in required_stages:
            check_dependencies(required)
            check_stage(output, required)
    check_dependencies(stage)
    if stage not in ("prepare", "fetch-hull", "download-model", "status"):
        prepared = read_json(output / "prepare.json")
        for source in prepared["source_files"]:
            if sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"Modified release source: {source['path']}")
    with timed(output, stage + (":" + model if model else "")):
        if stage == "prepare":
            from .release import prepare_release
            prepare_release(config)
        elif stage == "embed":
            from .embeddings import embed_candidates
            embed_candidates(config)
        elif stage == "score":
            from .indexing import score_candidates
            score_candidates(config)
        elif stage == "select":
            from .selection import select_candidates
            select_candidates(config)
        elif stage == "fetch-hull":
            from reference_baselines.datasets.fetch_hull import fetch_hull
            fetch_hull(Path(config["hull_database"]).parent)
        elif stage == "phases":
            from .phase_bank import prepare_phases
            prepare_phases(config)
        elif stage == "download-model":
            from .proxies import download_model
            download_model(config, model)
        elif stage == "predict":
            from .proxies import predict
            return predict(config, model, max_items, retry_errors)
        elif stage == "smoke-model":
            from .proxies import smoke_model
            smoke_model(config, model)
        elif stage == "hull":
            from .hulls import evaluate_hulls
            evaluate_hulls(config, model)
        elif stage == "report":
            from .reporting import generate_reports
            generate_reports(config, model=model)
        elif stage == "status":
            for path in sorted(output.glob("*.json")):
                data = read_json(path)
                if "prediction_status" in path.name:
                    print(path.name, data)
                elif path.name != "run.json":
                    print(path.name)
            print("Reports:", output / "reports" / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "embed", "score", "select", "fetch-hull", "phases", "download-model", "smoke-model", "predict", "hull", "report", "status"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.yml"))
    parser.add_argument("--model", choices=["megnet", "esen"],
                        help="Model for inference/hulls, or optional single-model report (default report uses both)")
    parser.add_argument("--start-batch", type=int)
    parser.add_argument("--end-batch", type=int)
    parser.add_argument("--output", type=Path, help="New output directory when changing the batch range")
    parser.add_argument("--max-items", type=int, help="Predict at most N new structures; resubmit to continue")
    parser.add_argument("--retry-errors", action="store_true", help="Retry cached failed predictions in this model environment")
    args = parser.parse_args()
    if args.stage in ("download-model", "smoke-model", "predict", "hull") and args.model is None:
        parser.error("This stage requires --model")
    if args.max_items is not None and args.max_items < 1:
        parser.error("--max-items must be positive")
    config = load_config(args.config, args.start_batch, args.end_batch)
    if args.output is not None:
        from reference_baselines.paths import ROOT
        config["output"] = str((ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve())
    execute(config, args.stage, args.model, args.max_items, args.retry_errors)


if __name__ == "__main__":
    main()
