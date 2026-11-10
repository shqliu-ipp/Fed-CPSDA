"""Edit CONFIG and run directly. Client DP and window DP stay separate."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_private_moments as experiment
import run_fed_private_adapter as previous


CONFIG = {
    **previous.CONFIG,
    "output_dir": previous.legacy_runner.CODE_ROOT / "runs" / "fed_private_moments_v2_final_seed42",
    "public_checkpoint_root": previous.CONFIG["output_dir"],
    "methods": experiment.METHODS,
    "whitening_floor": 0.01,
    "feature_kind": "backbone_pca",
    "moment_dim": 16,
    "ridge": 0.02,
    "use_unlabeled_covariance": True,
    "evaluate_test": True,
    "epsilon": 8.,
    "delta": 1e-5,
}


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    if (not args.public_source_acknowledged or not math.isfinite(args.ridge) or args.ridge <= 0
            or not math.isfinite(args.whitening_floor) or args.whitening_floor <= 0
            or args.target_clients < 1 or not args.methods
            or args.feature_kind not in {"backbone_pca", "projector"} or args.moment_dim < 1
            or len(set(args.methods)) != len(args.methods)
            or set(args.methods) - set(experiment.METHODS + experiment.DIAGNOSTIC_METHODS)):
        raise ValueError("Invalid one-release moment configuration")
    args.output_dir = Path(args.output_dir)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    base = experiment.adapter.base
    hashes = previous.legacy_runner.fingerprints()
    for name in ["Main.py", "fed_private_adapter.py", "fed_private_moments.py", "run_fed_private_moments.py"]:
        hashes[name] = hashlib.sha256((previous.legacy_runner.CODE_ROOT / name).read_bytes()).hexdigest()
    base.save_experiment_config(args.output_dir, {"args": vars(args), "method_version": experiment.VERSION, "source_sha256": hashes})
    base.write_json(args.output_dir / "privacy_scope.json", {
        "epsilon_per_adapter": args.epsilon, "delta": args.delta,
        "public_condition": "source domain, source encoder/whitening, roster, fixed allocation, window counts and label-status metadata",
        "ClientDPRidge": "complete target-client data replacement",
        "ClientDPHeadSelect": "pure epsilon client DP: exponential selection from source-only public classifiers; delta=0",
        "WindowDPRidge": "one preprocessed target window replacement; does NOT protect whole recordings or clients",
        "aggregator": "trusted central simulation, no cryptographic secure aggregation",
        "publication": "each adapter is an alternative; previous runs and simultaneous adapters require composition",
        "cross_unit_composition": "no joint client-DP epsilon=8 claim for publication including a window-DP adapter",
        "restricted": "all diagnostics, nonprivate models and evaluation metrics; do not publish the entire output directory"})
    rows = []
    tasks = previous.legacy_runner.make_tasks(args)
    for task in tasks:
        directory = args.output_dir / task.name
        result_file = directory / "restricted_results.json"
        if result_file.exists():
            result = json.loads(result_file.read_text(encoding="utf-8"))
            if [r["method"] for r in result] != list(args.methods):
                raise FileExistsError("Partial task exists; use a new output directory")
        else:
            result = experiment.run_task(task, args, device, directory)
        rows.extend(result)
        base.write_json(args.output_dir / "restricted_results.json", rows)
        base.write_csv(args.output_dir / "restricted_results.csv", rows)
        base.write_json(args.output_dir / "progress.json", {"completed": len(rows), "total": len(tasks) * len(args.methods)})
    return rows


if __name__ == "__main__":
    main()
