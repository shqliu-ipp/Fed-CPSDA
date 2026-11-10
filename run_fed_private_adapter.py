"""Direct, editable entry for corrected low-dimensional target-client DP pilots."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_private_adapter as experiment
import run_fed_cpsda as legacy_runner


CONFIG = {
    **legacy_runner.CONFIG,
    "output_dir": legacy_runner.CODE_ROOT / "runs" / "fed_cpsda_adapter_v2_seed42",
    "methods": experiment.METHODS,
    "seed": 42,
    "public_epochs": 30,
    "public_lr": 0.001,
    "adapter_dim": 8,
    "rounds": 4,
    "target_clients": 4,
    "head_steps": 80,
    "head_lr": 0.1,
    "head_clip": 0.2,
    "server_lr": 0.5,
    "prototype_scale": 1.0,
    "prototype_mix": 0.25,
    "prototype_weight": 0.15,
    "pseudo_start_round": 1,
    "pseudo_threshold": 0.85,
    "pseudo_floor": 0.7,
    "epsilon": 8.0,
    "delta": 1e-5,
    "public_source_acknowledged": True,
    "device": "auto",
}


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    if not args.public_source_acknowledged:
        raise ValueError("This protocol requires an explicitly public source domain")
    if (args.adapter_dim < 1 or args.public_epochs < 1 or args.head_steps < 1
            or args.target_clients < 1 or args.head_clip <= 0 or args.prototype_scale <= 0
            or not 0 <= args.prototype_mix <= 1 or not 0 < args.pseudo_floor <= args.pseudo_threshold <= 1
            or not args.methods or set(args.methods) - set(experiment.METHODS)):
        raise ValueError("Invalid private-adapter configuration")
    budget = experiment.JointBudget(args.epsilon, args.delta, args.rounds)
    args.output_dir = Path(args.output_dir)
    tasks = legacy_runner.make_tasks(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    fingerprints = legacy_runner.fingerprints()
    for name in ["Main.py", "fed_private_adapter.py", "run_fed_private_adapter.py"]:
        fingerprints[name] = hashlib.sha256((legacy_runner.CODE_ROOT / name).read_bytes()).hexdigest()
    cfg = {"args": vars(args), "method_version": experiment.VERSION, "source_sha256": fingerprints}
    # Reject incompatible runs before writing split/model artifacts.
    legacy_runner.industrial.exp.save_experiment_config(args.output_dir, cfg)
    private_methods = sum(m in {"FederatedHeadDP", "FedProtoDP", "FedCPSDA"} for m in args.methods)
    composed_rho = private_methods * budget.rho
    legacy_runner.industrial.exp.write_json(args.output_dir / "privacy_scope.json", {
        "per_method_epsilon": args.epsilon,
        "private_alternatives_trained": private_methods,
        "all_alternative_adapters_joint_epsilon_bound": composed_rho + 2 * math.sqrt(-composed_rho * math.log(args.delta)),
        "delta": args.delta,
        "scope": "per task, conditional public source, fixed target-client roster",
        "note": "Same-data reruns/seeds/tasks require additional composition; raw histories and test metrics are restricted research outputs, not DP publications"})
    rows = []
    for task in tasks:
        directory = args.output_dir / task.name
        result = directory / "results.json"
        if result.exists():
            previous = json.loads(result.read_text(encoding="utf-8"))
            if [r["method"] for r in previous] != list(args.methods):
                raise FileExistsError("Incomplete task exists; choose a new output directory to rerun it")
            rows.extend(previous)
        else:
            rows.extend(experiment.run_task(task, args, device, directory))
        base = legacy_runner.industrial.exp
        base.write_json(args.output_dir / "results.json", rows)
        base.write_csv(args.output_dir / "results.csv", rows)
        base.write_json(args.output_dir / "progress.json", {"completed": len(rows),
                        "total": len(tasks) * len(args.methods), "last_task": task.name})
    return rows


if __name__ == "__main__":
    main()
