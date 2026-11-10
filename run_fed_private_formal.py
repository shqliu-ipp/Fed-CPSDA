"""Formal, directly runnable public-source moment-DP experiment matrix."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
import statistics
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_private_moments as moments
import run_fed_private_moments as pilot
import run_three_dataset_experiment as industrial


ROOT = Path(__file__).resolve().parent
VERSION = "formal-public-source-moment-dp-v1"
CONFIG = {
    **pilot.CONFIG,
    "output_dir": ROOT / "runs" / "fed_private_formal_v1",
    "task_names": None,
    "datasets": ["PU", "DN", "WT"],
    "seeds": [42, 43, 44],
    "quick": False,
    "prepare_raw_data": False,
    "include_wt_installation": False,
    "include_wt_sensor": False,
    "public_epochs": 30,
    "public_lr": 0.001,
    "batch_size": 128,
    "num_workers": 0,
    "cpu_threads": 4,
    "device": "cuda:0",
    "main_epsilon": 8.,
    "epsilon_curve": [2., 4., 8., 16.],
    "run_covariance_ablation": True,
    "evaluate_test": True,
}


def fingerprints():
    names = ["Main.py", "unified_experiment.py", "experiment_protocol.py",
             "gear_experiment_protocol.py", "fed_cpsda_experiment.py",
             "fed_private_adapter.py", "fed_private_moments.py",
             "run_fed_cpsda.py", "run_fed_private_adapter.py",
             "run_fed_private_moments.py", "run_three_dataset_experiment.py",
             "run_fed_private_formal.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build_plan(args):
    if (not args.public_source_acknowledged or not args.evaluate_test
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in args.seeds)
            or args.methods != moments.METHODS or not args.use_unlabeled_covariance
            or args.public_epochs < 1 or args.batch_size < 1 or args.target_clients < 1
            or args.moment_dim < 1 or args.ridge <= 0 or args.whitening_floor <= 0
            or args.feature_kind not in {"backbone_pca", "projector"}
            or not args.epsilon_curve or len(set(args.epsilon_curve)) != len(args.epsilon_curve)):
        raise ValueError("Invalid formal matrix configuration")
    moments.adapter.JointBudget(args.main_epsilon, args.delta, 1)
    for epsilon in args.epsilon_curve:
        moments.adapter.JointBudget(epsilon, args.delta, 1)
    tasks = industrial.make_tasks(args)
    queues = {dataset: deque(t for t in tasks if t.dataset == dataset) for dataset in args.datasets}
    ordered = []
    while any(queues.values()):
        for dataset in args.datasets:
            if queues[dataset]:
                ordered.append(queues[dataset].popleft())
    stages = [("main", method, args.main_epsilon, True) for method in moments.METHODS]
    stages += [(f"epsilon_{epsilon:g}", "WindowDPRidge", epsilon, True)
               for epsilon in args.epsilon_curve if epsilon != args.main_epsilon]
    if args.run_covariance_ablation:
        stages += [("without_unlabeled_covariance", method, args.main_epsilon, False)
                   for method in ["RidgeNonPrivate", "WindowDPRidge"]]
    jobs = [(task, seed, stage, method, epsilon, covariance)
            for seed in args.seeds for task in ordered
            for stage, method, epsilon, covariance in stages]
    return ordered, jobs


def public_reference(task, args, device, hashes):
    base = moments.adapter.base
    reference = Path(args.public_checkpoint_root) / task.name
    reference.mkdir(parents=True, exist_ok=True)
    base.save_experiment_config(reference, {"args": vars(args), "method_version": VERSION,
                                            "source_sha256": hashes})
    checkpoint = reference / "public_source_model.pt"
    if checkpoint.exists():
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if loaded.get("public_source_only") is not True or loaded.get("version") != moments.adapter.VERSION:
            raise ValueError(f"Invalid public checkpoint: {checkpoint}")
        print(f"REUSE_PUBLIC {task.name} seed={args.seed}", flush=True)
        return
    base.seed_everything(args.seed)
    loaders, meta = moments.adapter.legacy.task_loaders(task, args, args.seed, reference)
    temporary = reference / "public_source_model.partial.pt"
    print(f"TRAIN_PUBLIC {task.name} seed={args.seed} epochs={args.public_epochs}", flush=True)
    started = time.time()
    model = moments.adapter.public_pretrain(loaders, meta["num_classes"], args, device, temporary)
    os.replace(temporary, checkpoint)
    base.write_json(reference / "restricted_training_status.json", {
        "completed": True, "seed": args.seed, "epochs": args.public_epochs,
        "elapsed_seconds": time.time() - started, "split_sha256": meta["split_sha256"]})
    del model, loaders
    if device.type == "cuda":
        torch.cuda.empty_cache()


def recover_adapter(task, args, device, directory, artifact_path):
    """Evaluate an already committed DP artifact without drawing new noise."""
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    method = args.methods[0]
    if artifact.get("version") != moments.VERSION or artifact.get("method") != method:
        raise ValueError(f"Incompatible interrupted artifact: {artifact_path}")
    data, clients, _, _, meta, provenance = moments.prepare_task(task, args, device, directory)
    if artifact["provenance"] != moments.public_release_provenance(provenance):
        raise ValueError("Recovered adapter uses another public model")
    private = method in {"ClientDPRidge", "WindowDPRidge"}
    client_unit = method in {"ClientDPRidge", "ClientRidgeNonPrivate"}
    labeled_count = len(data["labeled"][0])
    unlabeled_count = len(data["unlabeled"][0])
    count = len(clients) if client_unit else labeled_count + (unlabeled_count if args.use_unlabeled_covariance else 0)
    budget = moments.adapter.JointBudget(args.epsilon, args.delta, 1)
    unit = "complete_target_client" if client_unit else "single_target_window"
    if private and artifact["budget"] != moments.moment_budget_report(budget, count, unit):
        raise ValueError("Recovered adapter privacy parameters changed")
    weight, mean, matrix = artifact["weight"], artifact["whitening_mean"], artifact["whitening_matrix"]
    if not torch.isfinite(weight).all():
        raise ValueError("Nonfinite interrupted adapter")
    row = {"dataset": task.dataset, "task": task.name, "method": method,
           "seed": args.seed, "epsilon": args.epsilon if private else None,
           "delta": args.delta if private else None, "privacy_unit": unit if private else None,
           "privacy_units": count if private else None,
           "noise_std_normalized_mean": budget.sigma(count) if private else 0.,
           "release_rounds": int(private), "trainable_parameters": weight.numel(),
           "target_clients": len(clients), "target_labeled_windows": labeled_count,
           "target_unlabeled_windows": unlabeled_count,
           "uses_unlabeled_covariance": args.use_unlabeled_covariance,
           "feature_kind": args.feature_kind, "moment_dim": matrix.shape[1],
           "source_is_public": True, "evaluation_is_restricted": True,
           "version": moments.VERSION, **provenance}
    for split in ["val", "test"]:
        x, _, y = data[split]
        result = moments.metrics(moments.design(x, mean, matrix) @ weight, y, meta["num_classes"])
        row[split + "_acc"] = result["accuracy"]
        row[split + "_f1"] = result["macro_f1"]
        row[split + "_confusion_matrix"] = result["confusion_matrix"]
    moments.adapter.base.write_json(directory / "restricted_results.json", [row])
    print(f"RECOVER_ADAPTER {task.name} {method}; no new DP draw", flush=True)
    return row


def run_job(task, args, device, directory, hashes):
    base = moments.adapter.base
    directory.mkdir(parents=True, exist_ok=True)
    base.save_experiment_config(directory, {"args": vars(args), "method_version": VERSION,
                                            "source_sha256": hashes})
    method = args.methods[0]
    result_file = directory / "restricted_results.json"
    private = method in {"ClientDPRidge", "WindowDPRidge"}
    artifact = directory / f"{method}_{'released' if private else 'restricted'}_adapter.pt"
    if result_file.exists():
        rows = json.loads(result_file.read_text(encoding="utf-8"))
        if len(rows) != 1 or rows[0]["method"] != method or rows[0]["seed"] != args.seed:
            raise ValueError(f"Invalid completed result: {result_file}")
        if method != "PublicSourceOnly" and not artifact.exists():
            raise FileNotFoundError(f"Completed result is missing its adapter: {artifact}")
        return rows[0]
    if artifact.exists():
        return recover_adapter(task, args, device, directory, artifact)
    return moments.run_task(task, args, device, directory)[0]


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["stage"], row["dataset_display"], row["task"], row["method"], row["epsilon"])
        groups[key].append(row)
    result = []
    for key, members in groups.items():
        stage, dataset, task, method, epsilon = key
        output = {"stage": stage, "dataset": dataset, "task": task, "method": method,
                  "epsilon": epsilon, "runs": len(members), "privacy_unit": members[0]["privacy_unit"]}
        for metric in ["test_acc", "test_f1", "val_acc"]:
            values = [r[metric] for r in members]
            output[metric + "_mean"] = statistics.mean(values)
            output[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        result.append(output)
    return result


def save_progress(output, rows, total, status, current=None, delta=1e-5):
    base = moments.adapter.base
    base.write_json(output / "restricted_results.json", rows)
    if rows:
        base.write_csv(output / "restricted_results.csv", rows)
        base.write_json(output / "restricted_summary.json", summaries(rows))
        base.write_csv(output / "restricted_summary.csv", summaries(rows))
    stages = defaultdict(int)
    for row in rows:
        stages[row["stage"]] += 1
    base.write_json(output / "progress.json", {"status": status, "pid": os.getpid(),
                    "completed": len(rows), "total": total, "current": current,
                    "completed_by_stage": dict(stages),
                    "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S")})
    ledger = defaultdict(lambda: {"queries": 0, "rho": 0.})
    for row in rows:
        if row["epsilon"] is not None:
            item = ledger[row["task"]]
            item["queries"] += 1
            item["rho"] += moments.adapter.JointBudget(row["epsilon"], row["delta"], 1).rho
    for item in ledger.values():
        item["all_saved_dp_alternatives_window_epsilon_bound"] = item["rho"] + 2 * math.sqrt(-item["rho"] * math.log(delta))
    base.write_json(output / "restricted_completed_release_ledger.json", {
        "per_task": dict(ledger), "delta": delta,
        "scope": "conservative same-task window replacement composition across completed methods, budgets, ablations and seeds",
        "note": "Each saved alternative has its own epsilon. This ledger does not privatize nonprivate models, metrics or source/target role changes across tasks."})


def acquire_lock(output):
    path = output / "running.lock"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Existing run lock for PID {previous['pid']}. Verify that process has ended before archiving the stale lock.")
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "started_local_time": time.strftime("%Y-%m-%d %H:%M:%S")}, stream)
    return path


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    args.output_dir = Path(args.output_dir)
    tasks, jobs = build_plan(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        torch.ones(1, device=device)  # Fail before creating a run if CUDA cannot execute.
    hashes = fingerprints()
    base = moments.adapter.base
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.save_experiment_config(args.output_dir, {"args": vars(args), "method_version": VERSION,
                                                "source_sha256": hashes})
    lock = acquire_lock(args.output_dir)
    rows = []
    try:
        base.write_json(args.output_dir / "experiment_plan.json", {
            "version": VERSION, "tasks": [asdict(t) for t in tasks], "seeds": args.seeds,
            "public_trainings": len(tasks) * len(args.seeds), "total_results": len(jobs),
            "main_comparison_results": sum(j[2] == "main" for j in jobs),
            "epsilon_curve": args.epsilon_curve, "epochs": args.public_epochs,
            "primary_privacy_unit": "single_target_window",
            "client_dp_is_separate_diagnostic": True,
            "public_source_required": True, "fixed_parameters_no_test_selection": True,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})
        base.write_json(args.output_dir / "privacy_scope.json", {
            "source_is_public": True, "primary_method": "WindowDPRidge",
            "primary_unit": "one preprocessed target window replacement, including its label; fixed public allocation and label-status metadata",
            "client_unit": "complete target client replacement; ClientDPRidge only",
            "aggregator": "trusted central simulation; no cryptographic secure aggregation",
            "publication": "all metrics, nonprivate baselines, manifests and the full runs directory are restricted research outputs",
            "composition": "same-data alternative models and repeated seeds require composition; see completed release ledger",
            "cross_task_warning": "source/target roles reverse across benchmark tasks; publishing every public-source model invalidates a joint privacy claim for all raw domains",
            "deployment_interpretation": "each transfer task is a separate conditional-public-source scenario, not a single deployable all-domain DP system"})
        save_progress(args.output_dir, rows, len(jobs), "running", "initializing", args.delta)
        prepared = set()
        for task, seed, stage, method, epsilon, covariance in jobs:
            current = f"{task.name} seed={seed} {stage} {method}"
            save_progress(args.output_dir, rows, len(jobs), "running", current, args.delta)
            shared = SimpleNamespace(**{**vars(args), "seed": seed,
                "public_checkpoint_root": args.output_dir / "public_checkpoints" / f"seed{seed}"})
            key = (task.name, seed)
            if key not in prepared:
                public_reference(task, shared, device, hashes)
                prepared.add(key)
            job_args = SimpleNamespace(**{**vars(shared), "methods": [method], "epsilon": epsilon,
                "use_unlabeled_covariance": covariance})
            directory = args.output_dir / "jobs" / f"{task.name}__seed{seed}" / stage / method
            print(f"JOB {len(rows) + 1}/{len(jobs)} {current}", flush=True)
            started = time.time()
            row = run_job(task, job_args, device, directory, hashes)
            row = {**row, "stage": stage, "dataset_display": "SEU" if task.dataset == "DN" else task.dataset,
                   "formal_version": VERSION, "result_directory": str(directory)}
            if method == "PublicSourceOnly":
                row["trainable_parameters"] = 0
            rows.append(row)
            save_progress(args.output_dir, rows, len(jobs), "running", current, args.delta)
            print(f"DONE {len(rows)}/{len(jobs)} test={row['test_acc']:.2f}% seconds={time.time() - started:.1f}", flush=True)
        save_progress(args.output_dir, rows, len(jobs), "completed", delta=args.delta)
        print(f"COMPLETE {len(rows)}/{len(jobs)} results", flush=True)
    except BaseException:
        save_progress(args.output_dir, rows, len(jobs), "failed", traceback.format_exc(), args.delta)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return rows


if __name__ == "__main__":
    main()
