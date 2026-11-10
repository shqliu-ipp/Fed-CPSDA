"""Formal CrossOnlyDP(A) confirmation on PU, SEU and WT."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_cross_only_moments as method
import run_fed_private_formal as formal
import run_three_dataset_experiment as industrial
import run_pu_variant_a as helper


ROOT = Path(__file__).resolve().parent
CONFIG = {
    **formal.CONFIG,
    "output_dir": ROOT / "runs" / "a_three_dataset_confirmation_v3",
    "datasets": ["PU", "SEU_BEARING", "WT"],
    "task_names": None,
    "seeds": [42, 43, 44],
    "epsilons": [2., 4., 8., 16.],
    "noise_repeats": 5,
    "ridge": 0.02,
    "evaluate_test": True,
    "device": "cuda:0",
    "phase": "validation_then_test",
    "public_checkpoint_root": ROOT / "runs" / "fed_private_formal_v1" / "public_checkpoints",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(root, rows, total, stage, status, delta, current=None):
    base = method.previous.old.adapter.base
    base.write_json(root / f"restricted_{stage}_results.json", rows)
    if rows:
        base.write_csv(root / f"restricted_{stage}_results.csv", rows)
    base.write_json(root / "progress.json", {
        "status": status, "phase": stage, "completed": len(rows), "total": total,
        "pid": os.getpid(), "current": current,
        "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    by_task = {}
    budget_cls = method.previous.old.adapter.JointBudget
    for row in rows:
        if row["private"]:
            by_task.setdefault(row["task"], set()).add((row["seed"], row["budget_point"], row["noise_repeat"]))
    ledger = {}
    for task, trials in by_task.items():
        rho = sum(budget_cls(eps, delta, 1).rho for _, eps, _ in trials)
        ledger[task] = {"private_trials": len(trials), "joint_rho": rho,
                        "joint_epsilon_at_delta": rho + 2 * math.sqrt(-rho * math.log(delta))}
    base.write_json(root / f"restricted_{stage}_ledger.json", {"per_task": ledger, "delta": delta,
        "note": "A is one released postprocessing head per trial. Cross-dataset and repeated releases require composition; source/target roles are task-specific."})


def main(config=None):
    settings = {**CONFIG, **(config or {})}
    phase = settings.pop("phase")
    args0 = SimpleNamespace(**settings)
    tasks = industrial.make_tasks(args0)
    if args0.task_names:
        tasks = [t for t in tasks if t.name in set(args0.task_names)]
    expected_counts = {"PU": 6, "DN": 2, "SEU_BEARING": 2, "WT": 6}
    selected_counts = {dataset: [t.dataset for t in tasks].count(dataset)
                       for dataset in settings["datasets"]}
    if any(selected_counts.get(dataset, 0) != expected_counts[dataset]
           for dataset in settings["datasets"]):
        raise ValueError(f"Unexpected task counts: {selected_counts}")
    if phase != "validation_then_test" or settings["epsilons"] != [2., 4., 8., 16.] or settings["seeds"] != [42, 43, 44] or settings["noise_repeats"] != 5:
        raise ValueError("Use the fixed three-dataset A protocol")
    root = Path(settings["output_dir"])
    if root.exists() and any(p.name != "launcher.log" for p in root.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty output: {root}")
    root.mkdir(parents=True, exist_ok=True)
    base = method.previous.old.adapter.base
    hashes = formal.fingerprints()
    for name in ["fed_cross_only_moments.py", "run_a_three_dataset_confirmation.py",
                 "fed_covariance_borrowing.py", "fed_conditional_moments.py", "run_fed_private_formal.py",
                 "run_three_dataset_experiment.py"]:
        hashes[name] = sha(ROOT / name)
    prototype = {**settings, "epsilon": 2., "evaluate_test": False}
    base.save_experiment_config(root, {"args": prototype, "method_version": method.VERSION,
                                       "source_sha256": hashes})
    lock = helper.previous.common.formal.acquire_lock(root)
    total = len(tasks) * len(settings["seeds"]) * len(settings["epsilons"]) * (1 + settings["noise_repeats"])
    rows, stage = [], "not_started"
    validation_hashes = None
    try:
        torch.set_num_threads(settings["cpu_threads"])
        device = torch.device(settings["device"])
        base.write_json(root / "experiment_plan.json", {
            "method": "A: CrossOnlyDP = contrast-coded labeled cross moment + U-only covariance + trace-one PSD + ridge 0.02",
            "datasets": ["PU", "SEU", "WT"], "task_count": len(tasks),
            "task_names": [t.name for t in tasks], "seeds": settings["seeds"],
            "epsilons": settings["epsilons"], "noise_repeats": settings["noise_repeats"],
            "results_per_phase": total, "public_source_reused": True,
            "selection": "fixed parameters; no test-based selection; no source encoder retraining",
            "evaluation": "all validation models committed before test; test cannot fit or redraw noise",
            "privacy": "fixed-role target-window DP in trusted central simulation; not client-level DP",
            "data_scope": "PU six transfers, SEU/DN two transfers, WT six speed transfers; no Office31/PT600"})
        for stage in ["validation", "test"]:
            rows = []
            for seed in settings["seeds"]:
                shared = {**prototype, "seed": seed, "evaluate_test": stage == "test",
                          "public_checkpoint_root": Path(settings["public_checkpoint_root"]) / f"seed{seed}"}
                for task in tasks:
                    checkpoint = shared["public_checkpoint_root"] / task.name / "public_source_model.pt"
                    if not checkpoint.exists():
                        formal.public_reference(task, SimpleNamespace(**shared), device, hashes)
                    feature_dir = root / "features" / f"{task.name}__seed{seed}"
                    feature_dir.mkdir(parents=True, exist_ok=True)
                    args = SimpleNamespace(**shared)
                    data, _, mean, matrix, meta, provenance = method.previous.old.prepare_task(task, args, device, feature_dir)
                    x = method.previous.old.design(data["labeled"][0], mean, matrix)
                    u = method.previous.old.design(data["unlabeled"][0], mean, matrix)
                    public = method.previous.old.public_release_provenance(provenance)
                    split = "val" if stage == "validation" else "test"
                    phi = method.previous.old.design(data[split][0], mean, matrix)
                    for epsilon in settings["epsilons"]:
                        args.epsilon = epsilon
                        current = f"{stage} eps={epsilon:g} {task.name} seed={seed}"
                        save(root, rows, total, stage, "running", args.delta, current)
                        for private, repeat in [(False, -1)] + [(True, i) for i in range(settings["noise_repeats"])]:
                            directory = root / f"eps{epsilon:g}" / f"{task.name}__seed{seed}" / (f"dp_repeat{repeat}" if private else "nonprivate")
                            directory.mkdir(parents=True, exist_ok=True)
                            artifact = directory / "adapter_bundle.pt"
                            result_path = directory / f"restricted_{stage}_results.json"
                            if artifact.exists():
                                bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                                if bundle["version"] != method.VERSION or bundle["private"] != private or bundle["provenance"] != public:
                                    raise ValueError("Saved model mismatch")
                            else:
                                if stage == "test":
                                    raise FileNotFoundError("Test cannot fit missing model")
                                weight = method.fit(x, data["labeled"][2], u, meta["num_classes"], args, private)
                                bundle = {"version": method.VERSION, "weight": weight, "private": private,
                                    "whitening_mean": mean, "whitening_matrix": matrix, "classes": meta["num_classes"],
                                    "provenance": public,
                                    "privacy": method.privacy_report(args, len(x), len(u), meta["num_classes"]) if private else None}
                                temporary = directory / "adapter_bundle.partial.pt"
                                torch.save(bundle, temporary)
                                os.replace(temporary, artifact)
                            if result_path.exists():
                                trial_row = json.loads(result_path.read_text(encoding="utf-8"))
                            else:
                                metrics = method.previous.old.metrics(phi.double() @ bundle["weight"], data[split][2], bundle["classes"])
                                trial_row = {"dataset": "SEU" if task.dataset == "DN" else task.dataset,
                                    "task": task.name, "seed": seed, "noise_repeat": repeat,
                                    "method": "CrossOnlyDP" if private else "CrossOnlyNP", "private": private,
                                    "epsilon": epsilon if private else None, "budget_point": epsilon,
                                    "delta": args.delta if private else None, "version": method.VERSION,
                                    "privacy_unit": "single_target_window" if private else None,
                                    "source_is_public": True, "evaluation_is_restricted": True,
                                    "result_directory": str(directory), split + "_acc": metrics["accuracy"],
                                    split + "_f1": metrics["macro_f1"], split + "_confusion_matrix": metrics["confusion_matrix"]}
                                base.write_json(result_path, trial_row)
                            rows.append(trial_row)
                        print(f"COMPLETE {current}: {len(rows)}/{total}", flush=True)
            if len(rows) != total:
                raise ValueError("Incomplete phase")
            save(root, rows, total, stage, "completed", settings["delta"])
            current_hashes = {str(p): sha(p) for p in root.rglob("adapter_bundle.pt")}
            if stage == "validation":
                validation_hashes = current_hashes
            elif current_hashes != validation_hashes:
                raise ValueError("Model changed during test")
            else:
                base.write_json(root / "model_integrity.json", {"all_unchanged": True,
                    "bundles_checked": len(current_hashes), "model_sha256": current_hashes})
        return rows
    except BaseException:
        save(root, rows, total, stage, "failed", settings["delta"], traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
