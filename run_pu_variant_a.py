"""Direct PU variant-A sweep. Edit CONFIG; no SEU/WT or excluded direction."""

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

import fed_cross_only_moments as mechanism
import run_pu_covariance_borrowing as previous
from run_pu_privacy_budget_sweep import RETAINED_TASKS


ROOT = Path(__file__).resolve().parent
CONFIG = {
    "output_dir": ROOT / "runs" / "pu_variant_a_v1",
    "datasets": ["PU"], "task_names": RETAINED_TASKS,
    "epsilons": [2., 4., 8., 16.], "delta": 1e-5,
    "seeds": [42, 43, 44], "noise_repeats": 5,
    "ridge": 0.02, "batch_size": 128, "device": "cuda:0",
    "phase": "validation_then_test",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(root, rows, total, phase, status, delta, current=None):
    base = mechanism.previous.old.adapter.base
    base.write_json(root / f"restricted_{phase}_results.json", rows)
    base.write_json(root / "progress.json", {
        "status": status, "phase": phase, "completed": len(rows), "total": total,
        "pid": os.getpid(), "current": current,
        "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    if rows:
        base.write_csv(root / f"restricted_{phase}_results.csv", rows)
    by_task = {}
    for row in rows:
        if row["private"]:
            by_task.setdefault(row["task"], {})[(row["epsilon"], row["seed"], row["noise_repeat"])] = row["epsilon"]
    ledger = {}
    for task, releases in by_task.items():
        rho = sum(mechanism.previous.old.adapter.JointBudget(eps, delta, 1).rho for eps in releases.values())
        ledger[task] = {"private_models": len(releases), "joint_rho": rho,
                        "joint_epsilon_at_delta": rho + 2 * math.sqrt(-rho * math.log(delta))}
    base.write_json(root / f"restricted_{phase}_ledger.json", {
        "per_task": ledger, "delta": delta,
        "note": "Alternative models compose if jointly released. Previous experiments are additional costs. Metrics and NP models are not DP releases.",
    })


def evaluate(bundle, data, args, task, seed, repeat, directory, phase):
    split = "val" if phase == "validation" else "test"
    x, _, y = data[split]
    phi = mechanism.previous.old.design(x, bundle["whitening_mean"], bundle["whitening_matrix"])
    metrics = mechanism.previous.old.metrics(phi.double() @ bundle["weight"], y, bundle["classes"])
    private = bundle["private"]
    return {
        "dataset": "PU", "task": task.name, "seed": seed, "noise_repeat": repeat,
        "method": "CrossOnlyDP" if private else "CrossOnlyNP", "private": private,
        "epsilon": args.epsilon if private else None, "budget_point": args.epsilon,
        "delta": args.delta if private else None, "version": mechanism.VERSION,
        "privacy_unit": "single_target_window" if private else None,
        "source_is_public": True, "evaluation_is_restricted": True,
        "result_directory": str(directory),
        split + "_acc": metrics["accuracy"], split + "_f1": metrics["macro_f1"],
        split + "_confusion_matrix": metrics["confusion_matrix"],
    }


def main(config=None):
    settings = {**CONFIG, **(config or {})}
    phase = settings.pop("phase")
    root = Path(settings["output_dir"])
    if (phase not in {"validation", "test", "validation_then_test"}
            or settings["datasets"] != ["PU"] or not settings["task_names"]
            or len(set(settings["task_names"])) != len(settings["task_names"])
            or not set(settings["task_names"]) <= set(RETAINED_TASKS)
            or not settings["epsilons"] or len(set(settings["epsilons"])) != len(settings["epsilons"])
            or any(eps not in [2., 4., 8., 16.] for eps in settings["epsilons"])):
        raise ValueError("Only the specified five PU directions and fixed epsilon points are allowed")
    prototype = {**previous.CONFIG, **settings, "epsilon": settings["epsilons"][0]}
    tasks, _ = previous.plan(SimpleNamespace(**prototype))
    if {t.name for t in tasks} != set(settings["task_names"]):
        raise ValueError("Task selection mismatch")
    total = len(tasks) * len(settings["seeds"]) * (1 + settings["noise_repeats"]) * len(settings["epsilons"])
    root.mkdir(parents=True, exist_ok=True)
    hashes = previous.common.formal.fingerprints()
    for filename in ["fed_cross_only_moments.py", "run_pu_variant_a.py", "fed_covariance_borrowing.py",
                     "fed_conditional_moments.py", "run_pu_privacy_budget_sweep.py",
                     "run_pu_covariance_borrowing.py", "run_pu_conditional_moments.py"]:
        hashes[filename] = sha(ROOT / filename)
    base = mechanism.previous.old.adapter.base
    base.save_experiment_config(root, {"args": prototype, "method_version": mechanism.VERSION, "source_sha256": hashes})
    lock = previous.common.formal.acquire_lock(root)
    torch.set_num_threads(prototype["cpu_threads"])
    device = torch.device(settings["device"])
    stages = ["validation", "test"] if phase == "validation_then_test" else [phase]
    try:
        base.write_json(root / "experiment_plan.json", {
            "settings": settings, "results_each_phase": total,
            "frozen_mechanism": "L cross only; U covariance only; trace-one PSD; fixed ridge 0.02",
            "public_source_prior": False, "adaptive_regularization": False,
            "evaluation_policy": "all validation models before test; no test fitting or noise redraw",
            "baseline": "Existing independent-noise budget sweep, compared by task/seed averages",
            "scope": "exploratory five-direction PU subset; public-source window DP only",
        })
        validation_hashes = None
        for stage in stages:
            rows = []
            for eps in settings["epsilons"]:
                for seed in settings["seeds"]:
                    args = SimpleNamespace(**{**prototype, "epsilon": eps, "seed": seed,
                        "evaluate_test": stage == "test",
                        "public_checkpoint_root": Path(prototype["public_checkpoint_root"]) / f"seed{seed}"})
                    for task in tasks:
                        context = root / f"eps{eps:g}" / "contexts" / f"{task.name}__seed{seed}"
                        context.mkdir(parents=True, exist_ok=True)
                        current = f"{stage} eps={eps:g} {task.name} seed={seed}"
                        save(root, rows, total, stage, "running", args.delta, current)
                        pending = []
                        for private, repeat in [(False, -1)] + [(True, i) for i in range(settings["noise_repeats"])]:
                            directory = context / (f"dp_repeat{repeat}" if private else "nonprivate")
                            result_path = directory / f"restricted_{stage}_result.json"
                            if result_path.exists():
                                if not (directory / "adapter_bundle.pt").exists():
                                    raise FileNotFoundError("Result is missing its saved model")
                                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                            else:
                                pending.append((private, repeat, directory, result_path))
                        if not pending:
                            continue
                        data, _, mean, matrix, meta, provenance = mechanism.previous.old.prepare_task(task, args, device, context)
                        x = mechanism.previous.old.design(data["labeled"][0], mean, matrix)
                        u = mechanism.previous.old.design(data["unlabeled"][0], mean, matrix)
                        public_provenance = mechanism.previous.old.public_release_provenance(provenance)
                        for private, repeat, directory, result_path in pending:
                            directory.mkdir(parents=True, exist_ok=True)
                            artifact = directory / "adapter_bundle.pt"
                            if artifact.exists():
                                bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                                if (bundle["version"] != mechanism.VERSION or bundle["private"] != private
                                        or bundle["provenance"] != public_provenance):
                                    raise ValueError("Saved model mismatch")
                            else:
                                if stage == "test":
                                    raise FileNotFoundError("Complete validation first: testing cannot fit missing models")
                                bundle = {
                                    "version": mechanism.VERSION, "private": private,
                                    "weight": mechanism.fit(x, data["labeled"][2], u, meta["num_classes"], args, private),
                                    "whitening_mean": mean, "whitening_matrix": matrix,
                                    "classes": meta["num_classes"], "provenance": public_provenance,
                                    "privacy": mechanism.privacy_report(args, len(x), len(u), meta["num_classes"]) if private else None,
                                }
                                temporary = directory / "adapter_bundle.partial.pt"
                                torch.save(bundle, temporary)
                                os.replace(temporary, artifact)
                            row = evaluate(bundle, data, args, task, seed, repeat, directory, stage)
                            base.write_json(result_path, row)
                            rows.append(row)
                        print(f"COMPLETE {current}: {len(rows)}/{total}", flush=True)
            if len(rows) != total:
                raise ValueError("Incomplete phase")
            save(root, rows, total, stage, "completed", settings["delta"])
            if stage == "validation":
                validation_hashes = {str(p): sha(p) for p in root.rglob("adapter_bundle.pt")}
            elif validation_hashes is not None:
                current_hashes = {str(p): sha(p) for p in root.rglob("adapter_bundle.pt")}
                if current_hashes != validation_hashes:
                    raise ValueError("Models changed during test evaluation")
                base.write_json(root / "model_integrity.json", {
                    "bundles_checked": len(current_hashes), "all_unchanged": True,
                    "model_sha256": current_hashes,
                })
        return rows
    except BaseException:
        save(root, rows, total, stage, "failed", settings["delta"], traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
