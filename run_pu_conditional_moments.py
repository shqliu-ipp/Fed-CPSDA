"""Direct PU-only mechanism experiment; edit CONFIG, then run this file."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
import statistics
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_conditional_moments as mechanism
import run_fed_private_formal as formal
import run_fed_private_moments as pilot


CONFIG = {
    **pilot.CONFIG,
    "output_dir": formal.ROOT / "runs" / "pu_conditional_moments_v1",
    "public_checkpoint_root": formal.CONFIG["output_dir"] / "public_checkpoints",
    "datasets": ["PU"],
    "methods": mechanism.METHODS,
    "task_names": None,
    "quick": False,
    "seeds": [42, 43, 44],
    "noise_repeats": 5,
    "epsilon": 8.,
    "delta": 1e-5,
    "ridge": 0.02,
    "moment_dim": 16,
    "feature_kind": "backbone_pca",
    "teacher_temperature": 0.25,
    "pseudo_confidence": 0.75,
    "unlabeled_weight": 1.0,
    "support_noise_multiplier": 2.0,
    "prototype_cosine_floor": 0.5,
    "max_relative_curvature": 0.5,
    "max_teacher_change": 0.3,
    "batch_size": 128,
    "num_workers": 0,
    "cpu_threads": 4,
    "device": "cuda:0",
    "evaluate_test": True,
}


def plan(args):
    numeric = [args.ridge, args.teacher_temperature, args.pseudo_confidence, args.unlabeled_weight,
               args.support_noise_multiplier, args.prototype_cosine_floor,
               args.max_relative_curvature, args.max_teacher_change]
    if (not all(math.isfinite(value) for value in numeric)
            or args.datasets != ["PU"] or args.methods != mechanism.METHODS
            or args.quick or not args.public_source_acknowledged
            or args.feature_kind != "backbone_pca" or args.moment_dim < 1
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in args.seeds)
            or isinstance(args.noise_repeats, bool) or not isinstance(args.noise_repeats, int)
            or args.noise_repeats < 1 or not 0 < args.pseudo_confidence < 1
            or args.teacher_temperature <= 0 or args.ridge <= 0 or args.unlabeled_weight <= 0
            or args.support_noise_multiplier <= 0 or not -1 < args.prototype_cosine_floor < 1
            or args.max_relative_curvature <= 0 or args.max_teacher_change <= 0):
        raise ValueError("Invalid PU-only mechanism configuration; SEU and WT are explicitly excluded")
    mechanism.old.adapter.JointBudget(args.epsilon, args.delta, 1)
    tasks = formal.industrial.make_tasks(args)
    if any(task.dataset != "PU" for task in tasks):
        raise ValueError("Only PU tasks are allowed")
    return tasks, len(tasks) * len(args.seeds) * len(mechanism.METHODS) * (1 + args.noise_repeats)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["method"])].append(row)
    summary = []
    for (task, method), members in groups.items():
        seeds = defaultdict(list)
        for row in members:
            seeds[row["seed"]].append(row)
        item = {"task": task, "method": method, "runs": len(members), "training_seeds": len(seeds)}
        for metric in ["val_acc", "test_acc", "test_f1"]:
            if metric not in members[0]:
                continue
            values = [statistics.mean(r[metric] for r in group) for group in seeds.values()]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_training_seed_std"] = statistics.stdev(values) if len(values) > 1 else None
            noise_sd = [statistics.stdev(r[metric] for r in group) for group in seeds.values() if len(group) > 1]
            item[metric + "_mean_within_seed_noise_std"] = statistics.mean(noise_sd) if noise_sd else None
        summary.append(item)
    return summary


def save(root, rows, total, status, current=None, delta=1e-5):
    base = mechanism.old.adapter.base
    base.write_json(root / "restricted_results.json", rows)
    if rows:
        base.write_csv(root / "restricted_results.csv", rows)
        base.write_json(root / "restricted_summary.json", aggregate(rows))
        base.write_csv(root / "restricted_summary.csv", aggregate(rows))
    base.write_json(root / "progress.json", {"status": status, "pid": os.getpid(), "completed": len(rows),
                    "total": total, "current": current, "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S")})
    # Count shared releases by bundle, not once for every postprocessed ablation.
    groups = defaultdict(dict)
    for row in rows:
        if row["private"]:
            groups[row["task"]][(row["seed"], row["noise_repeat"])] = row["epsilon"]
    ledger = {}
    for task, trials in groups.items():
        rho = sum(2 * mechanism.old.adapter.JointBudget(eps, delta, 1).rho for eps in trials.values())
        ledger[task] = {"completed_dp_bundles": len(trials), "rho_all_five_heads": rho,
                        "epsilon_all_saved_dp_bundles": rho + 2 * math.sqrt(-rho * math.log(delta))}
    base.write_json(root / "restricted_release_ledger.json", {
        "per_task": ledger, "delta": delta,
        "note": "Each conditional family shares an adaptive parallel teacher/unlabeled protocol. The legacy comparison is an extra overlapping mechanism. Nonprivate models and metrics are excluded from DP claims."})


def evaluate_bundle(bundle, data, args, task, repeat, directory):
    result = []
    private = bundle["private"]
    for method in mechanism.METHODS:
        row = {"dataset": "PU", "task": task.name, "seed": args.seed, "noise_repeat": repeat,
               "method": method + ("DP" if private else "NP"), "private": private,
               "epsilon": args.epsilon if private else None, "delta": args.delta if private else None,
               "privacy_unit": "single_target_window" if private else None,
               "version": mechanism.VERSION, "source_is_public": True,
               "evaluation_is_restricted": True, "result_directory": str(directory)}
        for split in ["val", "test"] if args.evaluate_test else ["val"]:
            x, _, y = data[split]
            design = mechanism.old.design(x, bundle["whitening_mean"], bundle["whitening_matrix"])
            metrics = mechanism.old.metrics(design.double() @ bundle["heads"][method], y, bundle["classes"])
            row[split + "_acc"] = metrics["accuracy"]
            row[split + "_f1"] = metrics["macro_f1"]
            row[split + "_confusion_matrix"] = metrics["confusion_matrix"]
        result.append(row)
    return result


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    args.output_dir = Path(args.output_dir)
    args.public_checkpoint_root = Path(args.public_checkpoint_root)
    tasks, total = plan(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.ones(1, device=device)
    hashes = formal.fingerprints()
    for name in ["fed_conditional_moments.py", "run_pu_conditional_moments.py"]:
        hashes[name] = hashlib.sha256((formal.ROOT / name).read_bytes()).hexdigest()
    base = mechanism.old.adapter.base
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.save_experiment_config(args.output_dir, {"args": vars(args), "method_version": mechanism.VERSION,
                                                "source_sha256": hashes})
    lock = formal.acquire_lock(args.output_dir)
    rows = []
    try:
        base.write_json(args.output_dir / "experiment_plan.json", {
            "datasets": ["PU"], "tasks": [t.name for t in tasks], "seeds": args.seeds,
            "noise_repeats": args.noise_repeats, "total_results": total, "new_public_training": False,
            "methods": mechanism.METHODS, "epsilon_each_private_model": args.epsilon,
            "fixed_parameters": True, "test_based_selection": False,
            "privacy_scope": "conditional public source, fixed-role window replacement, trusted central aggregation simulation",
            "parallel_composition_condition": "labeled and unlabeled slots are disjoint and their roles are fixed public metadata; the unlabeled query only depends on the DP teacher and each window's own views",
            "restricted_outputs": "all metrics and nonprivate models; not a DP publication package"})
        for seed in args.seeds:
            for task in tasks:
                current = f"{task.name} seed={seed}"
                save(args.output_dir, rows, total, "running", current, args.delta)
                context = args.output_dir / "contexts" / f"{task.name}__seed{seed}"
                shared = SimpleNamespace(**{**vars(args), "seed": seed,
                    "public_checkpoint_root": args.public_checkpoint_root / f"seed{seed}"})
                trials = [(False, -1)] + [(True, rep) for rep in range(args.noise_repeats)]
                pending = []
                for private, repeat in trials:
                    directory = context / (f"dp_repeat{repeat}" if private else "nonprivate")
                    result_file = directory / "restricted_results.json"
                    if result_file.exists():
                        previous = json.loads(result_file.read_text(encoding="utf-8"))
                        expected = [m + ("DP" if private else "NP") for m in mechanism.METHODS]
                        if [r["method"] for r in previous] != expected:
                            raise ValueError("Incomplete trial results; preserve the bundle and recover metrics")
                        if not (directory / "adapter_bundle.pt").is_file():
                            raise FileNotFoundError("Completed trial is missing its adapter bundle")
                        rows.extend(previous)
                    else:
                        pending.append((private, repeat, directory))
                if not pending:
                    continue
                print(f"ENCODE {current}; PU only", flush=True)
                data, _, mean, matrix, meta, provenance = mechanism.old.prepare_task(task, shared, device, context)
                weak, strong = mechanism.encode_views(task, shared, device, context)
                x = mechanism.old.design(data["labeled"][0], mean, matrix)
                u = mechanism.old.design(data["unlabeled"][0], mean, matrix)
                weak, strong = [mechanism.old.design(view, mean, matrix) for view in [weak, strong]]
                labels = data["labeled"][2]
                for private, repeat, directory in pending:
                    directory.mkdir(parents=True, exist_ok=True)
                    artifact = directory / "adapter_bundle.pt"
                    if artifact.exists():
                        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                        if (bundle["version"] != mechanism.VERSION or bundle["private"] != private
                                or bundle["provenance"] != mechanism.old.public_release_provenance(provenance)):
                            raise ValueError("Incompatible interrupted bundle")
                    else:
                        heads, diagnostics = mechanism.fit(x, labels, u, weak, strong, meta["num_classes"], shared, private)
                        bundle = {"version": mechanism.VERSION, "heads": heads, "private": private,
                                  "whitening_mean": mean, "whitening_matrix": matrix,
                                  "feature_kind": args.feature_kind, "classes": meta["num_classes"],
                                  "diagnostics": diagnostics,
                                  "privacy": mechanism.privacy_report(shared, len(x), len(u)) if private else None,
                                  "provenance": mechanism.old.public_release_provenance(provenance)}
                        temporary = directory / "adapter_bundle.partial.pt"
                        torch.save(bundle, temporary)
                        os.replace(temporary, artifact)
                    trial_rows = evaluate_bundle(bundle, data, shared, task, repeat, directory)
                    base.write_json(directory / "restricted_results.json", trial_rows)
                    rows.extend(trial_rows)
                    scores = ", ".join(f"{r['method']}={r.get('test_acc', r['val_acc']):.2f}" for r in trial_rows)
                    print(f"RESULT {current} repeat={repeat}: {scores}", flush=True)
                    save(args.output_dir, rows, total, "running", current, args.delta)
        if len(rows) != total:
            raise ValueError("PU matrix result count mismatch")
        save(args.output_dir, rows, total, "completed", delta=args.delta)
        print(f"COMPLETE PU-only {len(rows)}/{total}", flush=True)
    except BaseException:
        save(args.output_dir, rows, total, "failed", traceback.format_exc(), args.delta)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return rows


if __name__ == "__main__":
    main()
