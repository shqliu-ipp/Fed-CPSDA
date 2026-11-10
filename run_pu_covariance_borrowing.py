"""PU-only covariance mechanism validation, followed by saved-model test inference."""

import os
os.environ.setdefault("USERNAME", "Administrator")

import hashlib
import json
import math
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_covariance_borrowing as mechanism
import run_pu_conditional_moments as common


CONFIG = {
    **common.CONFIG,
    "output_dir": common.formal.ROOT / "runs" / "pu_covariance_borrowing_v1",
    "methods": mechanism.METHODS,
    "borrowing_cap": 0.95,
    "evaluate_test": False,
}


def plan(args):
    if (args.datasets != ["PU"] or args.quick or args.methods != mechanism.METHODS
            or not args.public_source_acknowledged or args.feature_kind != "backbone_pca"
            or not math.isfinite(args.borrowing_cap) or not 0 <= args.borrowing_cap <= 1
            or not math.isfinite(args.ridge) or args.ridge <= 0
            or not args.seeds or len(set(args.seeds)) != len(args.seeds)
            or isinstance(args.noise_repeats, bool) or not isinstance(args.noise_repeats, int)
            or args.noise_repeats < 1):
        raise ValueError("Invalid PU-only covariance-borrowing plan")
    mechanism.old.adapter.JointBudget(args.epsilon, args.delta, 1)
    tasks = common.formal.industrial.make_tasks(args)
    if any(task.dataset != "PU" for task in tasks):
        raise ValueError("Only PU is permitted")
    return tasks, len(tasks) * len(args.seeds) * len(mechanism.METHODS) * (1 + args.noise_repeats)


def evaluate(bundle, data, args, task, repeat, directory):
    rows = []
    for method in mechanism.METHODS:
        private = bundle["private"]
        row = {"dataset": "PU", "task": task.name, "seed": args.seed, "noise_repeat": repeat,
               "method": method + ("DP" if private else "NP"), "private": private,
               "epsilon": args.epsilon if private else None, "delta": args.delta if private else None,
               "privacy_unit": "single_target_window" if private else None, "version": mechanism.VERSION,
               "source_is_public": True, "evaluation_is_restricted": True, "result_directory": str(directory)}
        for split in ["val", "test"] if args.evaluate_test else ["val"]:
            x, _, y = data[split]
            phi = mechanism.old.design(x, bundle["whitening_mean"], bundle["whitening_matrix"])
            metric = mechanism.old.metrics(phi.double() @ bundle["heads"][method], y, bundle["classes"])
            row[split + "_acc"], row[split + "_f1"] = metric["accuracy"], metric["macro_f1"]
            row[split + "_confusion_matrix"] = metric["confusion_matrix"]
        rows.append(row)
    return rows


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    args.output_dir = Path(args.output_dir)
    args.public_checkpoint_root = Path(args.public_checkpoint_root)
    tasks, total = plan(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.ones(1, device=device)
    hashes = common.formal.fingerprints()
    for name in ["fed_conditional_moments.py", "run_pu_conditional_moments.py",
                 "fed_covariance_borrowing.py", "run_pu_covariance_borrowing.py"]:
        hashes[name] = hashlib.sha256((common.formal.ROOT / name).read_bytes()).hexdigest()
    base = mechanism.old.adapter.base
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Evaluation phase is not a fitting parameter. Only existing bundles may be
    # tested; entering that phase is explicitly forbidden from creating models.
    fitting_args = {k: v for k, v in vars(args).items() if k != "evaluate_test"}
    base.save_experiment_config(args.output_dir, {"args": fitting_args, "method_version": mechanism.VERSION,
                                                "source_sha256": hashes})
    lock = common.formal.acquire_lock(args.output_dir)
    phase = "test_saved_models" if args.evaluate_test else "validation"
    result_name = "restricted_test_results.json" if args.evaluate_test else "restricted_validation_results.json"
    rows = []
    try:
        base.write_json(args.output_dir / "experiment_plan.json", {
            "datasets": ["PU"], "tasks": [t.name for t in tasks], "seeds": args.seeds,
            "noise_repeats": args.noise_repeats, "methods": mechanism.METHODS, "total_results": total,
            "epsilon_each_private_model": args.epsilon, "public_source_required": True,
            "evaluation_policy": "validation first; test phase only evaluates the exact saved models and never fits or draws noise",
            "adaptive_weight": "known Gaussian noise energy and released empirical covariance disparity; no validation/test optimization",
            "privacy_scope": "fixed-role window replacement; parallel disjoint labeled/unlabeled mechanisms",
            "scope_excludes": "complete-client privacy, raw preprocessing and private evaluation metrics"})
        for seed in args.seeds:
            for task in tasks:
                current = f"{phase}: {task.name} seed={seed}"
                common.save(args.output_dir, rows, total, "running", current, args.delta)
                context = args.output_dir / "contexts" / f"{task.name}__seed{seed}"
                shared = SimpleNamespace(**{**vars(args), "seed": seed,
                    "public_checkpoint_root": args.public_checkpoint_root / f"seed{seed}"})
                pending = []
                for private, repeat in [(False, -1)] + [(True, rep) for rep in range(args.noise_repeats)]:
                    directory = context / (f"dp_repeat{repeat}" if private else "nonprivate")
                    result_file = directory / result_name
                    if result_file.exists():
                        previous = json.loads(result_file.read_text(encoding="utf-8"))
                        if [r["method"] for r in previous] != [m + ("DP" if private else "NP") for m in mechanism.METHODS]:
                            raise ValueError("Incomplete trial results")
                        if not (directory / "adapter_bundle.pt").exists():
                            raise FileNotFoundError("Missing adapter bundle")
                        rows.extend(previous)
                    else:
                        pending.append((private, repeat, directory))
                if not pending:
                    continue
                data, _, mean, matrix, meta, provenance = mechanism.old.prepare_task(task, shared, device, context)
                x = mechanism.old.design(data["labeled"][0], mean, matrix)
                u = mechanism.old.design(data["unlabeled"][0], mean, matrix)
                for private, repeat, directory in pending:
                    directory.mkdir(parents=True, exist_ok=True)
                    artifact = directory / "adapter_bundle.pt"
                    if artifact.exists():
                        bundle = torch.load(artifact, map_location="cpu", weights_only=False)
                        if (bundle["version"] != mechanism.VERSION or bundle["private"] != private
                                or bundle["provenance"] != mechanism.old.public_release_provenance(provenance)):
                            raise ValueError("Incompatible saved model bundle")
                    else:
                        if args.evaluate_test:
                            raise FileNotFoundError("Test inference cannot fit missing models; complete validation first")
                        heads, diagnostics = mechanism.fit(x, data["labeled"][2], u, meta["num_classes"], shared, private)
                        bundle = {"version": mechanism.VERSION, "heads": heads, "private": private,
                                  "whitening_mean": mean, "whitening_matrix": matrix,
                                  "classes": meta["num_classes"], "feature_kind": args.feature_kind,
                                  "diagnostics": diagnostics,
                                  "privacy": mechanism.privacy_report(shared, len(x), len(u), meta["num_classes"]) if private else None,
                                  "provenance": mechanism.old.public_release_provenance(provenance)}
                        temporary = directory / "adapter_bundle.partial.pt"
                        torch.save(bundle, temporary)
                        os.replace(temporary, artifact)
                    trial_rows = evaluate(bundle, data, shared, task, repeat, directory)
                    base.write_json(directory / result_name, trial_rows)
                    rows.extend(trial_rows)
                    scores = ", ".join(f"{r['method']}={r.get('test_acc', r['val_acc']):.2f}" for r in trial_rows)
                    print(f"RESULT {current} repeat={repeat}: {scores}", flush=True)
                    common.save(args.output_dir, rows, total, "running", current, args.delta)
        if len(rows) != total:
            raise ValueError("Result count mismatch")
        base.write_json(args.output_dir / result_name, rows)
        common.save(args.output_dir, rows, total, "completed", phase, args.delta)
        print(f"COMPLETE PU-only {phase} {len(rows)}/{total}", flush=True)
    except BaseException:
        common.save(args.output_dir, rows, total, "failed", traceback.format_exc(), args.delta)
        raise
    finally:
        lock.unlink(missing_ok=True)
    return rows


if __name__ == "__main__":
    main()
