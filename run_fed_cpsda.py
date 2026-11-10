"""Direct entry point for Fed-CPSDA; edit CONFIG and run without CLI arguments."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

import fed_cpsda_experiment as fed
import gear_experiment_protocol as gear
import run_three_dataset_experiment as industrial


CODE_ROOT = Path(__file__).resolve().parent
CONFIG = {
    **industrial.CONFIG,
    "methods": fed.METHODS,
    "datasets": ["PU", "DN", "WT"],
    "task_names": [
        "PU_N15_M07_F10_to_N09_M07_F10",
        "SEU_20_0_C2_to_SEU_30_2_C2",
        "WT_20_I1_X_to_WT_35_I1_X",
    ],
    "seeds": [42],
    "output_dir": CODE_ROOT / "runs" / "fed_cpsda_pilot_v1",
    "prepare_raw_data": False,
    "rounds": 20,
    "local_epochs": 1,
    "client_shards": 4,
    "prototype_weight": 0.15,
    "pseudo_weight": 0.5,
    "pseudo_threshold": 0.85,
    "pseudo_floor": 0.65,
    "threshold_decay": 0.15,
    "epsilon": 8.0,
    "delta": 1e-5,
    "client_clip": 0.25,
    "batch_size": 128,
    "gpu_ids": None,
    "max_parallel": 1,
    "child_run": False,
}


def make_tasks(args):
    tasks = []
    pu = industrial.exp.make_tasks(args.data_root, "pu")
    all_tasks = {task.name: task for task in pu}
    root = Path(args.gear_data_root)
    pairs = [
        ("DN", "SEU_20_0_C2", "SEU_30_2_C2", "cross_speed_load_condition"),
        ("WT", "WT_20_I1_X", "WT_35_I1_X", "cross_speed"),
    ]
    for dataset, source, target, kind in pairs:
        all_tasks[f"{source}_to_{target}"] = industrial.exp.TransferTask(
            dataset, f"{source}_to_{target}", root / dataset / source, root / dataset / target, kind)
    missing = set(args.task_names) - set(all_tasks)
    if missing:
        raise ValueError(f"Unknown Fed-CPSDA task(s): {sorted(missing)}")
    return [all_tasks[name] for name in args.task_names]


def fingerprints():
    names = ["fed_cpsda_experiment.py", "run_fed_cpsda.py", "unified_experiment.py",
             "experiment_protocol.py", "gear_experiment_protocol.py"]
    return {name: hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() for name in names}


def run_one(task, method, seed, args, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    industrial.exp.seed_everything(seed)
    loaders, meta = fed.task_loaders(task, args, seed, directory)
    meta["source_sha256"] = fingerprints()
    signature = hashlib.sha256(json.dumps(industrial.exp.serial({
        "task": asdict(task), "method": method, "seed": seed, "args": vars(args),
        "split": meta["split_sha256"], "code": meta["source_sha256"]}), sort_keys=True).encode()).hexdigest()
    result_file = directory / "result.json"
    if result_file.exists():
        previous = json.loads(result_file.read_text(encoding="utf-8"))
        if previous["run_signature"] != signature:
            raise FileExistsError(f"Existing result/config mismatch: {directory}")
        if args.resume:
            return previous
    print(f"TRAIN {task.name} | {method} | seed={seed} | {device}", flush=True)
    started = time.time()
    metrics = fed.run_method(task, method, loaders, meta, args, device, directory, seed)
    row = {"dataset": task.dataset, "task": task.name, "task_type": task.task_type,
           "method": method, "seed": seed, "method_version": fed.VERSION,
           "run_signature": signature, "source": str(task.source), "target": str(task.target),
           **meta, **metrics, "elapsed_sec": time.time() - started}
    industrial.exp.write_json(result_file, row)
    print(f"  test={row['target_acc']:.2f}% macro-F1={row['target_f1']:.4f}", flush=True)
    return row


def summarize(rows):
    output = []
    keys = sorted({(row["dataset"], row["task"], row["method"]) for row in rows})
    for dataset, task, method in keys:
        selected = [row for row in rows if (row["dataset"], row["task"], row["method"]) == (dataset, task, method)]
        result = {"dataset": dataset, "task": task, "method": method, "runs": len(selected)}
        for metric in ["target_acc", "target_f1", "mean_pseudo_rate"]:
            values = [float(row[metric]) for row in selected]
            result[metric + "_mean"] = sum(values) / len(values)
            result[metric + "_std"] = (sum((x - result[metric + "_mean"]) ** 2 for x in values) / (len(values) - 1)) ** .5 if len(values) > 1 else 0.0
        output.append(result)
    return output


def save_results(output, rows):
    industrial.exp.write_json(output / "main_comparison_raw.json", rows)
    industrial.exp.write_csv(output / "main_comparison_raw.csv", rows)
    industrial.exp.write_json(output / "main_comparison_summary.json", summarize(rows))
    industrial.exp.write_csv(output / "main_comparison_summary.csv", summarize(rows))


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    if set(args.methods) & {"FedAvgDP", "FedProtoDP", "FedCPSDA"}:
        raise RuntimeError(
            "Legacy v1 DP path is disabled due to invalid privacy accounting. "
            "Run run_fed_private_moments.py; client DP and window DP are separate experiments. "
            "See Fed-CPSDA_DP修复结果与适用边界_2026-11-06.md in the project root.")
    args.output_dir = Path(args.output_dir)
    args.data_root = Path(args.data_root)
    args.gear_data_root = Path(args.gear_data_root)
    args.wt_raw_root = Path(args.wt_raw_root)
    args.dn_raw_root = Path(args.dn_raw_root)
    tasks = make_tasks(args)
    if args.prepare_raw_data:
        gear.prepare({"datasets": [d for d in args.datasets if d in {"DN", "WT"}],
                      "output_root": args.gear_data_root, "wt_raw_root": args.wt_raw_root,
                      "dn_raw_root": args.dn_raw_root})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    industrial.exp.save_experiment_config(args.output_dir, {
        "args": vars(args), "method_version": fed.VERSION, "source_sha256": fingerprints(),
        "budget": fed.ClientBudget(args.epsilon, args.delta, args.rounds, args.client_clip).report(args.client_shards * 2),
        "tasks": [asdict(task) for task in tasks], "total_jobs": len(tasks) * len(args.methods) * len(args.seeds)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    rows = []
    for task in tasks:
        for seed in args.seeds:
            for method in args.methods:
                directory = args.output_dir / "jobs" / f"{task.name}__{method}__seed{seed}"
                rows.append(run_one(task, method, seed, args, directory, device))
                save_results(args.output_dir, rows)
                industrial.exp.write_json(args.output_dir / "progress.json", {"finished": len(rows), "total": len(tasks) * len(args.methods) * len(args.seeds), "last": rows[-1]})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json")
    cli = parser.parse_args()
    main(json.loads(Path(cli.config_json).read_text(encoding="utf-8")) if cli.config_json else CONFIG)
