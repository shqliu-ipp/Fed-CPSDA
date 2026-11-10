"""Editable direct entry for PU + raw SEU gear + raw WT; shared training engine."""

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

import unified_experiment as exp
import gear_experiment_protocol as gear_protocol
import prepare_gear_datasets as preparation


CODE_ROOT = exp.CODE_ROOT
CONFIG = {
    **exp.RUN_CONFIG,
    "datasets": ["PU", "SEU_BEARING", "WT"],
    "gear_data_root": exp.PROJECT_ROOT / "DATA" / "industrial_raw_v1",
    "wt_raw_root": preparation.CONFIG["wt_raw_root"],
    "dn_raw_root": preparation.CONFIG["dn_raw_root"],
    "dn_bearing_raw_root": preparation.CONFIG["dn_bearing_raw_root"],
    "output_dir": exp.CODE_ROOT / "runs" / "three_industrial_raw_v1",
    "prepare_raw_data": True,
    "include_wt_installation": False,
    "include_wt_sensor": False,
    "task_names": None,
    "gpu_ids": None,
    "max_parallel": 1,
    "batch_size": 128,
}


def make_tasks(args):
    tasks = exp.make_tasks(args.data_root, "pu") if "PU" in args.datasets else []
    root = Path(args.gear_data_root)

    def pair(dataset, source, target, kind):
        for a, b in [(source, target), (target, source)]:
            tasks.append(exp.TransferTask(dataset, f"{a}_to_{b}", root / dataset / a, root / dataset / b, kind))

    if "DN" in args.datasets:
        pair("DN", "SEU_20_0_C2", "SEU_30_2_C2", "cross_speed_load_condition")
    if "SEU_BEARING" in args.datasets:
        pair("SEU_BEARING", "SEU_BEARING_20_0_C2", "SEU_BEARING_30_2_C2", "cross_speed_load_condition")
    if "WT" in args.datasets:
        for a, b in [(20, 35), (20, 50), (35, 50)]:
            pair("WT", f"WT_{a}_I1_X", f"WT_{b}_I1_X", "cross_speed")
        if args.include_wt_installation:
            for speed in [35, 50]:
                pair("WT", f"WT_{speed}_I1_X", f"WT_{speed}_I2_X", "cross_installation")
        if args.include_wt_sensor:
            pair("WT", "WT_50_I1_X", "WT_50_I1_Y", "paired_cross_sensor")
    if args.task_names:
        missing = set(args.task_names) - {task.name for task in tasks}
        if missing:
            raise ValueError(f"Unknown/disabled tasks: {sorted(missing)}")
        tasks = [task for task in tasks if task.name in args.task_names]
    if not tasks:
        raise ValueError("No experiment tasks selected")
    if args.quick:
        tasks = [next(task for task in tasks if task.dataset == dataset)
                 for dataset in args.datasets if any(task.dataset == dataset for task in tasks)]
    return tasks


def fingerprints():
    result = exp.source_fingerprint(exp.CODE_ROOT)
    for name in ["prepare_gear_datasets.py", "gear_experiment_protocol.py", "run_three_dataset_experiment.py"]:
        result[name] = hashlib.sha256((exp.CODE_ROOT / name).read_bytes()).hexdigest()
    return result


def prepare_data(args):
    selected = [dataset for dataset in args.datasets if dataset in {"WT", "DN", "SEU_BEARING"}]
    if selected:
        preparation.prepare({"datasets": selected, "output_root": args.gear_data_root,
            "wt_raw_root": args.wt_raw_root, "dn_raw_root": args.dn_raw_root,
            "dn_bearing_raw_root": args.dn_bearing_raw_root})
    if "PU" in args.datasets and not (Path(args.data_root) / "PU" / "manifest.json").is_file():
        raise FileNotFoundError("Prepare PU using prepare_pu_dataset.py before running the three-dataset matrix")


def run_job(task, method, seed, args, hashes, directory, device):
    directory.mkdir(parents=True, exist_ok=True)
    exp.seed_everything(seed)
    builder = exp.build_loaders if task.dataset == "PU" else gear_protocol.build_loaders
    loaders, meta = builder(task, args, seed, directory)
    meta["source_sha256"] = hashes
    signature = hashlib.sha256(json.dumps(exp.serial({"args": vars(args), "task": asdict(task),
        "method": method, "seed": seed, "source_sha256": hashes, "split": meta["split_sha256"]}), sort_keys=True).encode()).hexdigest()
    result_file = directory / "result.json"
    if result_file.exists():
        previous = json.loads(result_file.read_text(encoding="utf-8"))
        if previous["run_signature"] != signature:
            raise FileExistsError(f"Result/config mismatch: {directory}; choose a new output_dir")
        if args.resume:
            return previous
    print(f"TRAIN {task.name} | {method} | seed={seed} | {device}", flush=True)
    started = time.time()
    metrics = exp.run_method(method, loaders, meta, args, device, directory)
    row = {"dataset": task.dataset, "task": task.name, "task_type": task.task_type,
        "method": method, "seed": seed, "method_version": exp.METHOD_VERSION,
        "run_signature": signature, "source": str(task.source), "target": str(task.target),
        **meta, **metrics, "elapsed_sec": time.time() - started}
    exp.write_json(result_file, row)
    print(f"  test={row['target_acc']:.2f}% macro-F1={row['target_f1']:.4f}", flush=True)
    return row


def parallel(args, tasks):
    jobs = [(task, method, seed) for task in tasks for seed in args.seeds for method in args.methods]
    pending, running, failed, rows = list(jobs), [], [], []
    try:
        while pending or running:
            for gpu in args.gpu_ids:
                if not pending or len(running) >= args.max_parallel:
                    break
                if any(job["gpu"] == gpu for job in running):
                    continue
                task, method, seed = pending.pop(0)
                directory = args.output_dir / "jobs" / f"{task.name}__{method}__seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                config = {**vars(args), "gpu_ids": None, "child_run": True, "prepare_raw_data": False,
                    "output_dir": directory, "task_names": [task.name], "methods": [method], "seeds": [seed]}
                config_file = directory / "child_config.json"
                exp.write_json(config_file, config)
                log = (directory / "run.log").open("w", encoding="utf-8")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONIOENCODING": "utf-8"}
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config-json", str(config_file)],
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                running.append({"process": process, "gpu": gpu, "directory": directory, "log": log})
                print(f"LAUNCH GPU {gpu}: {task.name}, {method}, seed={seed}", flush=True)
            time.sleep(min(max(args.poll_interval, 0.1), 30))
            for job in running[:]:
                code = job["process"].poll()
                if code is None:
                    continue
                job["log"].close()
                running.remove(job)
                result = job["directory"] / "result.json"
                if code or not result.exists():
                    failed.append(str(job["directory"]))
                else:
                    rows.append(json.loads(result.read_text(encoding="utf-8")))
                exp.save_results(args.output_dir, rows)
                exp.write_json(args.output_dir / "progress.json", {
                    "finished": len(rows) + len(failed), "total": len(jobs), "failed": failed, "running": len(running)})
    finally:
        for job in running:
            job["process"].terminate()
            job["process"].wait()
            job["log"].close()
    if failed:
        raise RuntimeError(f"Failed jobs; inspect their run.log files: {failed}")
    return rows


def main(config=None):
    args = SimpleNamespace(**{**CONFIG, **(config or {})})
    args.data_root, args.gear_data_root, args.output_dir = map(Path, [args.data_root, args.gear_data_root, args.output_dir])
    if not args.datasets or set(args.datasets) - {"PU", "WT", "DN", "SEU_BEARING"} or len(set(args.datasets)) != len(args.datasets):
        raise ValueError("datasets must be a nonempty, unique subset of PU, WT, DN, SEU_BEARING")
    if (not args.methods or len(set(args.methods)) != len(args.methods)
            or set(args.methods) - (set(exp.RUN_CONFIG["methods"]) | set(exp.ABLATIONS))):
        raise ValueError("Methods must be known, nonempty and unique")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if args.quick:
        args.epochs, args.seeds, args.warmup_epochs = 3, args.seeds[:1], 0
        args.pseudo_start_epoch = args.align_start_epoch = args.rl_start_epoch = args.ours_cdan_start_epoch = 1
        args.privacy_start_epoch, args.privacy_ramp_epochs, args.selection_start_epoch = 1, 2, 3
    complete = max(args.warmup_epochs + 1, args.pseudo_start_epoch, args.align_start_epoch,
        args.rl_start_epoch, args.ours_cdan_start_epoch, args.privacy_start_epoch + args.privacy_ramp_epochs - 1)
    if not complete <= args.selection_start_epoch <= args.epochs:
        raise ValueError("Selection must follow module activation and fit within epochs")
    if args.gpu_ids:
        if len({str(g) for g in args.gpu_ids}) != len(args.gpu_ids) or any(not str(g).isdigit() for g in args.gpu_ids):
            raise ValueError("gpu_ids must contain unique nonnegative GPU indices")
        if args.max_parallel < 1:
            raise ValueError("max_parallel must be positive")
    tasks = make_tasks(args)
    if args.prepare_raw_data:
        prepare_data(args)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    hashes = fingerprints()
    exp.save_experiment_config(args.output_dir, {"args": vars(args), "method_version": exp.METHOD_VERSION,
        "source_sha256": hashes, "tasks": [asdict(task) for task in tasks],
        "runtime": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "total_jobs": len(tasks) * len(args.seeds) * len(args.methods)})
    if args.gpu_ids and not args.child_run:
        return parallel(args, tasks)
    rows = []
    for task in tasks:
        for seed in args.seeds:
            for method in args.methods:
                directory = args.output_dir if args.child_run else args.output_dir / "jobs" / f"{task.name}__{method}__seed{seed}"
                rows.append(run_job(task, method, seed, args, hashes, directory, device))
                exp.save_results(args.output_dir, rows)
                exp.write_json(args.output_dir / "progress.json", {"finished": len(rows), "last_task": task.name})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json")
    cli = parser.parse_args()
    main(json.loads(Path(cli.config_json).read_text(encoding="utf-8")) if cli.config_json else CONFIG)
