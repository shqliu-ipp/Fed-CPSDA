"""Prepare raw WT/SEU signals without using the historical derived images."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

from prepare_pu_dataset import spectrogram_rgb


CODE_ROOT = Path(__file__).resolve().parent
VERSION = "gear-native-stft-purged-block-v1"
CONFIG = {
    "wt_raw_root": Path(os.environ.get("FEDCPSDA_WT_RAW_ROOT", CODE_ROOT.parent / "data_raw" / "WT")),
    "dn_raw_root": Path(os.environ.get("FEDCPSDA_DN_RAW_ROOT", CODE_ROOT.parent / "data_raw" / "SEU_GEAR")),
    "dn_bearing_raw_root": Path(os.environ.get("FEDCPSDA_DN_BEARING_RAW_ROOT", CODE_ROOT.parent / "data_raw" / "SEU_BEARING")),
    "output_root": CODE_ROOT.parent / "DATA" / "industrial_raw_v1",
    "datasets": ["WT", "SEU_BEARING"],
    "wt_speeds": [20, 35, 50],
    "wt_installations": [1, 2],
    "wt_channels": {"X": 0, "Y": 1},
    "dn_sample_rate": 5120.0,
    "dn_channel": 1,
    "window_seconds": 0.256,
    "blocks_per_record": 100,
    "windows_per_block": 4,
    "block_edge_guard_seconds": 0.25,
    "image_size": 64, "nperseg": 256, "noverlap": 192, "db_range": 80.0,
    "verify_raw_hashes": False,
}
WT_CLASSES = {"broken": "B", "healthy": "N", "missing_tooth": "M",
              "root_crack": "R", "wear": "W"}
DN_CLASSES = {"Chipped": "chipped", "Health": "healthy", "Miss": "missing_tooth",
              "Root": "root_crack", "Surface": "surface"}
DN_BEARING_CLASSES = {"ball": "ball", "comb": "comb", "health": "healthy",
                      "inner": "inner", "outer": "outer"}


def dump(file, data):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(file)


def file_hash(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_layout(file):
    with Path(file).open(encoding="utf-8-sig", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line_number > 100:
                break
            for separator in ("\t", ","):
                fields = line.strip().rstrip(separator).split(separator)
                if len(fields) != 8:
                    continue
                try:
                    values = [float(value) for value in fields]
                except ValueError:
                    continue
                if np.isfinite(values).all():
                    return line_number, separator
    raise ValueError(f"No eight-channel numeric data found: {file}")


def block_windows(length, sample_rate, config):
    window = round(config["window_seconds"] * sample_rate)
    guard = int(np.ceil(config["block_edge_guard_seconds"] * sample_rate))
    count = config["windows_per_block"]
    if min(window, count, config["blocks_per_record"]) < 1 or guard < 0:
        raise ValueError("Invalid block/window configuration")
    edges = np.linspace(0, length, config["blocks_per_record"] + 1).astype(np.int64)
    result = []
    for block, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        first, last = int(left + guard), int(right - guard - window)
        if last - first < (count - 1) * window:
            raise ValueError(f"Recording too short for guarded block {block}")
        starts = np.linspace(first, last, count).astype(np.int64)
        if count > 1 and np.diff(starts).min() < window:
            raise ValueError("Overlapping windows")
        result.extend((block, j, int(start), int(start + window)) for j, start in enumerate(starts))
    return result


def inventory(dataset, config):
    records = []
    if dataset == "WT":
        root = Path(config["wt_raw_root"])
        for cls, prefix in WT_CLASSES.items():
            for installation in config["wt_installations"]:
                for speed in config["wt_speeds"]:
                    file = root / cls / str(installation) / f"{prefix}{installation}_{speed}.MAT"
                    records.append((file, {"class": cls, "installation": installation,
                        "speed_hz": speed, "record_id": f"WT_{file.stem}",
                        "channels": config["wt_channels"]}))
    elif dataset == "DN":
        root = Path(config["dn_raw_root"])
        for prefix, cls in DN_CLASSES.items():
            for condition in ["20_0", "30_2"]:
                file = root / "gearset" / f"{prefix}_{condition}.csv"
                records.append((file, {"class": cls, "condition": condition,
                    "record_id": f"SEU_{file.stem}", "channels": {"C2": config["dn_channel"]}}))
    elif dataset == "SEU_BEARING":
        root = Path(config["dn_bearing_raw_root"])
        for prefix, cls in DN_BEARING_CLASSES.items():
            for condition in ["20_0", "30_2"]:
                file = root / "bearingset" / f"{prefix}_{condition}.csv"
                records.append((file, {"class": cls, "condition": condition,
                    "record_id": f"SEU_BEARING_{file.stem}",
                    "channels": {"C2": config["dn_channel"]}}))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    missing = [str(file) for file, _ in records if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw files: {missing}")
    return root, records


def read_record(file, dataset, config):
    if dataset == "WT":
        contents = loadmat(file, variable_names=["Data", "SampleRate", "ChannelCount"], simplify_cells=True)
        data, rate = np.asarray(contents["Data"]), float(contents["SampleRate"])
        if data.ndim != 2 or data.shape[1] < 2 or rate != 48000:
            raise ValueError(f"Unexpected WT shape/rate: {data.shape}, {rate}")
        return data, rate, {"rate_source": "MAT SampleRate", "shape": list(data.shape)}
    skip, separator = csv_layout(file)
    signal = np.loadtxt(file, skiprows=skip, delimiter=separator, usecols=(config["dn_channel"],))
    return signal[:, None], float(config["dn_sample_rate"]), {
        "rate_source": "supplied preprocessing scripts; not inferred from Frequency Limit header",
        "numeric_start_line_zero_based": skip, "delimiter": separator,
        "shape": [len(signal), 1], "raw_column_zero_based": config["dn_channel"]}


def prepare_one(dataset, config):
    output = Path(config["output_root"]) / dataset
    portable = {k: v for k, v in config.items() if k not in {
        "wt_raw_root", "dn_raw_root", "dn_bearing_raw_root", "output_root",
        "datasets", "verify_raw_hashes"}}
    signature_data = {"version": VERSION, "dataset": dataset, "config": portable,
                      "code": file_hash(__file__), "stft_code": file_hash(CODE_ROOT / "prepare_pu_dataset.py")}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    guard_file = output / "preparation_config.json"
    if guard_file.exists():
        old = json.loads(guard_file.read_text(encoding="utf-8"))
        if old["signature"] != signature:
            raise FileExistsError(f"Preparation changed; choose a new output_root: {output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse an untracked output directory: {output}")
    dump(guard_file, {"signature": signature, **signature_data})
    manifest_file = output / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["preparation_signature"] != signature:
            raise ValueError("Manifest/config mismatch")
        if any(not (output / row["image"]).is_file() for row in manifest["samples"]):
            raise RuntimeError(f"Prepared images missing: {output}")
        if not config["verify_raw_hashes"]:
            print(f"Verified prepared {dataset}: {len(manifest['samples'])} images", flush=True)
            return output
    raw_root, inventory_rows = inventory(dataset, config)
    rows, records = [], []
    for index, (file, info) in enumerate(inventory_rows, 1):
        cache = output / "record_cache" / f"{info['record_id']}.json"
        stat = file.stat()
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
        if cached:
            record = cached["record"]
            if record["raw_bytes"] != stat.st_size or record["raw_mtime_ns"] != stat.st_mtime_ns:
                raise RuntimeError(f"Raw file changed after preparation: {file}")
            if config["verify_raw_hashes"] and file_hash(file) != record["sha256"]:
                raise RuntimeError(f"Raw hash mismatch: {file}")
            if any(not (output / row["image"]).is_file() for row in cached["samples"]):
                cached = None
        if not cached:
            data, rate, acquisition = read_record(file, dataset, config)
            windows = block_windows(len(data), rate, config)
            samples = []
            for sensor, raw_column in info["channels"].items():
                signal = data[:, raw_column if dataset == "WT" else 0]
                if not np.isfinite(signal).all():
                    raise ValueError(f"Nonfinite signal: {file}, {sensor}")
                if dataset == "WT":
                    domain = f"WT_{info['speed_hz']}_I{info['installation']}_{sensor}"
                elif dataset == "DN":
                    domain = f"SEU_{info['condition']}_C2"
                else:
                    domain = f"SEU_BEARING_{info['condition']}_C2"
                for block, j, start, stop in windows:
                    relative = Path(domain) / info["class"] / f"{info['record_id']}__b{block:03d}_w{j}.png"
                    destination = output / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    spectrogram_rgb(signal[start:stop], rate, config).save(destination)
                    samples.append({"image": relative.as_posix(), "domain": domain, "class": info["class"],
                        "record_id": info["record_id"], "block_id": block, "sensor": sensor,
                        "raw_column_zero_based": raw_column, "sample_rate": rate, "start": start, "stop": stop})
            record = {**info, "raw_file": file.relative_to(raw_root).as_posix(), "sha256": file_hash(file),
                "raw_bytes": stat.st_size, "raw_mtime_ns": stat.st_mtime_ns, "signal_samples": len(data),
                "sample_rate": rate, "acquisition": acquisition}
            cached = {"record": record, "samples": samples}
            dump(cache, cached)
            del data
        records.append(cached["record"])
        rows.extend(cached["samples"])
        print(f"{dataset} raw preparation {index}/{len(inventory_rows)}; {len(rows)} images", flush=True)
    manifest = {"version": VERSION, "preparation_signature": signature, "config": portable,
        "raw_root_at_preparation": str(raw_root), "records": records, "samples": rows,
        "domains": sorted({row["domain"] for row in rows}),
        "protocol_limit": "time-blocks within recordings; blocks/channels are not independent acquisitions"}
    dump(manifest_file, manifest)
    return output


def prepare(config=None):
    config = {**CONFIG, **(config or {})}
    return {dataset: prepare_one(dataset, config) for dataset in config["datasets"]}


if __name__ == "__main__":
    prepare()
