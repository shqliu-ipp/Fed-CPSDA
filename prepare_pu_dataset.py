"""Prepare auditable, recording-grouped PU vibration spectrograms."""

import hashlib
import io
import os
import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.signal import resample_poly, spectrogram
from matplotlib import colormaps


CODE_ROOT = Path(__file__).resolve().parent
CONFIG = {
    "raw_root": Path(os.environ.get("FEDCPSDA_PU_RAW_ROOT", CODE_ROOT.parent / "data_raw" / "PU")),
    "output_dir": CODE_ROOT.parent / "DATA" / "PU",
    "window_size": 16384,
    "windows_per_record": 12,
    "image_size": 64,
    "nperseg": 256,
    "noverlap": 192,
    "db_range": 80.0,
}
DOMAINS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
# KI04/KI14 contain both IR and OR damage according to the supplied fact sheets.
BEARINGS = {
    "healthy": ["K001", "K002", "K003"],
    "inner": ["KI16", "KI18", "KI21"],
    "outer": ["KA04", "KA15", "KA16"],
}
VERSION = "pu-stft-record-v2"


def read_vibration(mat_bytes, name):
    contents = loadmat(io.BytesIO(mat_bytes), simplify_cells=True)
    if name not in contents:
        raise ValueError(f"Missing MATLAB variable {name}")
    record = contents[name]
    channels = record["Y"]
    channel = next(c for c in channels if c["Name"] == "vibration_1")
    signal = np.asarray(channel["Data"], dtype=np.float64).reshape(-1)
    axes = record["X"]
    axis = axes[int(channel["XIndex"]) - 1]
    time = np.asarray(axis["Data"], dtype=np.float64).reshape(-1)
    if len(time) < 2 or len(signal) != len(time):
        raise ValueError("Vibration/time axis lengths do not match")
    # Exported timestamps are quantized; a median interval gives a biased rate.
    sample_rate = float((len(time) - 1) / (time[-1] - time[0]))
    if (not np.isfinite(signal).all() or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0) or not 60000 <= sample_rate <= 80000):
        raise ValueError(f"Invalid vibration data or sampling rate {sample_rate}")
    original_samples = len(signal)
    # Uniformize recorded timestamps, then anti-alias to a common physical rate.
    uniform_time = np.linspace(time[0], time[-1], len(time))
    uniform_signal = np.interp(uniform_time, time, signal)
    ratio = Fraction(64000 / sample_rate).limit_denominator(10000)
    signal = resample_poly(uniform_signal, ratio.numerator, ratio.denominator)
    return signal, 64000.0, {"observed_average_rate": sample_rate,
                            "original_samples": original_samples,
                            "resample_up": ratio.numerator, "resample_down": ratio.denominator}


def window_starts(length, window_size, count):
    available = length // window_size
    if count < 1 or count > available:
        raise ValueError(f"Requested {count} nonoverlapping windows, available={available}")
    return (np.linspace(0, available - 1, count).astype(int) * window_size).tolist()


def spectrogram_rgb(signal, sample_rate, config):
    _, _, power = spectrogram(
        signal, fs=sample_rate, window="hann", nperseg=config["nperseg"],
        noverlap=config["noverlap"], detrend="constant", scaling="density", mode="psd",
    )
    db = 10.0 * np.log10(np.maximum(power, 1e-20))
    db -= db.max()
    scaled = np.clip((db + config["db_range"]) / config["db_range"], 0.0, 1.0)
    rgb = (colormaps["turbo"](scaled[::-1])[..., :3] * 255).round().astype(np.uint8)
    return Image.fromarray(rgb).resize((config["image_size"], config["image_size"]), Image.Resampling.BILINEAR)


def prepare(config=None):
    config = {**CONFIG, **(config or {})}
    raw_root, output = Path(config["raw_root"]), Path(config["output_dir"])
    serial_config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    signature = hashlib.sha256(json.dumps(
        {"version": VERSION, "config": serial_config, "bearings": BEARINGS}, sort_keys=True,
    ).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["preparation_signature"] != signature:
            raise FileExistsError("Preparation changed: use a new output directory")
        missing = [r["image"] for r in previous["samples"] if not (output / r["image"]).is_file()]
        changed = [r["raw_file"] for r in previous["records"]
                   if not (raw_root / r["raw_file"]).is_file()
                   or hashlib.sha256((raw_root / r["raw_file"]).read_bytes()).hexdigest() != r["sha256"]]
        if not missing and not changed:
            print(f"Verified existing PU manifest: {len(previous['samples'])} images", flush=True)
            return output
        raise RuntimeError(f"Prepared dataset incomplete/changed: images={len(missing)}, records={len(changed)}")

    pattern = re.compile(r"(N\d+_M\d+_F\d+)_(K[AIB]?\d+)_(\d+)\.mat", re.I)
    class_by_bearing = {b: cls for cls, bearings in BEARINGS.items() for b in bearings}
    found = {}
    for file in sorted(raw_root.rglob("*.mat")):
        match = pattern.fullmatch(file.name)
        if not match:
            continue
        domain, bearing, repeat = match.groups()
        if domain in DOMAINS and bearing in class_by_bearing:
            key = (domain, bearing, int(repeat))
            if key in found:
                raise ValueError(f"Duplicate raw recording: {key}")
            found[key] = file
    expected = {(d, b, r) for d in DOMAINS for b in class_by_bearing for r in range(1, 21)}
    if set(found) != expected:
        raise ValueError(f"PU recording inventory differs: missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}")
    output.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for i, ((domain, bearing, repeat), file) in enumerate(sorted(found.items()), 1):
        data = file.read_bytes()
        signal, sample_rate, acquisition = read_vibration(data, file.stem)
        record_id = file.stem
        starts = window_starts(len(signal), config["window_size"], config["windows_per_record"])
        raw_relative = file.relative_to(raw_root).as_posix()
        records.append({"record_id": record_id, "raw_file": raw_relative,
                        "sha256": hashlib.sha256(data).hexdigest(), "bearing": bearing,
                        "domain": domain, "class": class_by_bearing[bearing],
                        "sample_rate": sample_rate, "signal_samples": len(signal),
                        "acquisition": acquisition})
        for j, start in enumerate(starts):
            relative = Path(domain) / class_by_bearing[bearing] / f"{record_id}__w{j:02d}.png"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            spectrogram_rgb(signal[start:start + config["window_size"]], sample_rate, config).save(destination)
            rows.append({"image": relative.as_posix(), "record_id": record_id,
                         "bearing": bearing, "domain": domain, "class": class_by_bearing[bearing],
                         "start": start, "stop": start + config["window_size"]})
        if i == 1 or i % 40 == 0 or i == len(found):
            print(f"PU preparation {i}/{len(found)} records, {len(rows)} images", flush=True)
    manifest = {"version": VERSION, "preparation_signature": signature, "config": serial_config,
                "bearings": BEARINGS, "domains": DOMAINS, "records": records, "samples": rows,
                "protocol": "recording-stratified-target; same bearing identities may occur across splits"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PU manifest saved: {manifest_path}", flush=True)
    return output


if __name__ == "__main__":
    prepare()
