"""Dataset provenance, independent splits and experiment fingerprints."""

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


METHOD_VERSION = "fs-pda-unified-v2"


def source_fingerprint(code_root):
    files = ["Main.py", "main_comparison_experiment.py", "unified_experiment.py",
             "experiment_protocol.py", "prepare_pu_dataset.py"]
    return {name: hashlib.sha256((Path(code_root) / name).read_bytes()).hexdigest() for name in files}


def recording_groups(dataset, dataset_name):
    root = Path(dataset.root)
    manifest_path = root.parent / "manifest.json"
    if dataset_name == "PU":
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepare PU first: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mapping = {r["image"]: r for r in manifest["samples"]}
        groups = []
        for image, label in dataset.samples:
            relative = Path(image).relative_to(root.parent).as_posix()
            row = mapping[relative]
            if row["class"] != dataset.classes[label]:
                raise ValueError(f"Manifest label mismatch: {relative}")
            groups.append(row["record_id"])
        return groups, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if dataset_name == "Office31":
        return [Path(image).relative_to(root).as_posix() for image, _ in dataset.samples], "individual-images"
    groups = []
    for image, _ in dataset.samples:
        match = re.match(r"(.+)_[XY]_\d+\.[^.]+$", Path(image).name, re.I)
        if not match:
            raise ValueError(f"Cannot identify original recording from {image}")
        groups.append(match.group(1))
    return groups, "filename-original-recording"


def grouped_target_split(dataset, groups, labeled_ratio, val_ratio, test_ratio, seed):
    if not (0 < labeled_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Independent labeled/validation/unlabeled/test splits require ratios in (0,1)")
    by_class = defaultdict(lambda: defaultdict(list))
    for index, ((_, label), group) in enumerate(zip(dataset.samples, groups)):
        by_class[label][group].append(index)
    rng = random.Random(seed)
    splits = {k: [] for k in ["target_labeled", "target_val", "target_unlabeled", "target_test"]}
    for label, members in sorted(by_class.items()):
        keys = sorted(members)
        if len(keys) < 5:
            raise ValueError(
                f"Class {dataset.classes[label]} has only {len(keys)} original recording(s). "
                "A recording-disjoint four-way split is not supported. Supply more raw recordings; "
                "do not randomly split windows from the same recording."
            )
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * test_ratio))
        pool = keys[n_test:]
        n_budget = max(2, round(len(pool) * labeled_ratio))
        n_budget = min(n_budget, len(pool) - 1)
        n_val = min(n_budget - 1, max(1, round(n_budget * val_ratio)))
        selected = {"target_test": keys[:n_test], "target_val": pool[:n_val],
                    "target_labeled": pool[n_val:n_budget], "target_unlabeled": pool[n_budget:]}
        for split, group_keys in selected.items():
            for key in group_keys:
                splits[split].extend(members[key])
    seen = set()
    for name, indices in splits.items():
        if not indices or seen.intersection(indices):
            raise ValueError(f"Invalid or overlapping split: {name}")
        seen.update(indices)
    if seen != set(range(len(dataset))):
        raise ValueError("Split does not cover the dataset")
    group_sets = {k: {groups[i] for i in v} for k, v in splits.items()}
    for a in group_sets:
        for b in group_sets:
            if a != b and group_sets[a] & group_sets[b]:
                raise ValueError("Original recording crosses target splits")
    return splits


def save_split_manifest(destination, source, target, source_groups, target_groups, splits, provenance):
    entries = {}
    for split, indices in {"source": list(range(len(source))), **splits}.items():
        dataset, groups = (source, source_groups) if split == "source" else (target, target_groups)
        entries[split] = [{"image": Path(dataset.samples[i][0]).relative_to(dataset.root).as_posix(),
                           "label": dataset.classes[dataset.samples[i][1]], "record_id": groups[i]}
                          for i in indices]
    data = {"protocol": provenance["actual_protocol"], "provenance": provenance,
            "classes": target.classes, "splits": entries}
    serialized = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing split changed: {destination}")
    destination.write_text(serialized, encoding="utf-8")
    return hashlib.sha256(serialized.encode()).hexdigest()
