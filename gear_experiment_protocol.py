"""Purged temporal blocks for single-recording gearbox domains."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from Main import TransformSubset, build_transforms


PROTOCOL = "within_recording_purged_time_blocks_v1"
ROLES = ["target_labeled", "target_val", "target_unlabeled", "target_test"]


def target_block_split(rows, labeled_ratio, val_ratio, test_ratio, seed):
    if not all(0 < ratio < 1 for ratio in [labeled_ratio, val_ratio, test_ratio]):
        raise ValueError("Split ratios must lie in (0,1)")
    records = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        records[row["record_id"]][row["block_id"]].append(index)
    splits = {role: [] for role in ROLES}
    for record, blocks in sorted(records.items()):
        ordered = sorted(blocks)
        n_test = max(1, round(len(ordered) * test_ratio))
        pool, test = ordered[:-n_test], ordered[-n_test:]
        if len(pool) < 3:
            raise ValueError("Insufficient blocks for four partitions")
        n_budget = min(len(pool) - 1, max(2, round(len(pool) * labeled_ratio)))
        n_val = min(n_budget - 1, max(1, round(n_budget * val_ratio)))
        record_seed = int(hashlib.sha256(f"{seed}:{record}".encode()).hexdigest()[:16], 16)
        random.Random(record_seed).shuffle(pool)
        selected = {"target_test": test, "target_val": pool[:n_val],
                    "target_labeled": pool[n_val:n_budget], "target_unlabeled": pool[n_budget:]}
        for role, members in selected.items():
            for block in members:
                splits[role].extend(blocks[block])
    flat = [i for indices in splits.values() for i in indices]
    if len(flat) != len(rows) or set(flat) != set(range(len(rows))):
        raise ValueError("Incomplete or overlapping target split")
    return splits


def assert_temporal_isolation(partitions, minimum_gap_seconds):
    intervals = defaultdict(lambda: defaultdict(list))
    for role, rows in partitions.items():
        for row in rows:
            intervals[row["record_id"]][row["block_id"]].append((role, row))
    for blocks in intervals.values():
        previous_stop = None
        for _, members in sorted(blocks.items()):
            roles = {role for role, _ in members}
            if len(roles) != 1:
                raise ValueError("A target temporal block crosses partitions")
            first = min(row["start"] / row["sample_rate"] for _, row in members)
            last = max(row["stop"] / row["sample_rate"] for _, row in members)
            if previous_stop is not None and first - previous_stop < minimum_gap_seconds - 1e-8:
                raise ValueError("Temporal guard gap violated")
            previous_stop = last


def build_loaders(task, args, seed, artifact_dir):
    manifest_file = task.target.parent / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    root = task.target.parent
    if task.source.parent != root:
        raise ValueError("Gearbox source and target must use the same manifest")
    mapping = {row["image"]: row for row in manifest["samples"]}
    source, target = ImageFolder(task.source), ImageFolder(task.target)
    if source.classes != target.classes:
        raise ValueError("Source/target class mappings differ")

    def lookup(dataset):
        rows = [mapping[Path(file).relative_to(root).as_posix()] for file, _ in dataset.samples]
        if any(row["class"] != dataset.classes[label] for row, (_, label) in zip(rows, dataset.samples)):
            raise ValueError("ImageFolder labels differ from raw manifest")
        return rows

    source_rows, target_rows = lookup(source), lookup(target)
    splits = target_block_split(target_rows, args.target_labeled_ratio,
                                args.target_val_ratio, args.target_test_ratio, seed)
    partitions = {role: [target_rows[i] for i in indices] for role, indices in splits.items()}
    gap = 2 * manifest["config"]["block_edge_guard_seconds"]
    assert_temporal_isolation(partitions, gap)
    # Paired X/Y channels share physical time: remove held-out intervals from source too.
    held_out = {(row["record_id"], row["block_id"])
                for role in ["target_val", "target_test"] for row in partitions[role]}
    source_indices = [i for i, row in enumerate(source_rows)
                      if (row["record_id"], row["block_id"]) not in held_out]
    if not source_indices:
        raise ValueError("No source training examples after paired-sensor purge")
    source_keys = {(source_rows[i]["record_id"], source_rows[i]["block_id"]) for i in source_indices}
    if source_keys & held_out:
        raise ValueError("Paired sensor leakage")
    profile = "fault_light" if args.augmentation_profile == "auto" else args.augmentation_profile
    normalization = ("imagenet" if args.pretrained_backbone else "default") if args.normalize_profile == "auto" else args.normalize_profile
    weak, strong, test = build_transforms(args.img_size, profile, normalization)
    sets = {"source": TransformSubset(source, source_indices, weak)}
    for role, indices in splits.items():
        sets[role] = TransformSubset(target, indices, test if role in ["target_val", "target_test"] else weak)
    sets["target_unlabeled_two_view"] = TransformSubset(target, splits["target_unlabeled"], weak, True, strong)
    kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    loaders = {role: DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)),
        shuffle=role not in ["target_val", "target_test"], drop_last=False,
        generator=torch.Generator().manual_seed(seed), **kwargs) for role, dataset in sets.items()}
    payload = {"protocol": PROTOCOL, "dataset_manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
        "classes": target.classes, "minimum_block_gap_seconds": gap,
        "test_selection": "chronologically last blocks, fixed across seeds",
        "paired_source_excluded_samples": len(source) - len(source_indices),
        "independent_acquisition_claim": False,
        "splits": {"source": [source_rows[i] for i in source_indices], **partitions}}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    destination = Path(artifact_dir) / "split_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing split changed: {destination}")
    destination.write_text(serialized, encoding="utf-8")
    meta = {"classes": target.classes, "num_classes": len(target.classes),
        "split_protocol": PROTOCOL, "split_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "augmentation_profile": profile, "normalize_profile": normalization,
        "backbone_type": args.backbone_type, "pretrained_backbone": args.pretrained_backbone,
        "paired_source_excluded_samples": payload["paired_source_excluded_samples"],
        "minimum_block_gap_seconds": gap,
        **{role + "_samples": len(dataset) for role, dataset in sets.items()}}
    return loaders, meta
