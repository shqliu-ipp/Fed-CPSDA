"""Canonical FS-PDA v2 training, shared by every public experiment entry point."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import datasets

from Main import (ReinforcedPrivacyDA, TransformSubset, build_backbone, build_transforms,
                  class_prototype_loss, seed_everything, source_actor_critic_loss,
                  stratified_target_split, split_labeled_train_val, supervised_contrastive_loss)
from experiment_protocol import (METHOD_VERSION, grouped_target_split, recording_groups,
                                 save_split_manifest, source_fingerprint)

CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
RUN_CONFIG = {
    "data_root": PROJECT_ROOT / "DATA", "output_dir": CODE_ROOT / "runs" / "industrial_v2",
    "task_preset": "pu", "task_names": None,
    "methods": ["SourceOnly", "TargetFewShot", "SourceTarget", "DeepCORAL", "DANN", "CDAN",
                "DANN_ST", "CDAN_ST", "SSDA", "Ours"],
    "seeds": [42, 2025, 3407], "epochs": 60, "batch_size": 128,
    "lr": 1e-4, "weight_decay": 1e-5, "num_workers": 0, "cpu_threads": 4,
    "img_size": 64, "patch_size": 8, "embed_dim": 128, "depth": 4, "heads": 4,
    "backbone_type": "conv", "pretrained_backbone": False,
    "target_labeled_ratio": 0.15, "target_val_ratio": 0.25, "target_test_ratio": 0.2,
    "augmentation_profile": "fault_light", "normalize_profile": "default",
    "privacy_noise": 0.02, "privacy_dropout": 0.1, "privacy_ramp_epochs": 20,
    "temperature": 0.1, "pseudo_threshold": 0.75, "pseudo_threshold_start": 0.90,
    "pseudo_min_threshold": 0.60, "class_balanced_pseudo": True,
    "pseudo_min_per_class": 1, "pseudo_max_per_class": 0, "pseudo_balance_multiplier": 1.5,
    "source_ce_weight": 1.5, "target_ce_weight": 1.0, "pseudo_weight": 0.7,
    "align_weight": 0.3, "consistency_weight": 0.2,
    "source_rl_weight": 0.1, "target_rl_weight": 0.2, "entropy_weight": 0.01,
    "domain_weight": 0.1, "coral_weight": 1.0,
    "ours_cdan_weight": 0.1, "ours_cdan_start_epoch": 20,
    "ours_cdan_bottleneck": 256, "ours_cdan_dropout": 0.1, "ours_cdan_unlabeled": "all",
    "grad_clip": 1.0, "warmup_epochs": 10, "pseudo_start_epoch": 15,
    "align_start_epoch": 20, "rl_start_epoch": 20, "privacy_start_epoch": 10,
    "selection_start_epoch": 30, "gpu_ids": None, "max_parallel": 1,
    "poll_interval": 5.0, "child_run": False, "quick": False,
    "resume": True, "allow_legacy_image_split": False, "device": "auto",
    "use_privacy": True, "use_rl": True, "use_alignment": True, "use_consistency": True,
}


@dataclass(frozen=True)
class TransferTask:
    dataset: str
    name: str
    source: Path
    target: Path
    task_type: str


def make_tasks(data_root, preset="full"):
    if preset not in {"full", "all", "representative", "pu"}:
        raise ValueError(f"Unknown task preset: {preset}")
    data_root = Path(data_root)
    tasks = []
    for source, target in [("WT1-20", "WT1-35"), ("WT1-20", "WT1-50"), ("WT1-35", "WT1-50"),
                           ("WT1-50", "WT1-50-Y"), ("WT1-50-Y", "WT1-50")]:
        tasks.append(TransferTask("WT1", f"{source}_to_{target}", data_root / "WT1" / source,
                                  data_root / "WT1" / target, "cross_sensor" if "Y" in source + target else "cross_load"))
    for source, target in [("DN", "DN2"), ("DN2", "DN")]:
        tasks.append(TransferTask("DN", f"{source}_to_{target}", data_root / "DN" / source,
                                  data_root / "DN" / target, "cross_condition"))
    base = "N15_M07_F10"
    for other, kind in [("N09_M07_F10", "cross_speed"), ("N15_M01_F10", "cross_torque"),
                        ("N15_M07_F04", "cross_radial_load")]:
        for source, target in [(base, other), (other, base)]:
            tasks.append(TransferTask("PU", f"PU_{source}_to_{target}", data_root / "PU" / source,
                                      data_root / "PU" / target, kind))
    if preset == "all":
        for source, target in [("amazon", "dslr"), ("amazon", "webcam"), ("dslr", "webcam"), ("webcam", "dslr")]:
            tasks.append(TransferTask("Office31", f"office_{source}_to_{target}", data_root / "office31" / source,
                                      data_root / "office31" / target, "generic_visual_supplement"))
    if preset == "representative":
        names = {"WT1-35_to_WT1-50", "DN_to_DN2", f"PU_{base}_to_N09_M07_F10"}
        tasks = [t for t in tasks if t.name in names]
    if preset == "pu":
        tasks = [t for t in tasks if t.dataset == "PU"]
    return tasks


def selected_tasks(args):
    if getattr(args, "source_dir", None) and getattr(args, "target_dir", None):
        return [TransferTask(getattr(args, "dataset_name", "DN"), "single_source_to_target",
                             Path(args.source_dir), Path(args.target_dir), "single_task")]
    tasks = make_tasks(args.data_root, "all" if args.task_names else args.task_preset)
    if args.task_names:
        missing = set(args.task_names) - {t.name for t in tasks}
        if missing:
            raise ValueError(f"Unknown task names: {sorted(missing)}")
        tasks = [t for t in tasks if t.name in args.task_names]
    return tasks[:1] if args.quick else tasks


def serial(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(serial(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_experiment_config(output, config):
    path = Path(output) / "experiment_config.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for key in ["args", "method_version", "source_sha256"]:
            if previous.get(key) != serial(config[key]):
                raise FileExistsError(f"Experiment config/code changed: {path}. Choose a new output_dir.")
    write_json(path, config)


def repeat(loader):
    # Re-enter DataLoader each pass; itertools.cycle would cache augmented batches.
    while True:
        yield from loader


def build_loaders(task, args, seed, artifact_dir=None):
    profile = args.augmentation_profile
    if profile == "auto":
        profile = "generic" if task.dataset == "Office31" else "fault_light"
    normalization = args.normalize_profile
    if normalization == "auto":
        normalization = "imagenet" if args.pretrained_backbone else "default"
    weak, strong, test = build_transforms(args.img_size, profile, normalization)
    source, target = datasets.ImageFolder(task.source), datasets.ImageFolder(task.target)
    if source.classes != target.classes:
        raise ValueError("Source/target class mappings differ")
    source_groups, source_origin = recording_groups(source, task.dataset)
    target_groups, target_origin = recording_groups(target, task.dataset)
    protocol = "original_recording"
    try:
        splits = grouped_target_split(target, target_groups, args.target_labeled_ratio,
                                      args.target_val_ratio, args.target_test_ratio, seed)
    except ValueError:
        if not args.allow_legacy_image_split or task.dataset == "PU":
            raise
        labeled, unlabeled, test_idx = stratified_target_split(target, args.target_labeled_ratio, args.target_test_ratio, seed)
        labeled, val = split_labeled_train_val(target, labeled, args.target_val_ratio, seed)
        if not val:
            raise ValueError("Independent validation labels unavailable")
        splits = {"target_labeled": labeled, "target_val": val, "target_unlabeled": unlabeled, "target_test": test_idx}
        protocol = "legacy_image_split_exploratory"
    sets = {"source": TransformSubset(source, range(len(source)), weak)}
    for name, indices in splits.items():
        sets[name] = TransformSubset(target, indices, test if name in {"target_val", "target_test"} else weak)
    sets["target_unlabeled_two_view"] = TransformSubset(target, splits["target_unlabeled"], weak, True, strong)
    kwargs = {"num_workers": args.num_workers, "pin_memory": torch.cuda.is_available()}
    if args.num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    loaders = {name: DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)),
                               shuffle=name not in {"target_val", "target_test"}, drop_last=False,
                               generator=torch.Generator().manual_seed(seed), **kwargs)
               for name, dataset in sets.items()}
    meta = {"classes": source.classes, "num_classes": len(source.classes),
            "split_protocol": protocol, "augmentation_profile": profile, "normalize_profile": normalization,
            "backbone_type": args.backbone_type, "pretrained_backbone": args.pretrained_backbone,
            **{name + "_samples": len(dataset) for name, dataset in sets.items()}}
    if artifact_dir:
        meta["split_sha256"] = save_split_manifest(
            Path(artifact_dir) / "split_manifest.json", source, target, source_groups, target_groups,
            splits, {"source": source_origin, "target": target_origin, "actual_protocol": protocol})
    return loaders, meta


class GradReverse(Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class ConditionalDomainHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=256, dropout=0.1, conditional=True):
        super().__init__()
        self.conditional = conditional
        self.net = nn.Sequential(nn.Linear(feature_dim * (num_classes if conditional else 1), hidden_dim),
                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))

    def forward(self, features, logits, coefficient):
        if self.conditional:
            features = torch.bmm(logits.softmax(1).unsqueeze(2), features.unsqueeze(1)).flatten(1) / math.sqrt(features.size(1))
        return self.net(GradReverse.apply(features, coefficient))


class DAClassifier(nn.Module):
    def __init__(self, num_classes, args):
        super().__init__()
        self.backbone = build_backbone(args, privacy_noise=0.0, privacy_dropout=0.0)
        self.norm = nn.LayerNorm(self.backbone.feat_dim)
        self.actor = nn.Sequential(nn.Linear(self.backbone.feat_dim, self.backbone.feat_dim),
                                   nn.GELU(), nn.Dropout(0.1), nn.Linear(self.backbone.feat_dim, num_classes))

    def forward(self, images):
        features = self.norm(self.backbone(images, use_privacy=False))
        return self.actor(features), None, features


def linear_ramp(epoch, start, end):
    return 0.0 if epoch < start else min(1.0, (epoch - start + 1) / max(end - start + 1, 1))


def class_balanced_pseudo_mask(confidence, pseudo, num_classes, threshold, args):
    mask = confidence >= threshold
    if not args.class_balanced_pseudo:
        return mask
    cap = args.pseudo_max_per_class or max(args.pseudo_min_per_class, math.ceil(max(float(mask.sum()) / num_classes, 1.0) * args.pseudo_balance_multiplier))
    balanced = torch.zeros_like(mask)
    for label in range(num_classes):
        candidates = torch.where((pseudo == label) & mask)[0]
        selected = candidates[confidence[candidates].argsort(descending=True)[:cap]]
        balanced[selected] = True
        need = max(0, args.pseudo_min_per_class - len(selected))
        candidates = torch.where((pseudo == label) & (confidence >= args.pseudo_min_threshold) & ~balanced)[0]
        balanced[candidates[confidence[candidates].argsort(descending=True)[:need]]] = True
    return balanced


def domain_loss(head, feat_s, logits_s, feat_t, logits_t, coefficient):
    source_logits, target_logits = head(feat_s, logits_s, coefficient), head(feat_t, logits_t, coefficient)
    loss = F.cross_entropy(source_logits, torch.zeros(len(feat_s), dtype=torch.long, device=feat_s.device))
    loss += F.cross_entropy(target_logits, torch.ones(len(feat_t), dtype=torch.long, device=feat_t.device))
    accuracy = 0.5 * ((source_logits.argmax(1) == 0).float().mean() + (target_logits.argmax(1) == 1).float().mean())
    return loss, accuracy.detach()


def coral_loss(source, target):
    source, target = source - source.mean(0), target - target.mean(0)
    cov_s = source.T @ source / max(len(source) - 1, 1)
    cov_t = target.T @ target / max(len(target) - 1, 1)
    return (cov_s - cov_t).square().mean()


def train_epoch(model, head, loaders, optimizer, args, device, classes, epoch, method):
    ours = method.startswith("Ours")
    model.train()
    if head is not None:
        head.train()
    privacy_ramp = linear_ramp(epoch, args.privacy_start_epoch, args.privacy_start_epoch + args.privacy_ramp_epochs - 1)
    model.backbone.privacy_noise = args.privacy_noise * privacy_ramp if ours and args.use_privacy else 0.0
    model.backbone.privacy_drop.p = args.privacy_dropout * privacy_ramp if ours and args.use_privacy else 0.0
    pseudo_ramp = linear_ramp(epoch, args.pseudo_start_epoch, args.epochs)
    threshold = args.pseudo_threshold_start + pseudo_ramp * (args.pseudo_threshold - args.pseudo_threshold_start)
    active = lambda start: epoch > args.warmup_epochs and epoch >= start
    iterators = {name: repeat(loaders[name]) for name in ["source", "target_labeled", "target_unlabeled_two_view"]}
    steps = max(len(loaders["source"]), len(loaders["target_unlabeled_two_view"]))
    totals = {}
    for _ in range(steps):
        x_s, y_s = next(iterators["source"])
        x_l, y_l = next(iterators["target_labeled"])
        x_w, x_a, _ = next(iterators["target_unlabeled_two_view"])
        x_s, y_s, x_l, y_l, x_w, x_a = [x.to(device) for x in [x_s, y_s, x_l, y_l, x_w, x_a]]
        if method == "TargetFewShot":
            components = {"target_ce": args.target_ce_weight * F.cross_entropy(model(x_l)[0], y_l)}
        else:
            z_s, v_s, f_s = model(x_s)
            components = {"source_ce": args.source_ce_weight * F.cross_entropy(z_s, y_s)}
            use_labels = ours or method in {"SourceTarget", "DANN_ST", "CDAN_ST", "SSDA"}
            need_unlabeled = ours or method not in {"SourceOnly", "SourceTarget"}
            if use_labels:
                z_l, _, f_l = model(x_l)
                components["target_ce"] = args.target_ce_weight * F.cross_entropy(z_l, y_l)
            if need_unlabeled:
                z_w, v_w, f_w = model(x_w)
                with torch.no_grad():
                    probabilities = z_w.softmax(1)
                    confidence, pseudo = probabilities.max(1)
                    uncertainty = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1) / math.log(classes)
                    mask = class_balanced_pseudo_mask(confidence, pseudo, classes, threshold, args) if ours else confidence >= args.pseudo_threshold
            if method == "DeepCORAL":
                components["coral"] = args.coral_weight * coral_loss(f_s, f_w)
            if ours or method == "SSDA":
                if (active(args.pseudo_start_epoch) or not ours) and mask.any():
                    components["pseudo_ce"] = args.pseudo_weight * (F.cross_entropy(z_w[mask], pseudo[mask], reduction="none") * confidence[mask]).mean()
                    if ours and args.use_consistency:
                        z_a, _, f_a = model(x_a)
                        components["consistency"] = args.consistency_weight * (F.cross_entropy(z_a[mask], pseudo[mask], reduction="none") * confidence[mask]).mean()
                if ours and args.use_alignment and active(args.align_start_epoch):
                    alignment = supervised_contrastive_loss(torch.cat([f_s, f_l]), torch.cat([y_s, y_l]), args.temperature)
                    alignment += class_prototype_loss(f_s, y_s, f_l, y_l, classes, args.temperature)
                    if active(args.pseudo_start_epoch) and mask.any():
                        if not args.use_consistency:
                            _, _, f_a = model(x_a)
                        alignment += 0.5 * class_prototype_loss(f_s, y_s, torch.cat([f_w[mask], f_a[mask]]), torch.cat([pseudo[mask], pseudo[mask]]), classes, args.temperature)
                    components["alignment"] = args.align_weight * alignment
                if ours and args.use_rl and active(args.rl_start_epoch):
                    source_rl, _ = source_actor_critic_loss(z_s, v_s, y_s, 0.0, args.entropy_weight)
                    components["source_rl"] = args.source_rl_weight * source_rl
                    if mask.any():
                        reward = (confidence - uncertainty).clamp(-1, 1)
                        distribution = torch.distributions.Categorical(logits=z_w[mask])
                        advantage = reward[mask] - v_w[mask].detach()
                        actor = -(distribution.log_prob(pseudo[mask]) * advantage * confidence[mask]).mean()
                        components["target_rl"] = args.target_rl_weight * (actor + 0.5 * F.mse_loss(v_w[mask], reward[mask]))
            if head is not None and (not ours or active(args.ours_cdan_start_epoch)):
                coefficient = linear_ramp(epoch, args.ours_cdan_start_epoch if ours else 1, args.epochs)
                target_features, target_logits = f_w, z_w
                if ours:
                    target_features, target_logits = f_l, z_l
                    keep = torch.ones(len(f_w), dtype=torch.bool, device=device) if args.ours_cdan_unlabeled == "all" else mask
                    if args.ours_cdan_unlabeled != "none" and keep.any():
                        target_features, target_logits = torch.cat([f_l, f_w[keep]]), torch.cat([z_l, z_w[keep]])
                adversarial, domain_accuracy = domain_loss(head, f_s, z_s, target_features, target_logits, coefficient)
                components["domain"] = (args.ours_cdan_weight if ours else args.domain_weight) * adversarial
                totals["domain_acc"] = totals.get("domain_acc", 0.0) + float(domain_accuracy)
            if need_unlabeled:
                totals["pseudo_rate"] = totals.get("pseudo_rate", 0.0) + float(mask.float().mean())
        loss = sum(components.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
        torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
        optimizer.step()
        for name, value in {**components, "loss": loss}.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {**{k: v / steps for k, v in totals.items()}, "privacy_ramp": privacy_ramp if ours and args.use_privacy else 0.0,
            "privacy_noise_actual": model.backbone.privacy_noise, "privacy_dropout_actual": model.backbone.privacy_drop.p,
            "pseudo_threshold": threshold, "rl_active": ours and args.use_rl and active(args.rl_start_epoch),
            "alignment_active": ours and args.use_alignment and active(args.align_start_epoch),
            "cdan_active": ours and head is not None and active(args.ours_cdan_start_epoch)}


@torch.no_grad()
def evaluate_classifier(model, loader, device, num_classes, with_predictions=False):
    model.eval()
    predictions, labels = [], []
    for images, target in loader:
        predictions.extend(model(images.to(device))[0].argmax(1).cpu().tolist())
        labels.extend(target.tolist())
    result = (float(np.mean(np.array(predictions) == labels)),
              float(f1_score(labels, predictions, average="macro", labels=list(range(num_classes)), zero_division=0)))
    return (*result, labels, predictions) if with_predictions else result


ABLATIONS = {"Ours_NoRL": {"use_rl": False}, "Ours_NoPrivacy": {"use_privacy": False},
             "Ours_NoAlign": {"use_alignment": False}, "Ours_NoCDAN": {"ours_cdan_weight": 0.0},
             "Ours_NoConsistency": {"use_consistency": False}, "Ours_NoBalance": {"class_balanced_pseudo": False}}


def run_method(method, loaders, meta, args, device, artifact_dir):
    args = SimpleNamespace(**{**vars(args), **ABLATIONS.get(method, {})})
    ours = method.startswith("Ours")
    if method not in set(RUN_CONFIG["methods"]) | set(ABLATIONS):
        raise ValueError(f"Unknown method: {method}")
    classes = meta["num_classes"]
    model = (ReinforcedPrivacyDA(classes, build_backbone(args)) if ours else DAClassifier(classes, args)).to(device)
    conditional = ours or method.startswith("CDAN")
    head = ConditionalDomainHead(model.backbone.feat_dim, classes, args.ours_cdan_bottleneck,
                                 args.ours_cdan_dropout, conditional).to(device) if (
                                     method in {"DANN", "DANN_ST", "CDAN", "CDAN_ST"} or ours and args.ours_cdan_weight > 0) else None
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best_key, best_epoch = [], (-1.0, -1.0), None
    for epoch in range(1, args.epochs + 1):
        stats = train_epoch(model, head, loaders, optimizer, args, device, classes, epoch, method)
        val_acc, val_f1 = evaluate_classifier(model, loaders["target_val"], device, classes)
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **stats,
               "target_val_acc": val_acc, "target_val_f1": val_f1}
        history.append(row)
        scheduler.step()
        if epoch >= args.selection_start_epoch and (val_acc, val_f1) > best_key:
            best_key, best_epoch = (val_acc, val_f1), epoch
            torch.save({"model": model.state_dict(), "domain_head": head.state_dict() if head is not None else None,
                        "epoch": epoch, "validation": best_key, "training_state": row,
                        "args": serial(vars(args)), "classes": meta["classes"], "method_version": METHOD_VERSION,
                        "source_sha256": meta["source_sha256"], "split_sha256": meta["split_sha256"]},
                       artifact_dir / "best_model.pt")
        write_json(artifact_dir / "progress.json", {"epoch": epoch, "epochs": args.epochs, "best_epoch": best_epoch, **row})
        write_csv(artifact_dir / "train_history.csv", history)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  {method} epoch={epoch}/{args.epochs} loss={stats['loss']:.4f} val={val_acc:.2%} best={best_epoch}", flush=True)
    if best_epoch is None:
        raise RuntimeError("No eligible checkpoint was selected")
    checkpoint = torch.load(artifact_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    accuracy, macro_f1, labels, predictions = evaluate_classifier(model, loaders["target_test"], device, classes, True)
    write_json(artifact_dir / "test_predictions.json", {"labels": labels, "predictions": predictions,
               "classes": meta["classes"], "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(classes))).tolist()})
    return {"target_acc": accuracy * 100, "target_f1": macro_f1, "best_epoch": best_epoch,
            "best_val_acc": best_key[0] * 100, "best_val_f1": best_key[1],
            "best_privacy_noise": checkpoint["training_state"]["privacy_noise_actual"],
            "best_rl_active": checkpoint["training_state"]["rl_active"],
            "best_alignment_active": checkpoint["training_state"]["alignment_active"],
            "best_cdan_active": checkpoint["training_state"]["cdan_active"]}


def summarize(rows):
    summary = []
    keys = sorted({(r["dataset"], r["task"], r["method"]) for r in rows})
    for dataset, task, method in keys:
        selected = [r for r in rows if (r["dataset"], r["task"], r["method"]) == (dataset, task, method)]
        row = {"dataset": dataset, "task": task, "method": method, "runs": len(selected)}
        for metric in ["target_acc", "target_f1"]:
            values = np.array([float(r[metric]) for r in selected])
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def save_results(output, rows):
    for name, values in [("raw", rows), ("summary", summarize(rows))]:
        write_json(output / f"main_comparison_{name}.json", values)
        write_csv(output / f"main_comparison_{name}.csv", values)


def launch_parallel(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(t, m, s) for t in selected_tasks(args) for s in args.seeds for m in args.methods]
    save_experiment_config(output, {"args": vars(args), "method_version": METHOD_VERSION,
               "source_sha256": source_fingerprint(CODE_ROOT), "total_jobs": len(jobs)})
    pending, running, finished, failed, rows = list(jobs), [], 0, [], []
    gpu_ids = [str(g) for g in args.gpu_ids]
    while pending or running:
        for gpu in gpu_ids:
            if not pending or len(running) >= args.max_parallel:
                break
            if any(job["gpu"] == gpu for job in running):
                continue
            task, method, seed = pending.pop(0)
            directory = output / "jobs" / f"{task.name}__{method}__seed{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            config = {**vars(args), "gpu_ids": None, "child_run": True, "output_dir": directory,
                      "task_names": [task.name], "seeds": [seed], "methods": [method]}
            write_json(directory / "child_config.json", config)
            log = (directory / "run.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--config-json", str(directory / "child_config.json")],
                                       stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append({"process": process, "gpu": gpu, "directory": directory, "log": log})
            print(f"LAUNCH gpu={gpu} {task.name} {method} seed={seed}", flush=True)
        time.sleep(min(args.poll_interval, 30))
        for job in running[:]:
            code = job["process"].poll()
            if code is None:
                continue
            job["log"].close()
            running.remove(job)
            finished += 1
            result = job["directory"] / "main_comparison_raw.json"
            if code or not result.exists():
                failed.append(str(job["directory"]))
            else:
                rows.extend(json.loads(result.read_text(encoding="utf-8")))
            save_results(output, rows)
            write_json(output / "progress.json", {"finished": finished, "total": len(jobs), "failed": failed, "running": len(running)})
            print(f"FINISHED {finished}/{len(jobs)}, failed={len(failed)}", flush=True)
    if failed:
        raise RuntimeError(f"Failed experiments: {failed}")


def main(config=None):
    if config is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config-json")
        cli = parser.parse_args()
        config = json.loads(Path(cli.config_json).read_text(encoding="utf-8")) if cli.config_json else RUN_CONFIG
    args = SimpleNamespace(**{**RUN_CONFIG, **config})
    args.data_root, args.output_dir = Path(args.data_root), Path(args.output_dir)
    if args.gpu_ids:
        args.max_parallel = args.max_parallel or len(args.gpu_ids)
        if args.max_parallel < 1:
            raise ValueError("max_parallel must be positive")
        if len({str(gpu) for gpu in args.gpu_ids}) != len(args.gpu_ids):
            raise ValueError("gpu_ids must not contain duplicates")
    if args.quick:
        args.epochs, args.seeds, args.warmup_epochs = 3, args.seeds[:1], 0
        args.pseudo_start_epoch = args.align_start_epoch = args.rl_start_epoch = args.ours_cdan_start_epoch = 1
        args.privacy_start_epoch, args.privacy_ramp_epochs, args.selection_start_epoch = 1, 2, 3
    complete_start = max(args.warmup_epochs + 1, args.pseudo_start_epoch, args.align_start_epoch,
                         args.rl_start_epoch, args.ours_cdan_start_epoch,
                         args.privacy_start_epoch + args.privacy_ramp_epochs - 1)
    if not (complete_start <= args.selection_start_epoch <= args.epochs):
        raise ValueError(f"Checkpoint selection must start after all modules and privacy ramp: required {complete_start}, selection={args.selection_start_epoch}, epochs={args.epochs}")
    if args.gpu_ids and not args.child_run:
        launch_parallel(args)
        return
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = source_fingerprint(CODE_ROOT)
    save_experiment_config(args.output_dir, {"args": vars(args), "device": str(device),
               "method_version": METHOD_VERSION, "source_sha256": fingerprints,
               "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                           "cuda": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                           "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None}})
    tasks = selected_tasks(args)
    rows = []
    total = len(tasks) * len(args.seeds) * len(args.methods)
    for task in tasks:
        for seed in args.seeds:
            for method in args.methods:
                directory = args.output_dir if args.child_run else args.output_dir / "jobs" / f"{task.name}__{method}__seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                seed_everything(seed)
                loaders, meta = build_loaders(task, args, seed, directory)
                meta["source_sha256"] = fingerprints
                signature = hashlib.sha256(json.dumps(serial({"args": vars(args), "task": asdict(task), "method": method,
                    "seed": seed, "code": fingerprints, "split": meta["split_sha256"]}), sort_keys=True).encode()).hexdigest()
                result_file = directory / "result.json"
                if args.resume and result_file.exists():
                    previous = json.loads(result_file.read_text(encoding="utf-8"))
                    if previous["run_signature"] != signature:
                        raise FileExistsError(f"Completed result uses a different config/code: {directory}. Choose a new output_dir.")
                    rows.append(previous)
                    save_results(args.output_dir, rows)
                    continue
                print(f"[{len(rows)+1}/{total}] {task.name} | {method} | seed={seed} | {device}", flush=True)
                start = time.time()
                metrics = run_method(method, loaders, meta, args, device, directory)
                row = {"dataset": task.dataset, "task": task.name, "task_type": task.task_type, "method": method,
                       "seed": seed, "method_version": METHOD_VERSION, "run_signature": signature,
                       "source": str(task.source), "target": str(task.target), **meta,
                       "ours_cdan_weight": args.ours_cdan_weight, **metrics, "elapsed_sec": time.time() - start}
                write_json(result_file, row)
                rows.append(row)
                save_results(args.output_dir, rows)
                write_json(args.output_dir / "progress.json", {"finished": len(rows), "total": total, "last": row})
                print(f"  test={metrics['target_acc']:.2f}% macro_f1={metrics['target_f1']:.4f} best_epoch={metrics['best_epoch']}", flush=True)
                del loaders
    print(f"Completed {len(rows)} runs: {args.output_dir}", flush=True)
    return rows


if __name__ == "__main__":
    main()
