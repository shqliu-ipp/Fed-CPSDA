"""Fed-CPSDA: condition-aware private federated semi-supervised adaptation.

The implementation is an experimental research artifact. Secure aggregation is
simulated by aggregating client updates in the server process; no raw samples
or client-level updates are written to disk.
"""

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Subset

import gear_experiment_protocol as gear_protocol
import unified_experiment as base


VERSION = "fed-cpsda-v1"
METHODS = ["FedAvg", "FedAvgDP", "FedProtoDP", "FedCPSDA"]


def rdp_budget(epsilon, delta, queries):
    log_delta = -math.log(delta)
    rho = (epsilon / (math.sqrt(log_delta + epsilon) + math.sqrt(log_delta))) ** 2
    return rho, 2 * math.sqrt(queries / (2 * rho))


@dataclass(frozen=True)
class ClientBudget:
    epsilon: float = 8.0
    delta: float = 1e-5
    rounds: int = 20
    clip: float = 0.25

    @property
    def rho(self):
        return rdp_budget(self.epsilon, self.delta, 1)[0]

    def aggregate_noise_std(self, clients):
        _, base_sigma = rdp_budget(self.epsilon, self.delta, self.rounds)
        return 2 * self.clip * base_sigma / max(clients, 1)

    def report(self, clients):
        return {**asdict(self), "rho_per_round": self.rho,
                "aggregate_noise_std": self.aggregate_noise_std(clients),
                "scope": "client-level model-update release under simulated secure aggregation",
                "training_data_dp": True}


class FedClassifier(nn.Module):
    def __init__(self, classes, args):
        super().__init__()
        self.backbone = base.build_backbone(args, privacy_noise=0.0, privacy_dropout=0.0)
        self.norm = nn.LayerNorm(self.backbone.feat_dim)
        self.head = nn.Sequential(nn.Linear(self.backbone.feat_dim, self.backbone.feat_dim),
                                  nn.GELU(), nn.Dropout(0.1), nn.Linear(self.backbone.feat_dim, classes))

    def forward(self, images):
        features = self.norm(self.backbone(images, use_privacy=False))
        return self.head(features), features


def _relative_path(image, root):
    return Path(image).relative_to(root).as_posix()


def _dataset_groups(dataset, entries):
    root = Path(dataset.image_folder.root)
    mapping = {}
    for row in entries:
        image = Path(row["image"])
        # Gear manifests are relative to the dataset root; PU split manifests
        # are already relative to the individual condition's ImageFolder.
        if "domain" in row:
            if row["domain"] != root.name:
                raise ValueError(f"Group provenance domain differs from {root.name}: {row['domain']}")
            image = image.relative_to(row["domain"])
        if image.is_absolute() or ".." in image.parts:
            raise ValueError(f"Invalid group provenance path: {row['image']}")
        key = image.as_posix()
        if key in mapping:
            raise ValueError(f"Duplicate group provenance for {key}")
        mapping[key] = row["record_id"]
    result = []
    for base_index in dataset.indices:
        image, _ = dataset.image_folder.samples[base_index]
        key = _relative_path(image, root)
        if key not in mapping:
            raise ValueError(f"Missing group provenance for {key}")
        result.append(mapping[key])
    return result


def _partition_positions(groups, shards, seed):
    keys = sorted(set(groups))
    if len(keys) < 2:
        return [list(range(len(groups)))]
    rng = random.Random(seed)
    rng.shuffle(keys)
    assignments = {key: index % min(shards, len(keys)) for index, key in enumerate(keys)}
    result = [[] for _ in range(max(assignments.values()) + 1)]
    for position, key in enumerate(groups):
        result[assignments[key]].append(position)
    return [part for part in result if part]


def _loader(dataset, positions, batch_size, seed, two_views=False):
    subset = Subset(dataset, positions)
    return DataLoader(subset, batch_size=min(batch_size, len(subset)), shuffle=True,
                      drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


@dataclass
class Client:
    name: str
    kind: str
    labeled: object
    unlabeled: object
    weight: int


def make_clients(loaders, artifact_dir, args, seed):
    manifest = json.loads((Path(artifact_dir) / "split_manifest.json").read_text(encoding="utf-8"))
    clients = []
    source_ds = loaders["source"].dataset
    source_groups = _dataset_groups(source_ds, manifest["splits"]["source"])
    for index, positions in enumerate(_partition_positions(source_groups, args.client_shards, seed)):
        clients.append(Client(f"source_{index}", "source", _loader(source_ds, positions, args.batch_size, seed + index),
                              None, len(positions)))

    labeled_ds = loaders["target_labeled"].dataset
    unlabeled_ds = loaders["target_unlabeled_two_view"].dataset
    labeled_groups = _dataset_groups(labeled_ds, manifest["splits"]["target_labeled"])
    unlabeled_groups = _dataset_groups(unlabeled_ds, manifest["splits"]["target_unlabeled"])
    labeled_parts = _partition_positions(labeled_groups, args.client_shards, seed + 11)
    unlabeled_parts = _partition_positions(unlabeled_groups, args.client_shards, seed + 17)
    count = max(len(labeled_parts), len(unlabeled_parts))
    for index in range(count):
        lp = labeled_parts[index % len(labeled_parts)] if labeled_parts else []
        up = unlabeled_parts[index % len(unlabeled_parts)] if unlabeled_parts else []
        labeled = _loader(labeled_ds, lp, args.batch_size, seed + 100 + index) if lp else None
        unlabeled = _loader(unlabeled_ds, up, args.batch_size, seed + 200 + index, True) if up else None
        clients.append(Client(f"target_{index}", "target", labeled, unlabeled, len(lp) + len(up)))
    if len(clients) < 2:
        raise ValueError("Fed-CPSDA requires at least two clients")
    return clients


def _cycle(loader):
    while True:
        for item in loader:
            yield item


def _prototype_loss(features, labels, prototypes):
    if not prototypes:
        return features.sum() * 0
    valid = torch.tensor([int(y) in prototypes for y in labels.detach().cpu().tolist()], device=features.device)
    if not valid.any():
        return features.sum() * 0
    target = torch.stack([prototypes[int(y)] for y in labels[valid].detach().cpu().tolist()]).to(features.device)
    return (1 - F.cosine_similarity(features[valid], target, dim=1)).mean()


def _pseudo_from_prototypes(features, logits, prototypes, threshold):
    if not prototypes:
        probabilities = logits.softmax(1)
    else:
        matrix = torch.stack([prototypes[i] for i in sorted(prototypes)]).to(features.device)
        proto_logits = 4 * F.normalize(features, dim=1) @ F.normalize(matrix, dim=1).T
        probabilities = (logits.softmax(1) + proto_logits.softmax(1)).div(2)
    confidence, labels = probabilities.max(1)
    return labels, confidence, confidence >= threshold


def _local_train(global_state, client, classes, prototypes, args, device, method, round_index):
    model = FedClassifier(classes, args).to(device)
    model.load_state_dict(global_state)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(len(client.labeled) if client.labeled else 0, len(client.unlabeled) if client.unlabeled else 0,
                len(client.labeled) if client.kind == "source" else 0)
    if steps == 0:
        return global_state, {}, 0, 0.0
    labeled_stream = _cycle(client.labeled) if client.labeled else None
    unlabeled_stream = _cycle(client.unlabeled) if client.unlabeled else None
    pseudo_count = 0
    seen_count = 0
    for _ in range(args.local_epochs):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            if client.kind == "source":
                images, labels = next(labeled_stream)
                images, labels = images.to(device), labels.to(device)
                logits, features = model(images)
                loss = F.cross_entropy(logits, labels)
                if method in {"FedProtoDP", "FedCPSDA"}:
                    loss = loss + args.prototype_weight * _prototype_loss(features, labels, prototypes)
            else:
                images, labels = next(labeled_stream) if labeled_stream else (None, None)
                weak, strong, _ = next(unlabeled_stream) if unlabeled_stream else (None, None, None)
                loss = torch.zeros((), device=device)
                if images is not None:
                    logits, features = model(images.to(device))
                    loss = loss + F.cross_entropy(logits, labels.to(device))
                    if method in {"FedProtoDP", "FedCPSDA"}:
                        loss = loss + args.prototype_weight * _prototype_loss(features, labels.to(device), prototypes)
                if weak is not None and method == "FedCPSDA":
                    with torch.no_grad():
                        weak_logits, weak_features = model(weak.to(device))
                        threshold = max(args.pseudo_floor, args.pseudo_threshold -
                                        args.threshold_decay * round_index / max(args.rounds - 1, 1))
                        pseudo, confidence, mask = _pseudo_from_prototypes(weak_features, weak_logits, prototypes, threshold)
                    strong_logits, _ = model(strong.to(device))
                    seen_count += len(mask)
                    pseudo_count += int(mask.sum())
                    if mask.any():
                        loss = loss + args.pseudo_weight * (confidence[mask] *
                            F.cross_entropy(strong_logits[mask], pseudo[mask], reduction="none")).mean()
            if torch.isfinite(loss) and loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
    delta = {key: model.state_dict()[key].detach().cpu() - value.detach().cpu() for key, value in global_state.items()}
    local_prototypes = {}
    with torch.no_grad():
        if client.labeled:
            for images, labels in client.labeled:
                labels = labels.to(device)
                _, features = model(images.to(device))
                for label in labels.tolist():
                    local_prototypes.setdefault(int(label), []).append(features[labels == label].mean(0).cpu())
    local_prototypes = {key: F.normalize(torch.stack(value).mean(0), dim=0) for key, value in local_prototypes.items()}
    return delta, local_prototypes, client.weight, pseudo_count / max(seen_count, 1)


def _clip_delta(delta, radius):
    norm = torch.sqrt(sum(value.float().square().sum() for value in delta.values()))
    scale = min(1.0, radius / max(float(norm), 1e-12))
    return {key: value * scale for key, value in delta.items()}, float(norm), scale < 1


def _aggregate(state, updates, prototypes, args, method, budget):
    total = sum(weight for _, _, weight, _ in updates)
    merged = {key: torch.zeros_like(value, dtype=torch.float32) for key, value in state.items()}
    clipped = 0
    for delta, _, weight, _ in updates:
        current = delta
        if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"}:
            current, _, was_clipped = _clip_delta(current, budget.clip)
            clipped += int(was_clipped)
        for key in merged:
            merged[key] += current[key].float() * (weight / total)
    if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"}:
        std = budget.aggregate_noise_std(len(updates))
        for key in merged:
            if merged[key].is_floating_point():
                merged[key] += std * torch.randn_like(merged[key])
    new_state = {key: (state[key].float() + merged[key]).to(state[key].dtype) for key in state}
    proto_out = {}
    available = sorted({key for _, local_proto, _, _ in updates for key in local_proto})
    for key in available:
        vectors = [item[1][key] for item in updates if key in item[1]]
        if vectors:
            proto_out[key] = F.normalize(torch.stack(vectors).mean(0), dim=0)
    return new_state, proto_out, clipped


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    labels, predictions = [], []
    for images, y in loader:
        predictions.extend(model(images.to(device))[0].argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    return {"accuracy": float(np.mean(np.asarray(predictions) == labels)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(classes)), zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(classes))).tolist(),
            "labels": labels, "predictions": predictions}


def run_method(task, method, loaders, meta, args, device, artifact_dir, seed):
    if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"}:
        raise RuntimeError(
            "Legacy v1 DP training is disabled: weighted-update sensitivity, "
            "prototype releases and private model selection were not jointly accounted. "
            "Use run_fed_private_adapter.py for the client-DP diagnostic or "
            "run_fed_private_moments.py for explicitly separated client/window DP. "
            "Window DP is not complete-client DP.")
    clients = make_clients(loaders, artifact_dir, args, seed)
    model = FedClassifier(meta["num_classes"], args).to(device)
    # Keep the simulated server model on CPU between client rounds. This also
    # makes the secure aggregation path explicit and avoids device-dependent
    # update arithmetic.
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    prototypes = {}
    budget = ClientBudget(args.epsilon, args.delta, args.rounds, args.client_clip)
    history, best = [], (-1.0, -1.0)
    best_state, best_round = None, None
    for round_index in range(args.rounds):
        updates = []
        rates = []
        for client in clients:
            delta, local_proto, weight, pseudo_rate = _local_train(
                state, client, meta["num_classes"], prototypes, args, device, method, round_index)
            updates.append((delta, local_proto, weight, client.name))
            rates.append(pseudo_rate)
        state, prototypes, clipped = _aggregate(state, updates, prototypes, args, method, budget)
        model.load_state_dict(state)
        val = evaluate(model, loaders["target_val"], device, meta["num_classes"])
        row = {"round": round_index + 1, "val_acc": val["accuracy"], "val_f1": val["macro_f1"],
               "pseudo_rate": float(np.mean(rates)), "clients": len(clients),
               "clipped_clients": clipped, "prototype_classes": len(prototypes),
               "epsilon": args.epsilon if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"} else None,
               "aggregate_noise_std": budget.aggregate_noise_std(len(clients)) if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"} else 0.0}
        history.append(row)
        if (val["accuracy"], val["macro_f1"]) > best:
            best = (val["accuracy"], val["macro_f1"])
            best_round = round_index + 1
            best_state = copy.deepcopy(state)
        base.write_json(Path(artifact_dir) / "progress.json", {"round": round_index + 1, "rounds": args.rounds, **row})
        base.write_csv(Path(artifact_dir) / "train_history.csv", history)
        if round_index == 0 or (round_index + 1) % 5 == 0 or round_index + 1 == args.rounds:
            print(f"  {method} round={round_index + 1}/{args.rounds} val={val['accuracy']:.2%} "
                  f"pseudo={row['pseudo_rate']:.2%} proto={row['prototype_classes']}", flush=True)
    model.load_state_dict(best_state)
    test = evaluate(model, loaders["target_test"], device, meta["num_classes"])
    base.write_json(Path(artifact_dir) / "test_predictions.json", test)
    base.write_json(Path(artifact_dir) / "privacy_receipt.json", {
        "method": method, "version": VERSION, "budget": budget.report(len(clients)),
        "secure_aggregation": method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"},
        "raw_samples_exported": False, "client_updates_exported": False,
        "best_round": best_round, "clients": [client.name for client in clients]})
    return {"target_acc": test["accuracy"] * 100, "target_f1": test["macro_f1"],
            "best_round": best_round, "best_val_acc": best[0] * 100, "best_val_f1": best[1],
            "num_clients": len(clients), "mean_pseudo_rate": float(np.mean([r["pseudo_rate"] for r in history])),
            "epsilon": args.epsilon if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"} else None,
            "aggregate_noise_std": budget.aggregate_noise_std(len(clients)) if method in {"FedAvgDP", "FedProtoDP", "FedCPSDA"} else 0.0,
            "privacy_scope": "client-level model update; simulated secure aggregation"}


def task_loaders(task, args, seed, artifact_dir):
    if task.dataset == "PU":
        return base.build_loaders(task, args, seed, artifact_dir)
    return gear_protocol.build_loaders(task, args, seed, artifact_dir)

