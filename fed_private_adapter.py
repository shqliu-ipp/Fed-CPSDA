"""Target-client DP adaptation conditional on an explicitly public source model.

One simulated trusted aggregator releases a joint noised head/prototype packet
per round. Local data, diagnostics and evaluation outputs remain restricted
research artifacts; no cryptographic secure aggregation is implemented here.
"""

import copy
import math
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch import nn
from torch.nn import functional as F

import fed_cpsda_experiment as legacy
import unified_experiment as base


VERSION = "fed-cpsda-public-source-target-client-dp-v2"
METHODS = ["PublicSourceOnly", "FederatedHead", "FederatedHeadDP", "FedProtoDP", "FedCPSDA"]


@dataclass(frozen=True)
class JointBudget:
    epsilon: float
    delta: float
    rounds: int

    def __post_init__(self):
        if (not math.isfinite(self.epsilon) or self.epsilon <= 0 or not 0 < self.delta < 1
                or isinstance(self.rounds, bool) or not isinstance(self.rounds, int) or self.rounds < 1):
            raise ValueError("Invalid joint privacy budget")

    @property
    def rho(self):
        return (self.epsilon / (math.sqrt(-math.log(self.delta) + self.epsilon)
                                + math.sqrt(-math.log(self.delta)))) ** 2

    def sigma(self, clients):
        if clients < 1:
            raise ValueError("A public fixed client roster is required")
        return (2 / clients) * math.sqrt(self.rounds / (2 * self.rho))

    def spent(self, rounds):
        if not 0 <= rounds <= self.rounds:
            raise ValueError("Round count exceeds budget")
        rho = self.rho * rounds / self.rounds
        return rho + 2 * math.sqrt(-rho * math.log(self.delta))

    def report(self, clients):
        return {**asdict(self), "fixed_clients": clients, "joint_clip": 1.0,
                "replacement_sensitivity": 2 / clients, "joint_noise_std": self.sigma(clients),
                "rho_per_round": self.rho / self.rounds,
                "scope": "replacement of one target client's complete training data; fixed public roster",
                "public_condition": "source samples, source-trained encoder/head, client roster and allocation metadata",
                "protection_excludes": "public source; private diagnostics, validation/test data and unnoised local artifacts",
                "aggregator": "trusted central DP simulation; no cryptographic secure aggregation",
                "accounting": "full participation, no subsampling amplification; joint head and prototype vector"}


def secure_gaussian(shape, std, dtype=torch.float32):
    # Public reproducibility seeds must not determine released DP noise.
    rng = secrets.SystemRandom()
    count = math.prod(shape)
    return torch.tensor([rng.normalvariate(0, std) for _ in range(count)], dtype=dtype).reshape(shape)


def aggregate_joint(packets, budget, private=True):
    stacked = torch.stack(packets).float().cpu()
    if not torch.isfinite(stacked).all():
        raise FloatingPointError("Nonfinite joint client update")
    norms = stacked.norm(dim=1, keepdim=True)
    clipped = stacked / norms.clamp_min(1.0)
    mean = clipped.mean(0)
    if private:
        mean += secure_gaussian(mean.shape, budget.sigma(len(packets)))
    return mean, float((norms > 1).float().mean())


class PublicEncoder(nn.Module):
    def __init__(self, classes, args):
        super().__init__()
        self.backbone = base.build_backbone(args, privacy_noise=0., privacy_dropout=0.)
        self.projector = nn.Linear(self.backbone.feat_dim, args.adapter_dim)
        self.head = nn.Linear(args.adapter_dim, classes)

    def features(self, images):
        return F.normalize(self.projector(self.backbone(images, use_privacy=False)), dim=1)

    def forward(self, images):
        return self.head(self.features(images))


def public_pretrain(loaders, classes, args, device, path):
    # Target labels, features and validation are never used to construct this artifact.
    model = PublicEncoder(classes, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.public_lr, weight_decay=args.weight_decay)
    for epoch in range(args.public_epochs):
        model.train()
        loss_sum, count = 0., 0
        for images, labels in loaders["source"]:
            images, labels = images.to(device), labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            count += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"PUBLIC_SOURCE epoch={epoch + 1}/{args.public_epochs} loss={loss_sum / count:.4f}", flush=True)
    model.eval().requires_grad_(False)
    torch.save({"model": model.cpu().state_dict(), "args": base.serial(vars(args)),
                "version": VERSION, "public_source_only": True}, path)
    return model.to(device)


@torch.no_grad()
def encode(model, dataset, args, device, two_views=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    first, second, labels = [], [], []
    model.eval()
    for batch in loader:
        first.append(model.features(batch[0].to(device)).cpu())
        labels.append(batch[-1].cpu())
        if two_views:
            second.append(model.features(batch[1].to(device)).cpu())
    features = torch.cat(first)
    return features, torch.cat(second) if two_views else features.clone(), torch.cat(labels)


def _record_parts(dataset, entries, shards, seed):
    groups = legacy._dataset_groups(dataset, entries)
    return legacy._partition_positions(groups, shards, seed), groups


def partition_clients(loaders, manifest, args, seed):
    # Use a single assignment for TL/U of each physical record; never repeat a
    # labeled shard to fill an unlabeled shard (the legacy modulo did this).
    ld = loaders["target_labeled"].dataset
    ud = loaders["target_unlabeled_two_view"].dataset
    lg = legacy._dataset_groups(ld, manifest["splits"]["target_labeled"])
    ug = legacy._dataset_groups(ud, manifest["splits"]["target_unlabeled"])
    keys = sorted(set(lg + ug))
    if not keys or args.target_clients < 1:
        raise ValueError("No target clients")
    import random
    random.Random(seed).shuffle(keys)
    roster = min(args.target_clients, len(keys))
    owner = {key: i % roster for i, key in enumerate(keys)}
    return [{"labeled": [i for i, key in enumerate(lg) if owner[key] == c],
             "unlabeled": [i for i, key in enumerate(ug) if owner[key] == c],
             "records": [key for key in keys if owner[key] == c]} for c in range(roster)]


def _flat_head(head):
    return torch.cat([p.detach().cpu().flatten() for p in head.parameters()])


def _load_head(vector, classes, dim):
    head = nn.Linear(dim, classes)
    offset = 0
    with torch.no_grad():
        for p in head.parameters():
            p.copy_(vector[offset:offset + p.numel()].reshape_as(p))
            offset += p.numel()
    if offset != len(vector):
        raise ValueError("Head shape mismatch")
    return head


def make_packet(initial, client, data, public_proto, global_proto, classes, args, method, round_index):
    head = _load_head(initial, classes, args.adapter_dim)
    optimizer = torch.optim.SGD(head.parameters(), lr=args.head_lr)
    li, ui = client["labeled"], client["unlabeled"]
    lx, _, ly = data["labeled"]
    ux, ua, _ = data["unlabeled"]
    selected = torch.zeros(len(ui), dtype=torch.bool)
    pseudo = torch.zeros(len(ui), dtype=torch.long)
    threshold = args.pseudo_threshold - (args.pseudo_threshold - args.pseudo_floor) * round_index / max(args.rounds - 1, 1)
    if method == "FedCPSDA" and ui and round_index >= args.pseudo_start_round:
        with torch.no_grad():
            logits = head(ux[ui])
            probabilities = logits.softmax(1)
            confidence, pseudo = probabilities.max(1)
            # Only previously DP-released/global public prototypes guide selection.
            nearest = (ux[ui] @ F.normalize(global_proto, dim=1).T).argmax(1)
            selected = (confidence >= threshold) & (head(ua[ui]).argmax(1) == pseudo) & (nearest == pseudo)
    for _ in range(args.head_steps):
        loss = initial.sum() * 0
        if li:
            loss = F.cross_entropy(head(lx[li]), ly[li])
        if method == "FedCPSDA" and selected.any():
            loss = loss + args.pseudo_weight * F.cross_entropy(head(ua[ui][selected]), pseudo[selected])
        if method in {"FedProtoDP", "FedCPSDA"}:
            # Postprocessing of the previous DP packet; no private cross-client data.
            loss = loss + args.prototype_weight * F.cross_entropy(head(global_proto), torch.arange(classes))
        if not loss.requires_grad:
            break
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite head training")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    update = (_flat_head(head) - initial) / args.head_clip
    prototypes = torch.zeros(classes, args.adapter_dim)
    for c in range(classes):
        vectors = []
        if li:
            vectors.append(lx[li][ly[li] == c])
        if method == "FedCPSDA" and ui:
            vectors.append(ux[ui][selected & (pseudo == c)])
        vectors = [v for v in vectors if len(v)]
        if vectors:
            prototypes[c] = torch.cat(vectors).mean(0)
    # Joint clipping bounds every private output, including class presence. No
    # class counts or masks are released separately. Head-only controls include
    # a zero prototype block for identical packet dimension.
    if method not in {"FedProtoDP", "FedCPSDA"}:
        prototypes.zero_()
    packet = torch.cat([update, (prototypes / args.prototype_scale).flatten()])
    return packet, {"pseudo_selected": int(selected.sum()), "unlabeled_seen": len(ui)}


@torch.no_grad()
def evaluate_head(vector, data, classes, dim):
    x, _, y = data
    prediction = _load_head(vector, classes, dim)(x).argmax(1).numpy()
    return {"accuracy": 100 * float(np.mean(prediction == y.numpy())),
            "macro_f1": float(f1_score(y, prediction, average="macro", labels=list(range(classes)), zero_division=0)),
            "confusion_matrix": confusion_matrix(y, prediction, labels=list(range(classes))).tolist()}


def run_task(task, args, device, directory):
    base.seed_everything(args.seed)
    directory.mkdir(parents=True, exist_ok=True)
    loaders, meta = legacy.task_loaders(task, args, args.seed, directory)
    manifest = __import__("json").loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    model = public_pretrain(loaders, meta["num_classes"], args, device, directory / "public_source_model.pt")
    # Source cache uses deterministic evaluation transforms, with source indices only.
    _, _, transform = base.build_transforms(args.img_size, args.augmentation_profile, args.normalize_profile)
    source = loaders["source"].dataset
    clean_source = base.TransformSubset(source.image_folder, source.indices, transform)
    sx, _, sy = encode(model, clean_source, args, device)
    classes = meta["num_classes"]
    public_proto = torch.stack([sx[sy == c].mean(0) for c in range(classes)])
    data = {"labeled": encode(model, loaders["target_labeled"].dataset, args, device),
            "unlabeled": encode(model, loaders["target_unlabeled_two_view"].dataset, args, device, True),
            "val": encode(model, loaders["target_val"].dataset, args, device),
            "test": encode(model, loaders["target_test"].dataset, args, device)}
    clients = partition_clients(loaders, manifest, args, args.seed)
    # Allocation is a declared public simulation assumption, not a private output.
    base.write_json(directory / "restricted_client_manifest.json", clients)
    del model.backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    initial = _flat_head(model.head)
    budget = JointBudget(args.epsilon, args.delta, args.rounds)
    results = []
    for method in args.methods:
        private = method in {"FederatedHeadDP", "FedProtoDP", "FedCPSDA"}
        state, proto = initial.clone(), public_proto.clone()
        history = []
        steps = 0 if method == "PublicSourceOnly" else args.rounds
        print(f"ADAPT {task.name} {method} clients={len(clients)} parameters={len(state)}", flush=True)
        for ri in range(steps):
            packets, diagnostics = [], []
            for client in clients:
                packet, stats = make_packet(state, client, data, public_proto, proto, classes, args, method, ri)
                packets.append(packet)
                diagnostics.append(stats)
            released, clipped = aggregate_joint(packets, budget, private)
            state += args.server_lr * args.head_clip * released[:len(state)]
            if method in {"FedProtoDP", "FedCPSDA"}:
                proposed = released[len(state):].reshape(classes, args.adapter_dim) * args.prototype_scale
                # Public-source shrinkage reduces the effect of noisy/absent classes.
                proto = (1 - args.prototype_mix) * public_proto + args.prototype_mix * proposed
            metrics = evaluate_head(state, data["val"], classes, args.adapter_dim)
            seen = sum(d["unlabeled_seen"] for d in diagnostics)
            row = {"round": ri + 1, "val_accuracy": metrics["accuracy"],
                   "restricted_pseudo_rate": sum(d["pseudo_selected"] for d in diagnostics) / max(seen, 1),
                   "restricted_clipped_fraction": clipped,
                   "epsilon_spent": budget.spent(ri + 1) if private else None}
            history.append(row)
            print(f"  round={ri + 1}/{steps} val={metrics['accuracy']:.2f}% epsilon={row['epsilon_spent']}", flush=True)
        # Fixed final round, no private validation-driven model selection.
        test = evaluate_head(state, data["test"], classes, args.adapter_dim)
        base.write_json(directory / f"{method}_restricted_history.json", history)
        torch.save({"head": state, "prototype": proto, "version": VERSION,
                    "budget": budget.report(len(clients)) if private else None,
                    "public_source_checkpoint": "public_source_model.pt"}, directory / f"{method}_released_adapter.pt")
        row = {"dataset": task.dataset, "task": task.name, "method": method,
               "seed": args.seed, "target_acc": test["accuracy"], "target_f1": test["macro_f1"],
               "epsilon": args.epsilon if private else None, "delta": args.delta if private else None,
               "rounds": steps, "clients": len(clients), "trainable_parameters": len(state),
               "split_sha256": meta["split_sha256"], "source_is_public": True,
               "version": VERSION, "test_results_are_restricted_evaluation": True,
               "confusion_matrix": test["confusion_matrix"]}
        results.append(row)
        base.write_json(directory / "results.json", results)
        print(f"RESULT {method} test={row['target_acc']:.2f}%", flush=True)
    return results
