"""One-release federated ridge adaptation with explicit privacy units.

The source encoder and source-only whitening are public. Window DP is a
separate, weaker adjacency than complete-client DP; neither is relabeled as
the other. Raw features and diagnostics are restricted to the trusted process.
"""

import hashlib
import itertools
import json
import math
import secrets
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn import functional as F

import fed_private_adapter as adapter


VERSION = "public-source-one-release-moments-v2"
METHODS = ["PublicSourceOnly", "RidgeNonPrivate", "ClientRidgeNonPrivate", "ClientDPRidge", "WindowDPRidge"]
DIAGNOSTIC_METHODS = ["ClientDPHeadSelect"]


def public_whitening(features, floor, dimension=None):
    mean = features.mean(0)
    centered = features - mean
    covariance = centered.T @ centered / len(features)
    values, vectors = torch.linalg.eigh(covariance.double())
    dimension = min(dimension or features.shape[1], features.shape[1])
    values, vectors = values[-dimension:], vectors[:, -dimension:]
    matrix = vectors * values.clamp_min(max(float(values.max()) * floor, 1e-8)).rsqrt()
    return mean, matrix.float()


def design(features, mean, matrix):
    transformed = (features - mean) @ matrix
    # The bias participates in normalization, so every feature vector is bounded.
    return F.normalize(torch.cat([transformed, torch.ones(len(features), 1)], dim=1), dim=1)


def pack_symmetric(matrix):
    size = matrix.shape[-1]
    i, j = torch.triu_indices(size, size)
    scale = torch.where(i == j, 1., math.sqrt(2.))
    return matrix[..., i, j] * scale


def unpack_symmetric(vector, size):
    i, j = torch.triu_indices(size, size)
    scale = torch.where(i == j, 1., math.sqrt(2.))
    output = vector.new_zeros(size, size)
    values = vector / scale
    output[i, j] = values
    output[j, i] = values
    return output


def window_packets(labeled, labels, unlabeled, classes, use_unlabeled):
    x = torch.cat([labeled, unlabeled]) if use_unlabeled else labeled
    gram = pack_symmetric(x[:, :, None] * x[:, None, :])
    cross = torch.zeros(len(x), x.shape[1], classes)
    cross[:len(labeled)] = labeled[:, :, None] * F.one_hot(labels, classes).float()[:, None, :]
    # Each outer product has norm <=1. Their joint vector has norm <=sqrt(2).
    return torch.cat([gram, cross.flatten(1)], dim=1) / math.sqrt(2.)


def client_packets(packets, clients, labeled_count, use_unlabeled):
    output = []
    for client in clients:
        indices = list(client["labeled"])
        if use_unlabeled:
            indices += [labeled_count + i for i in client["unlabeled"]]
        output.append(packets[indices].mean(0) if indices else torch.zeros(packets.shape[1]))
    return torch.stack(output)


def solve_release(released, dimension, classes, labeled_fraction, ridge):
    split = dimension * (dimension + 1) // 2
    covariance = unpack_symmetric(released[:split] * math.sqrt(2.), dimension)
    cross = released[split:].reshape(dimension, classes) * math.sqrt(2.)
    # The labeled quota is declared public metadata, never inferred from labels.
    cross = cross / labeled_fraction
    values, vectors = torch.linalg.eigh(covariance.double())
    # Projection onto the PSD cone and regularization are DP postprocessing.
    inverse = (vectors * (values.clamp_min(0) + ridge).reciprocal()) @ vectors.T
    return (inverse @ cross.double()).float()


def metrics(scores, labels, classes):
    predictions = scores.argmax(1).numpy()
    return {"accuracy": 100 * float(np.mean(predictions == labels.numpy())),
            "macro_f1": float(f1_score(labels, predictions, labels=list(range(classes)),
                                        average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(classes))).tolist()}


def public_release_provenance(provenance):
    # A hash of a private target split is itself a deterministic private output.
    return {key: provenance[key] for key in ["checkpoint", "checkpoint_sha256", "source_is_public"]}


def moment_budget_report(budget, count, unit):
    return {"epsilon": budget.epsilon, "delta": budget.delta, "release_rounds": 1,
            "fixed_privacy_units": count, "joint_clip": 1.,
            "replacement_sensitivity": 2 / count, "joint_noise_std": budget.sigma(count),
            "rho_total": budget.rho,
            "scope": unit + " replacement, fixed public allocation and label-status metadata",
            "statistic": "joint symmetric feature covariance and labeled cross moment",
            "accounting": "one full-participation Gaussian release; no subsampling amplification",
            "aggregator": "trusted central simulation, no cryptographic secure aggregation"}


def public_candidates(source, source_labels, classes, mean, matrix, ridge, seed):
    """A data-independent/public bank, fixed before accessing any target data."""
    x = design(source, mean, matrix)
    gram = x.T @ x / len(x)
    cross = x.T @ F.one_hot(source_labels, classes).float() / len(x)
    source_ridge = torch.linalg.solve(gram + ridge * torch.eye(x.shape[1]), cross)
    source_centroids = torch.stack([x[source_labels == c].mean(0) for c in range(classes)], dim=1)
    generator = torch.Generator().manual_seed(seed)
    anchors = [source_ridge, source_centroids]
    candidates = []
    for anchor in anchors:
        centered = anchor - anchor.mean(1, keepdim=True)
        centered = centered / centered.norm().clamp_min(1e-8)
        for permutation in itertools.permutations(range(classes)):
            candidates.append(centered[:, list(permutation)])
        for scale in [.1, .3, 1., 3.]:
            for _ in range(64):
                noise = torch.randn(centered.shape, generator=generator)
                noise = noise - noise.mean(1, keepdim=True)
                candidates.append(centered + scale * noise / noise.norm().clamp_min(1e-8))
    for _ in range(256):
        candidates.append(torch.randn(source_ridge.shape, generator=generator))
    return torch.stack(candidates)


def client_utilities(candidates, features, labels, clients):
    predictions = torch.einsum("nd,mdk->mnk", features, candidates).argmax(2)
    utilities = []
    for client in clients:
        indices = client["labeled"]
        utilities.append((predictions[:, indices] == labels[indices]).float().mean(1)
                         if indices else torch.zeros(len(candidates)))
    return torch.stack(utilities)


def exponential_select(utilities, epsilon):
    if (not math.isfinite(epsilon) or epsilon <= 0 or utilities.ndim != 2
            or not torch.isfinite(utilities).all() or (utilities < 0).any() or (utilities > 1).any()):
        raise ValueError("Each complete client's utility must be bounded in [0,1]")
    # Replacing one complete client changes each candidate's summed utility by
    # at most one. This is the pure-epsilon exponential mechanism, not Gaussian DP.
    probabilities = (epsilon / 2 * utilities.double().sum(0)).softmax(0)
    draw = secrets.SystemRandom().random()
    index = int(torch.searchsorted(probabilities.cumsum(0), torch.tensor(draw, dtype=torch.float64)))
    return min(index, utilities.shape[1] - 1)


def prepare_task(task, args, device, directory):
    base = adapter.base
    base.seed_everything(args.seed)
    loaders, meta = adapter.legacy.task_loaders(task, args, args.seed, directory)
    reference = Path(args.public_checkpoint_root) / task.name
    old_manifest = json.loads((reference / "split_manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    if old_manifest != manifest:
        raise ValueError("Public checkpoint reference uses a different split")
    checkpoint_path = reference / "public_source_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("public_source_only") is not True or checkpoint.get("version") != adapter.VERSION:
        raise ValueError("Expected the source-only public encoder checkpoint")
    for name in ["adapter_dim", "img_size", "augmentation_profile", "normalize_profile", "seed"]:
        if checkpoint["args"][name] != getattr(args, name):
            raise ValueError(f"Public source checkpoint argument differs: {name}")
    model = adapter.PublicEncoder(meta["num_classes"], args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    class FeatureView(torch.nn.Module):
        def features(self, images):
            return model.backbone(images, use_privacy=False) if args.feature_kind == "backbone_pca" else model.features(images)
    feature_model = FeatureView()
    _, _, transform = base.build_transforms(args.img_size, args.augmentation_profile, args.normalize_profile)

    def clean(dataset):
        return base.TransformSubset(dataset.image_folder, dataset.indices, transform)

    # Deterministic, per-image transforms prevent target data preprocessing from
    # introducing an unaccounted cross-window dependency.
    source_data = adapter.encode(feature_model, clean(loaders["source"].dataset), args, device)
    source = source_data[0]
    data = {"labeled": adapter.encode(feature_model, clean(loaders["target_labeled"].dataset), args, device),
            "unlabeled": adapter.encode(feature_model, clean(loaders["target_unlabeled_two_view"].dataset), args, device),
            "val": adapter.encode(feature_model, loaders["target_val"].dataset, args, device)}
    if args.evaluate_test:
        data["test"] = adapter.encode(feature_model, loaders["target_test"].dataset, args, device)
    model.cpu()
    for split in ["val", "test"] if args.evaluate_test else ["val"]:
        with torch.no_grad():
            features = F.normalize(model.projector(data[split][0]), dim=1) if args.feature_kind == "backbone_pca" else data[split][0]
            data[split + "_public_scores"] = model.head(features)
    mean, matrix = public_whitening(source, args.whitening_floor, args.moment_dim)
    data["source"] = source_data
    clients = adapter.partition_clients(loaders, manifest, args, args.seed)
    provenance = {"checkpoint": str(checkpoint_path),
                  "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  "split_sha256": meta["split_sha256"], "source_is_public": True}
    return data, clients, mean, matrix, meta, provenance


def run_task(task, args, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    data, clients, mean, matrix, meta, provenance = prepare_task(task, args, device, directory)
    classes = meta["num_classes"]
    lx, _, labels = data["labeled"]
    ux = data["unlabeled"][0]
    labeled = design(lx, mean, matrix)
    unlabeled = design(ux, mean, matrix)
    packets = window_packets(labeled, labels, unlabeled, classes, args.use_unlabeled_covariance)
    fraction = len(labeled) / len(packets)
    budget = adapter.JointBudget(args.epsilon, args.delta, 1)
    candidates = public_candidates(data["source"][0], data["source"][2], classes,
                                   mean, matrix, args.ridge, args.seed) if "ClientDPHeadSelect" in args.methods else None
    rows = []
    for method in args.methods:
        unit, count, private = "none", 0, False
        if method == "PublicSourceOnly":
            state = None
        elif method == "ClientDPHeadSelect":
            private, unit, count = True, "complete_target_client", len(clients)
            utilities = client_utilities(candidates, labeled, labels, clients)
            chosen = exponential_select(utilities, args.epsilon)
            state = candidates[chosen]
            torch.save({"version": VERSION, "weight": state, "whitening_mean": mean,
                        "whitening_matrix": matrix, "method": method, "privacy_unit": unit,
                        "feature_kind": args.feature_kind,
                        "candidate_index": chosen, "candidate_count": len(candidates),
                        "budget": {"epsilon": args.epsilon, "delta": 0., "releases": 1,
                                   "mechanism": "exponential selection", "utility_sensitivity": 1.,
                                   "scope": "one complete target client replacement; source-only public candidate bank"},
                        "provenance": public_release_provenance(provenance)}, directory / f"{method}_released_adapter.pt")
        else:
            private = method in {"ClientDPRidge", "WindowDPRidge"}
            unit = "complete_target_client" if method in {"ClientDPRidge", "ClientRidgeNonPrivate"} else "single_target_window"
            contribution = client_packets(packets, clients, len(labeled), args.use_unlabeled_covariance) if method in {"ClientDPRidge", "ClientRidgeNonPrivate"} else packets
            count = len(contribution)
            released, _ = adapter.aggregate_joint(list(contribution), budget, private)
            # Equal client aggregation is deliberately retained for client DP;
            # individual client sizes are not used as private aggregate weights.
            state = solve_release(released, labeled.shape[1], classes, fraction, args.ridge)
            artifact = {"version": VERSION, "weight": state, "whitening_mean": mean,
                        "whitening_matrix": matrix, "method": method,
                        "feature_kind": args.feature_kind,
                        "privacy_unit": unit if private else None,
                        "budget": moment_budget_report(budget, count, unit) if private else None,
                        "provenance": public_release_provenance(provenance)}
            torch.save(artifact, directory / f"{method}_{'released' if private else 'restricted'}_adapter.pt")
        row = {"dataset": task.dataset, "task": task.name, "method": method,
               "seed": args.seed, "epsilon": args.epsilon if private else None,
               "delta": (0. if method == "ClientDPHeadSelect" else args.delta) if private else None,
               "privacy_unit": unit if private else None,
               "privacy_units": count if private else None,
               "noise_std_normalized_mean": budget.sigma(count) if private and method != "ClientDPHeadSelect" else 0.,
               "release_rounds": int(private), "trainable_parameters": labeled.shape[1] * classes,
               "target_clients": len(clients), "target_labeled_windows": len(labeled),
               "target_unlabeled_windows": len(unlabeled),
               "uses_unlabeled_covariance": args.use_unlabeled_covariance,
               "feature_kind": args.feature_kind, "moment_dim": matrix.shape[1],
               "source_is_public": True, "evaluation_is_restricted": True,
               "version": VERSION, **provenance}
        for split in ["val", "test"] if args.evaluate_test else ["val"]:
            x, _, y = data[split]
            with torch.no_grad():
                scores = data[split + "_public_scores"] if state is None else design(x, mean, matrix) @ state
            result = metrics(scores, y, classes)
            row[split + "_acc"] = result["accuracy"]
            row[split + "_f1"] = result["macro_f1"]
            row[split + "_confusion_matrix"] = result["confusion_matrix"]
        rows.append(row)
        adapter.base.write_json(directory / "restricted_results.json", rows)
        print(f"RESULT {task.dataset} {method} val={row['val_acc']:.2f}%"
              + (f" test={row['test_acc']:.2f}%" if args.evaluate_test else "")
              + f" privacy_unit={row['privacy_unit']} N={count}", flush=True)
    return rows
