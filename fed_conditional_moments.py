"""Private teacher, conditional unlabeled moments and bounded refinement.

Fixed disjoint labeled/unlabeled window slots use adaptive parallel composition.
No target validation/test values participate in fitting or private selection.
"""

import math

import torch
from torch.nn import functional as F

import fed_private_moments as old


VERSION = "conditional-window-moments-v1"
METHODS = ["GlobalCov", "LabelOnly", "ConditionalSSL", "NoClassGate", "NoTrust"]


def psd(matrix, trace_cap=None):
    matrix = (matrix.double() + matrix.double().T) / 2
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_min(0)
    if trace_cap is not None:
        values *= min(1., max(float(trace_cap), 0.) / max(float(values.sum()), 1e-12))
    return (vectors * values) @ vectors.T


def labeled_packets(x, y, classes):
    return old.window_packets(x, y, x[:0], classes, False)


def conditional_packets(x, weak, strong, teacher, classes, args):
    with torch.no_grad():
        probabilities = torch.stack([(view.double() @ teacher / args.teacher_temperature).softmax(1)
                                     for view in [x, weak, strong]])
        target = probabilities.mean(0)
        predictions = probabilities.argmax(2)
        confidence, label = target.max(1)
        agree = (predictions == label[None]).all(0)
        weight = confidence * agree * (confidence >= args.pseudo_confidence)
        responsibilities = target * weight[:, None]
        gram = old.pack_symmetric(x.double()[:, :, None] * x.double()[:, None, :])
        covariances = (responsibilities[:, :, None] * gram[:, None, :]).flatten(1)
        means = (responsibilities[:, :, None] * x.double()[:, None, :]).flatten(1)
        # ||p||_2 <=1, ||x||_2 <=1. Each of the three blocks has norm <=1.
        return torch.cat([covariances, means, responsibilities], dim=1).float() / math.sqrt(3.)


def mean_release(packets, budget, private):
    return old.adapter.aggregate_joint(list(packets), budget, private)[0].double()


def teacher_from_release(released, dimension, classes, ridge):
    split = dimension * (dimension + 1) // 2
    covariance = psd(old.unpack_symmetric(released[:split] * math.sqrt(2.), dimension))
    cross = released[split:].reshape(dimension, classes) * math.sqrt(2.)
    precision = covariance + ridge * torch.eye(dimension, dtype=torch.float64)
    return torch.linalg.solve(precision, cross), precision, cross


def decode_conditional(released, dimension, classes):
    symmetric = dimension * (dimension + 1) // 2
    released = released.double() * math.sqrt(3.)
    covariance = released[:classes * symmetric].reshape(classes, symmetric)
    offset = classes * symmetric
    means = released[offset:offset + classes * dimension].reshape(classes, dimension)
    masses = released[offset + classes * dimension:].clamp(0, 1)
    if len(masses) != classes:
        raise ValueError("Conditional packet has an invalid shape")
    projected = torch.stack([psd(old.unpack_symmetric(covariance[c], dimension), masses[c])
                             for c in range(classes)])
    means = means * (masses / means.norm(dim=1).clamp_min(1e-12)).clamp(max=1.)[:, None]
    return projected, means, masses


def refinement(teacher, precision, cross, released, classes, sigma_raw, args,
               class_gate=True, trust=True):
    dimension = teacher.shape[0]
    covariances, means, masses = decode_conditional(released, dimension, classes)
    if class_gate:
        noise_radius = args.support_noise_multiplier * sigma_raw * math.sqrt(dimension)
        reliability = masses / (masses + noise_radius + 1e-12)
        reliability = reliability * (masses > args.support_noise_multiplier * sigma_raw)
        geometry = F.cosine_similarity(means, cross.T, dim=1)
        geometry = ((geometry - args.prototype_cosine_floor) / (1 - args.prototype_cosine_floor)).clamp(0, 1)
        # Only suppress over-represented pseudo-classes; never amplify rare noise.
        balanced = ((masses.sum() / classes) / masses.clamp_min(1e-12)).clamp(max=1.)
        gate = reliability * geometry * balanced
    else:
        gate = torch.ones(classes, dtype=torch.float64)
    covariance = (covariances * gate[:, None, None]).sum(0)
    target = (means * gate[:, None]).T
    beta = args.unlabeled_weight
    if trust:
        values, vectors = torch.linalg.eigh(precision)
        inverse_root = (vectors * values.rsqrt()) @ vectors.T
        relative_curvature = float(torch.linalg.eigvalsh(inverse_root @ covariance @ inverse_root).max().clamp_min(0))
        beta = min(beta, args.max_relative_curvature / max(relative_curvature, 1e-12))
    # Residual refinement is anchored to the DP teacher, not a raw private head.
    update = torch.linalg.solve(precision + beta * covariance, beta * (target - covariance @ teacher))
    if trust:
        limit = args.max_teacher_change * max(float(teacher.norm()), 1e-12)
        update *= min(1., limit / max(float(update.norm()), 1e-12))
    return teacher + update, {"effective_unlabeled_weight": beta,
                              "noisy_selected_mass": float(masses.sum()),
                              "mean_class_gate": float(gate.mean()),
                              "relative_teacher_change": float(update.norm() / teacher.norm().clamp_min(1e-12))}


def fit(x, y, u, weak, strong, classes, args, private=True):
    budget = old.adapter.JointBudget(args.epsilon, args.delta, 1)
    label_release = mean_release(labeled_packets(x, y, classes), budget, private)
    teacher, precision, cross = teacher_from_release(label_release, x.shape[1], classes, args.ridge)
    # Freeze this DP teacher before computing any per-window unlabeled payload.
    packets = conditional_packets(u, weak, strong, teacher, classes, args)
    unlabeled_release = mean_release(packets, budget, private)
    raw_sigma = math.sqrt(3.) * budget.sigma(len(u)) if private else 0.
    heads, diagnostics = {"LabelOnly": teacher}, {}
    for method, gate, trust in [("ConditionalSSL", True, True),
                                 ("NoClassGate", False, True), ("NoTrust", True, False)]:
        heads[method], diagnostics[method] = refinement(teacher, precision, cross, unlabeled_release,
                                                      classes, raw_sigma, args, gate, trust)
    # A fresh legacy mechanism is a separate comparison, not free postprocessing.
    global_release = mean_release(old.window_packets(x, y, u, classes, True), budget, private)
    heads["GlobalCov"] = old.solve_release(global_release, x.shape[1], classes,
                                           len(x) / (len(x) + len(u)), args.ridge).double()
    if any(not torch.isfinite(head).all() for head in heads.values()):
        raise FloatingPointError("Nonfinite conditional adapter")
    return heads, diagnostics


def privacy_report(args, labeled, unlabeled):
    budget = old.adapter.JointBudget(args.epsilon, args.delta, 1)
    joint_rho = 2 * budget.rho
    return {"unit": "one target training window replacement with fixed public labeled/unlabeled role",
            "epsilon_each_model": args.epsilon, "delta": args.delta,
            "conditional_protocol": "DP labeled teacher, then teacher-conditioned Gaussian unlabeled statistics",
            "composition": "adaptive parallel composition across disjoint labeled/unlabeled slots: max(rho_labeled,rho_unlabeled)",
            "rho_conditional_models_joint": budget.rho,
            "teacher_ablation_models": "LabelOnly, ConditionalSSL, NoClassGate and NoTrust share the same two releases; no extra privacy cost between them",
            "legacy_comparison": "GlobalCov is a separate overlapping-data Gaussian mechanism",
            "all_five_models_joint_rho": joint_rho,
            "all_five_models_joint_epsilon": joint_rho + 2 * math.sqrt(-joint_rho * math.log(args.delta)),
            "labeled_windows": labeled, "unlabeled_windows": unlabeled,
            "labeled_sensitivity": 2 / labeled, "unlabeled_sensitivity": 2 / unlabeled,
            "labeled_normalized_sigma": budget.sigma(labeled),
            "unlabeled_normalized_sigma": budget.sigma(unlabeled),
            "scope": "conditional public source; trusted central aggregation simulation; no client-level DP claim"}


def encode_views(task, args, device, directory):
    old.adapter.base.seed_everything(args.seed + 10007)
    loaders, meta = old.adapter.legacy.task_loaders(task, args, args.seed, directory)
    checkpoint = torch.load(args.public_checkpoint_root / task.name / "public_source_model.pt",
                            map_location="cpu", weights_only=False)
    model = old.adapter.PublicEncoder(meta["num_classes"], args).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)

    class ViewEncoder(torch.nn.Module):
        def features(self, images):
            return model.backbone(images, use_privacy=False)

    weak, strong, _ = old.adapter.encode(ViewEncoder(), loaders["target_unlabeled_two_view"].dataset,
                                         args, device, two_views=True)
    return weak, strong
