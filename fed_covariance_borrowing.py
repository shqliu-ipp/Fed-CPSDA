"""Contrast-coded private labels and noise-aware covariance borrowing.

The labeled and unlabeled releases protect fixed disjoint window slots in
parallel. The shrinkage rule only uses released statistics and known noise.
"""

import math

import torch

import fed_private_moments as old
from fed_conditional_moments import mean_release, psd


VERSION = "contrast-covariance-borrowing-v1"
METHODS = ["GlobalCov", "ContrastLabel", "AdaptiveBorrow", "NoTrace", "FixedBorrow"]


def contrasts(classes):
    basis = torch.zeros(classes, classes - 1, dtype=torch.float64)
    for j in range(classes - 1):
        denominator = math.sqrt((j + 1) * (j + 2))
        basis[:j + 1, j] = 1 / denominator
        basis[j + 1, j] = -(j + 1) / denominator
    return basis


def labeled_packets(x, y, classes):
    basis = contrasts(classes)
    gram = old.pack_symmetric(x.double()[:, :, None] * x.double()[:, None, :])
    cross = (x.double()[:, :, None] * basis[y][:, None, :]).flatten(1)
    bound = math.sqrt(2 - 1 / classes)
    return torch.cat([gram, cross], dim=1).float() / bound


def unlabeled_packets(x):
    return old.pack_symmetric(x[:, :, None] * x[:, None, :])


def trace_one(matrix):
    """Euclidean projection onto PSD matrices with the known trace of one."""
    values, vectors = torch.linalg.eigh((matrix.double() + matrix.double().T) / 2)
    ordered = values.sort(descending=True).values
    index = torch.arange(1, len(values) + 1, dtype=torch.float64)
    shifts = (ordered.cumsum(0) - 1) / index
    active = torch.nonzero(ordered > shifts).flatten()
    threshold = shifts[int(active[-1])]
    projected = (values - threshold).clamp_min(0)
    return (vectors * projected) @ vectors.T


def borrowing_weight(labeled, unlabeled, sigma_l, sigma_u, cap):
    dimension = labeled.shape[0]
    entries = dimension * (dimension + 1) / 2
    variance_l = entries * sigma_l ** 2
    variance_u = entries * sigma_u ** 2
    squared_gap = float((labeled - unlabeled).square().sum())
    # Isometric packing makes the Gaussian Frobenius noise variance known.
    # This estimates disparity of fixed empirical covariances, without assuming
    # that correlated windows are independent population observations.
    estimated_disparity = max(0., squared_gap - variance_l - variance_u)
    alpha = min(cap, variance_l / max(variance_l + variance_u + estimated_disparity, 1e-20))
    return alpha, {"covariance_squared_gap": squared_gap,
                   "estimated_empirical_disparity": estimated_disparity,
                   "labeled_covariance_noise_energy": variance_l,
                   "unlabeled_covariance_noise_energy": variance_u}


def fit(x, y, u, classes, args, private=True):
    budget = old.adapter.JointBudget(args.epsilon, args.delta, 1)
    dimension = x.shape[1]
    symmetric = dimension * (dimension + 1) // 2
    bound = math.sqrt(2 - 1 / classes)
    label = mean_release(labeled_packets(x, y, classes), budget, private) * bound
    cov_l = old.unpack_symmetric(label[:symmetric], dimension)
    cross = label[symmetric:].reshape(dimension, classes - 1)
    cov_u = old.unpack_symmetric(mean_release(unlabeled_packets(u), budget, private), dimension)
    sigma_l = bound * budget.sigma(len(x)) if private else 0.
    sigma_u = budget.sigma(len(u)) if private else 0.
    alpha, diagnostic = borrowing_weight(cov_l, cov_u, sigma_l, sigma_u, args.borrowing_cap)
    basis = contrasts(classes)
    heads, diagnostics = {}, {}
    for method, weight, use_trace in [("ContrastLabel", 0., True), ("AdaptiveBorrow", alpha, True),
                                       ("NoTrace", alpha, False), ("FixedBorrow", args.borrowing_cap, True)]:
        covariance = (1 - weight) * cov_l + weight * cov_u
        covariance = trace_one(covariance) if use_trace else psd(covariance)
        coefficient = torch.linalg.solve(covariance + args.ridge * torch.eye(dimension, dtype=torch.float64), cross)
        heads[method] = coefficient @ basis.T
        diagnostics[method] = {**diagnostic, "borrowing_weight": weight,
                                "projected_covariance_trace": float(covariance.trace())}
    # Retain the previous global-covariance mechanism as an independent control.
    released = mean_release(old.window_packets(x, y, u, classes, True), budget, private)
    heads["GlobalCov"] = old.solve_release(released, dimension, classes,
                                           len(x) / (len(x) + len(u)), args.ridge).double()
    if any(not torch.isfinite(head).all() for head in heads.values()):
        raise FloatingPointError("Nonfinite covariance-borrowing adapter")
    return heads, diagnostics


def privacy_report(args, labeled, unlabeled, classes):
    budget = old.adapter.JointBudget(args.epsilon, args.delta, 1)
    rho = 2 * budget.rho
    return {"unit": "one target training window replacement; fixed public role and counts",
            "epsilon_each_model": args.epsilon, "delta": args.delta,
            "label_contribution_bound": math.sqrt(2 - 1 / classes),
            "label_code_dimension": classes - 1,
            "composition": "parallel composition of disjoint labeled contrast moments and unlabeled covariance",
            "rho_borrowing_family_joint": budget.rho,
            "all_five_models_joint_rho": rho,
            "all_five_models_joint_epsilon": rho + 2 * math.sqrt(-rho * math.log(args.delta)),
            "labeled_windows": labeled, "unlabeled_windows": unlabeled,
            "scope": "conditional public source; window DP, trusted central simulation",
            "note": "ContrastLabel, AdaptiveBorrow, NoTrace and FixedBorrow share two releases. GlobalCov is a separate overlapping Gaussian mechanism."}
