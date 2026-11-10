"""Variant A: contrast cross-moments from labels, covariance from U only."""

import math

import torch

import fed_covariance_borrowing as previous


VERSION = "cross-only-unlabeled-covariance-v1"


def labeled_packets(x, y, classes):
    bound = math.sqrt(1 - 1 / classes)
    basis = previous.contrasts(classes)
    return ((x.double()[:, :, None] * basis[y][:, None, :]).flatten(1) / bound).float()


def fit(x, y, u, classes, args, private=True):
    if not len(x) or not len(u) or classes < 2:
        raise ValueError("Nonempty disjoint labeled/unlabeled slots and K>=2 required")
    budget = previous.old.adapter.JointBudget(args.epsilon, args.delta, 1)
    bound = math.sqrt(1 - 1 / classes)
    cross = previous.mean_release(labeled_packets(x, y, classes), budget, private)
    cross = cross.reshape(x.shape[1], classes - 1) * bound
    released_cov = previous.mean_release(previous.unlabeled_packets(u), budget, private)
    covariance = previous.trace_one(previous.old.unpack_symmetric(released_cov, x.shape[1]))
    weight = torch.linalg.solve(covariance + args.ridge * torch.eye(x.shape[1], dtype=torch.float64), cross)
    weight = weight @ previous.contrasts(classes).T
    if not torch.isfinite(weight).all():
        raise FloatingPointError("Nonfinite variant-A head")
    return weight


def privacy_report(args, labeled, unlabeled, classes):
    budget = previous.old.adapter.JointBudget(args.epsilon, args.delta, 1)
    bound = math.sqrt(1 - 1 / classes)
    return {
        "epsilon_each_model": args.epsilon, "delta": args.delta,
        "rho": budget.rho,
        "unit": "one target training window replacement; fixed public role and counts",
        "labeled_windows": labeled, "unlabeled_windows": unlabeled,
        "label_contribution_bound": bound,
        "label_mean_sensitivity": 2 * bound / labeled,
        "unlabeled_mean_sensitivity": 2 / unlabeled,
        "cross_noise_std": bound * budget.sigma(labeled),
        "covariance_packed_noise_std": budget.sigma(unlabeled),
        "cross_noise_ratio_to_previous_joint": bound / math.sqrt(2 - 1 / classes),
        "composition": "parallel disjoint labeled cross-moment and unlabeled covariance releases",
        "scope": "public-source conditional window DP; trusted central simulation",
        "excludes": "client/record privacy, role changes, preprocessing, metrics and nonprivate models",
    }
