"""Self-contained CrossOnlyDP/A statistic-release head.

The source encoder is public and frozen. Labeled target features release a
contrast-coded cross moment, while unlabeled target features release a second
moment. Both packets are clipped before Gaussian release. The noisy second
moment is projected onto the PSD trace-one set and the final head is obtained
by a ridge solve.
"""

import math
import secrets
from dataclasses import dataclass

import torch


VERSION = "cross-only-unlabeled-covariance-v1"


@dataclass(frozen=True)
class JointBudget:
    """One zCDP-calibrated release with a fixed public roster."""

    epsilon: float
    delta: float
    rounds: int = 1

    def __post_init__(self):
        if (not math.isfinite(self.epsilon) or self.epsilon <= 0
                or not 0 < self.delta < 1 or self.rounds < 1):
            raise ValueError("Invalid privacy budget")

    @property
    def rho(self):
        return (self.epsilon / (math.sqrt(-math.log(self.delta) + self.epsilon)
                                + math.sqrt(-math.log(self.delta)))) ** 2

    def sigma(self, count):
        if count < 1:
            raise ValueError("A nonempty protected packet set is required")
        return (2 / count) * math.sqrt(self.rounds / (2 * self.rho))


def _gaussian(shape, std, dtype):
    generator = secrets.SystemRandom()
    values = [generator.normalvariate(0.0, std) for _ in range(math.prod(shape))]
    return torch.tensor(values, dtype=dtype).reshape(shape)


def _mean_release(packets, budget, private):
    stacked = torch.stack(list(packets)).float()
    if not torch.isfinite(stacked).all():
        raise FloatingPointError("Nonfinite statistic packet")
    norms = stacked.norm(dim=1, keepdim=True)
    released = (stacked / norms.clamp_min(1.0)).mean(0)
    if private:
        released = released + _gaussian(tuple(released.shape), budget.sigma(len(stacked)), released.dtype)
    return released.double()


def pack_symmetric(matrix):
    size = matrix.shape[-1]
    i, j = torch.triu_indices(size, size, device=matrix.device)
    scale = torch.where(i == j, 1.0, math.sqrt(2.0)).to(matrix)
    return matrix[..., i, j] * scale


def unpack_symmetric(vector, size):
    i, j = torch.triu_indices(size, size, device=vector.device)
    scale = torch.where(i == j, 1.0, math.sqrt(2.0)).to(vector)
    output = vector.new_zeros(size, size)
    values = vector / scale
    output[i, j] = values
    output[j, i] = values
    return output


def contrasts(classes):
    """Return an orthonormal Helmert contrast basis for ``classes`` labels."""
    if classes < 2:
        raise ValueError("At least two classes are required")
    basis = torch.zeros(classes, classes - 1, dtype=torch.float64)
    for column in range(classes - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[:column + 1, column] = 1 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def labeled_packets(features, labels, classes):
    bound = math.sqrt(1 - 1 / classes)
    basis = contrasts(classes).to(features.device)
    return ((features.double()[:, :, None] * basis[labels][:, None, :])
            .flatten(1) / bound).float()


def unlabeled_packets(features):
    return pack_symmetric(features[:, :, None] * features[:, None, :])


def trace_one(matrix):
    """Project a symmetric matrix onto the PSD trace-one set."""
    values, vectors = torch.linalg.eigh((matrix.double() + matrix.double().T) / 2)
    ordered = values.sort(descending=True).values
    index = torch.arange(1, len(values) + 1, dtype=torch.float64, device=values.device)
    shifts = (ordered.cumsum(0) - 1) / index
    active = torch.nonzero(ordered > shifts).flatten()
    threshold = shifts[int(active[-1])]
    projected = (values - threshold).clamp_min(0)
    return (vectors * projected) @ vectors.T


def fit(features_labeled, labels, features_unlabeled, classes, args, private=True):
    """Fit the released CrossOnlyDP/A diagnosis head.

    ``features_labeled`` and ``features_unlabeled`` must already be mapped by
    the frozen public source encoder and normalized to unit norm. ``args``
    supplies ``epsilon``, ``delta`` and ``ridge``.
    """
    if not len(features_labeled) or not len(features_unlabeled) or classes < 2:
        raise ValueError("Nonempty labeled/unlabeled sets and K>=2 are required")
    budget = JointBudget(float(args.epsilon), float(args.delta), 1)
    bound = math.sqrt(1 - 1 / classes)
    cross = _mean_release(labeled_packets(features_labeled, labels, classes), budget, private)
    cross = cross.reshape(features_labeled.shape[1], classes - 1) * bound
    released_covariance = _mean_release(unlabeled_packets(features_unlabeled), budget, private)
    covariance = trace_one(unpack_symmetric(released_covariance, features_labeled.shape[1]))
    identity = torch.eye(features_labeled.shape[1], dtype=torch.float64, device=covariance.device)
    weight = torch.linalg.solve(covariance + float(args.ridge) * identity, cross)
    weight = weight @ contrasts(classes).to(weight).T
    if not torch.isfinite(weight).all():
        raise FloatingPointError("Nonfinite CrossOnlyDP head")
    return weight


def privacy_report(epsilon, delta, labeled_count, unlabeled_count, classes):
    budget = JointBudget(float(epsilon), float(delta), 1)
    bound = math.sqrt(1 - 1 / classes)
    return {
        "epsilon": epsilon,
        "delta": delta,
        "rho": budget.rho,
        "privacy_unit": "one target training window replacement",
        "labeled_windows": labeled_count,
        "unlabeled_windows": unlabeled_count,
        "label_mean_sensitivity": 2 * bound / labeled_count,
        "unlabeled_mean_sensitivity": 2 / unlabeled_count,
        "composition": "parallel disjoint labeled cross-moment and unlabeled second-moment releases",
        "postprocessing": "PSD trace-one projection followed by ridge solve",
    }


__all__ = [
    "VERSION", "JointBudget", "contrasts", "fit", "labeled_packets",
    "privacy_report", "trace_one", "unlabeled_packets",
]

