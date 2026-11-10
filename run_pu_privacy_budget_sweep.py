"""PU-only privacy-utility validation sweep on five pre-specified directions."""

from pathlib import Path

import run_pu_covariance_borrowing as experiment


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RETAINED_TASKS = [
    "PU_N15_M07_F10_to_N09_M07_F10",
    "PU_N15_M07_F10_to_N15_M01_F10",
    "PU_N15_M01_F10_to_N15_M07_F10",
    "PU_N15_M07_F10_to_N15_M07_F04",
    "PU_N15_M07_F04_to_N15_M07_F10",
]
EPSILONS = [2.0, 4.0, 8.0, 16.0]


CONFIG = {
    "tasks": RETAINED_TASKS,
    "epsilons": EPSILONS,
    "seeds": [42, 43, 44],
    "noise_repeats": 5,
    "delta": 1e-5,
    "ridge": 0.02,
    "borrowing_cap": 0.95,
    "device": "cuda:0",
    "evaluate_test": False,
}


def main(config=None):
    settings = {**CONFIG, **(config or {})}
    tasks = list(settings["tasks"])
    if tasks != RETAINED_TASKS:
        raise ValueError("This pre-registered sweep only permits the five retained PU directions")
    for epsilon in settings["epsilons"]:
        if epsilon not in EPSILONS:
            raise ValueError("Use one of the pre-registered privacy budgets: 2, 4, 8, 16")
        output_dir = RUNS / f"pu_covariance_borrowing_eps{int(epsilon)}_five_dirs"
        experiment.main({
            "output_dir": output_dir,
            "datasets": ["PU"],
            "task_names": tasks,
            "seeds": settings["seeds"],
            "noise_repeats": settings["noise_repeats"],
            "epsilon": float(epsilon),
            "delta": settings["delta"],
            "ridge": settings["ridge"],
            "borrowing_cap": settings["borrowing_cap"],
            "device": settings["device"],
            "evaluate_test": settings["evaluate_test"],
        })


if __name__ == "__main__":
    main()
