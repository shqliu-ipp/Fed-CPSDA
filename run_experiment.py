"""Direct entry point for the final three-dataset Fed-CPSDA experiment.

Edit RUN_CONFIG below and run this file directly. No command-line arguments
are required.
"""

from pathlib import Path

from run_a_three_dataset_confirmation import CONFIG as _BASE_CONFIG
from run_a_three_dataset_confirmation import main


RUN_CONFIG = {
    **_BASE_CONFIG,
    "datasets": ["PU", "SEU_BEARING", "WT"],
    "epsilons": [2.0, 4.0, 8.0, 16.0],
    "seeds": [42, 43, 44],
    "noise_repeats": 5,
    "ridge": 0.02,
    "batch_size": 128,
    "device": "cuda:0",
    "phase": "validation_then_test",
    "output_dir": Path(__file__).resolve().parent / "runs" / "fedcpsda_final",
}


if __name__ == "__main__":
    main(RUN_CONFIG)

