# Fed-CPSDA

Reference implementation for **Fed-CPSDA: Privacy-preserving conditional statistics for federated semi-supervised fault diagnosis**.

The repository contains the final CrossOnlyDP statistic-release method used in the paper, the recording-disjoint data preparation protocol, and the three-dataset experiment entry point. Raw datasets, public checkpoints, generated spectrograms, model weights, and experiment results are intentionally excluded from the repository.

## Repository layout

- `fed_cross_only_moments.py`: CrossOnlyDP/A, the main method used in the paper.
- `fed_covariance_borrowing.py`, `fed_conditional_moments.py`, `fed_private_moments.py`: privacy accounting, packet construction, and post-processing utilities.
- `Main.py`, `unified_experiment.py`: public source encoder, feature extraction, training utilities, and evaluation helpers.
- `prepare_pu_dataset.py`, `prepare_gear_datasets.py`: 64 x 64 RGB spectrogram preparation for PU, SEU and WT.
- `experiment_protocol.py`, `gear_experiment_protocol.py`: recording-disjoint and temporally purged target splits.
- `run_experiment.py`: editable direct entry point for the final PU + SEU bearing + WT experiment.
- `run_a_three_dataset_confirmation.py`, `run_three_dataset_experiment.py`: full experiment implementations.

## Installation

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches the target CUDA version when a GPU is used.

## Data paths

The datasets are not distributed with this repository. Set the following environment variables, or edit the corresponding `CONFIG` dictionaries in the preparation scripts:

```text
FEDCPSDA_PU_RAW_ROOT       raw Paderborn bearing dataset
FEDCPSDA_WT_RAW_ROOT       raw WT planetary gearbox dataset
FEDCPSDA_DN_RAW_ROOT       raw SEU gearbox dataset
FEDCPSDA_DN_BEARING_RAW_ROOT  raw SEU bearing dataset, when used
```

The prepared data are written below `DATA/` and are ignored by Git. Each preparation script writes a manifest containing preprocessing parameters and file hashes.

## Running the final experiment

Open `run_experiment.py` and edit `RUN_CONFIG` before running. The most common settings are `datasets`, `epsilons`, `seeds`, `noise_repeats`, `batch_size`, `ridge`, and `device`.

```bash
python prepare_pu_dataset.py
python prepare_gear_datasets.py
python run_experiment.py
```

The default protocol uses the PU, SEU bearing and WT planetary-gear domains, target-window replacement privacy, 64 x 64 RGB spectrograms, and the privacy budgets used in the paper. Outputs are written to `runs/`; this directory is ignored by Git.

For a single GPU, set for example:

```python
"device": "cuda:0",
"batch_size": 128,
```

The implementation does not upload raw samples. The privacy guarantee applies to the released bounded target statistics under the fixed public-source and trusted-coordinator assumptions described in the paper.

