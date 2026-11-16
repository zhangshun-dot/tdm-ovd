# TDM — Token Distribution Modulation for Open-Vocabulary Detection in Remote Sensing

Reference implementation of the measurement-driven, training-free prompt-calibration
framework described in the paper *"Token Distribution Modulation: Information-Theoretic
Prompt Optimization for Open-Vocabulary Detection in Remote Sensing"* (JSTARS submission).

## What this repository contains

- `src/` — core scripts implementing the method and the evaluation protocol:
  - `axes_definition.py` — the seven semantic prompt axes and their discrete levels.
  - `ofat_experiment.py` — one-factor-at-a-time (OFAT) measurement of per-axis detection-rate gain.
  - `run_dior_suite.py`, `unified_eval.py` — detector evaluation and 11-point AP / detection-rate computation.
  - `native_tdm_selection.py`, `heldout_probe_v2.py`, `aggregate_heldout_fast.py` —
    prompt selection and disjoint calibration/test (held-out) evaluation.
  - `cross_detector_tdm.py`, `bootstrap_tdm_overhead.py`, `tdm_leave_one_out.py` —
    cross-model transfer, bootstrap confidence intervals, and leave-one-out analysis.
  - `run_dota_heldout.py`, `third_dataset_dota_tdm.py` — independent DOTA-v1.5 validation.
- `results/` — aggregated summary tables (JSON) reported in the paper. These are small
  summary files, not raw detections.

## Data availability

This repository distributes **code and aggregated result tables only**. The image datasets
used in the experiments are third-party benchmarks and are **not** redistributed here:

- DIOR — https://gcheng-nwpu.github.io/Dataset-DIOR/download.html
- DOTA-v1.5 — https://captain-whu.github.io/DOTA/dataset.html
- COCO — https://cocodataset.org/#download
- NWPU VHR-10 — obtainable from the authors of the original benchmark.

Detector checkpoints (GroundingDINO, OWLv2) are loaded from their official releases and
are likewise not redistributed. See `DATA_AVAILABILITY.md` for the full statement.

## Path configuration

The scripts use a placeholder `<REPO_ROOT>` where the original experiment tree root was.
Set this to your own directory, or edit the path constants at the top of each script
(`IMAGE_DIR`, `RESULTS_DIR`, `CACHE_DIR`, etc.) to point to your local dataset and result
locations.

## Requirements

Python 3.9+, `numpy`, `scipy`, `pandas`, `Pillow`, `pycocotools`, `torch`, `transformers`,
and the official GroundingDINO / OWLv2 packages. GPU inference requires CUDA; the
aggregation and significance scripts run on CPU.

## Citation

If you use this code, please cite the corresponding JSTARS paper (see the manuscript).

## License

Released for academic, non-commercial use. See `LICENSE`.
