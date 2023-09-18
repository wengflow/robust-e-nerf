# Robust *e*-NeRF

[![Project Page](https://img.shields.io/badge/Project_Page-black?style=for-the-badge
)](https://wengflow.github.io/robust-e-nerf) &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-black?style=for-the-badge)](https://arxiv.org/abs/2309.08596) &nbsp; [![Simulator](https://img.shields.io/badge/Simulator-black?style=for-the-badge)](https://github.com/wengflow/rpg_esim) &nbsp; [![Dataset](https://img.shields.io/badge/Dataset-black?style=for-the-badge)](https://huggingface.co/datasets/wengflow/robust-e-nerf)

<p align="center">
   <img src="assets/comparison.gif" alt="Comparison" width="58%"/>
   <img src="assets/results%20(square).gif" alt="Results" width="38%"/>
</p>

This repository contains the source code for the ICCV 2023 paper — [Robust *e*-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion](https://arxiv.org/abs/2309.08596), which is mainly built on the PyTorch Lightning framework. Robust *e*-NeRF is a novel method to directly and robustly reconstruct NeRFs from moving event cameras under various real-world conditions, especially from sparse and noisy events generated under non-uniform motion.

If you find Robust *e*-NeRF useful for your work, please consider citing:
```bibtex
@inproceedings{low2023_robust-e-nerf,
  title = {Robust e-NeRF: NeRF from Sparse & Noisy Events under Non-Uniform Motion},
  author = {Low, Weng Fei and Lee, Gim Hee},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2023}
}
```

## Installation

We recommend using [Conda](https://conda.io/) to set up an environment with the appropriate dependencies for running Robust *e*-NeRF, as follows:

```bash
git clone https://github.com/wengflow/robust-e-nerf.git
cd robust-e-nerf
conda env create -f environment.yml
```

If a manual installation is preferred, the list of dependencies can be found in `environment.yml`.

## Dataset Setup

### Robust *e*-NeRF Synthetic Dataset

Our synthetic experiments are performed on a set of sequences simulated using an [improved ESIM](https://github.com/wengflow/rpg_esim) event camera simulator with different camera configurations on NeRF Realistic Synthetic $360\degree$ scenes. In our [Robust *e*-NeRF Synthetic Event Dataset](https://huggingface.co/datasets/wengflow/robust-e-nerf), we provide the simulated sequences used to study the collective effect of camera speed profile, contrast threshold variation and refractory period. Minor modifications to the ESIM configuration files provided in the dataset enables the simulation of other sequences used in our synthetic experiments.

To run Robust *e*-NeRF on our synthetic dataset:

1. Setup the dataset according to the [official instructions](https://huggingface.co/datasets/wengflow/robust-e-nerf#setup)
2. Preprocess each sequence in the raw dataset with:
   ```bash
   python scripts/preprocess_esim.py <sequence_path>/esim.conf <sequence_path>/esim.bag <sequence_path>
   ```

### TUM-VIE

Our qualitative real experiments are performed on the `mocap-1d-trans`, `mocap-desk2` and `office_maze` sequences of the [TUM-VIE dataset](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset), which are setup as follows:

1. Download the following raw data for each sequence and calibration files into a common folder:
   - `<sequence_name>-events_left.h5`
   - `<sequence_name>-vi_gt_data.tar.gz`
   - `camera-calibration{A, B}.json`
   - `mocap-imu-calibration{A, B}.json`
2. Uncompress all `<sequence_name>-vi_gt_data.tar.gz` and then remove them
3. Preprocess each sequence in the raw dataset with:
   ```bash
   python scripts/tum_vie_to_esim.py <sequence_name> <raw_dataset_path> <preprocessed_dataset_path>
   ```
   Note that we trim the end of the `offize-maze` sequence with the additional argument of `--end_timestamp 20778222508`

## Usage

Train, validate or test Robust *e*-NeRF with:
```bash
python scripts/run.py {train, val, test} <config_file_path>
```

## Configuration

In the `configs/` folder, we provide two sets of configuration files:

1. `{train, test}/synthetic.yaml`
2. `{train, test}/{mocap-1d-trans, mocap-desk2, office_maze}.yaml`

used to train or test Robust *e*-NeRF for the synthetic and real experiments, respectively.

The specific experimental setting described in the configuration files are given as follows:

| Configuration File | Experiment | Sequence | Opt. $C_p$ | Opt. $\tau$ | $\ell_{grad}$ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `synthetic.yaml` | Synthetic | `ficus` under the easy/default setting | ✗ | ✗ | ✗ |
| `<sequence_name>.yaml` | Real | `<sequence_name>` | ✓ | ✓ | ✓ |

You should modify the following parameters in the given configuration files to reflect the correct or preferred paths:
1. `data.dataset_directory`: Path to the preprocessed sequence
2. `model.checkpoint_filepath`: Path to the pretrained model
3. `logger.save_dir`: Preferred path to the logs

To reproduce our synthetic experiment results under any specific setting, as reported in the paper, you should modify `{train, test}/synthetic.yaml` as follows:
| Experimental Setting | Parameter(s) To Be Modified |
| :--- | :--- |
| Sequence | `data.dataset_directory` |
| Opt. $C_p$ | `model.contrast_threshold.freeze` |
| Opt. $\tau$  | `model.refractory_period.freeze` |
| w/ or w/o $\ell_{grad}$ | `loss.weight.log_intensity_grad` |

You should also modify `model.checkpoint_filepath` and `logger.name` accordingly.

Note that in our synthetic experiments, when the contrast threshold $C_p$ and/or refractory period $\tau$ are jointly optimized, we poorly initialize them, by manually overriding the calibrated values, to assess the robustness of the joint optimization.

Please refer to the [Robust *e*-NeRF paper](https://arxiv.org/abs/2309.08596) for more details.
