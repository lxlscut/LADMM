# Large-Scale ADMM Hyperspectral Clustering

This repository provides a scalable ADMM-based framework for hyperspectral image clustering. The workflow follows a teacher–student distillation pipeline, includes training scripts for the Houston, Trento, and Urban benchmarks, and ships utilities for aggregating experiment results.

## Draft Paper

The accompanying manuscript is currently titled *Contrastive Knowledge Distillation of Unfolded ADMM Self-Representation for Scalable Hyperspectral Image Clustering*. The draft is still under internal advisor review and has not been submitted or released publicly yet.

## Repository Layout

- `Houston.py` / `Trento.py` / `Urban.py`: dataset-specific entry points that parse CLI arguments and launch the training routine.
- `houston.sh` / `Trento.sh` / `Urban.sh`: batch scripts that iterate over predefined random seeds and store the console logs of each run.
- `TOOLS/`
  - `train.py`: high-level training orchestration that wires the data loader, teacher stage, and student stage.
  - `Teacher.py`: unfolded ADMM network that learns self-representation matrices in stage 1.
  - `Student.py`: convolutional classifier that consumes the teacher outputs in stage 2.
  - `get_data.py`: utilities for loading hyperspectral cubes, cropping, patch extraction, and PCA preprocessing.
  - `draw_result_Plt.py`: helper that renders predicted and ground-truth maps as colored PDFs.
  - Additional helpers for loss definitions, sampling, evaluation, and logging.
- `Get_average_result.py`: scans result folders to compute mean ACC/NMI/Kappa, class-wise accuracy, and runtime.
- `checkpoint/`: stores teacher and student checkpoints that are produced during training.
- `result/`: default location where batch scripts dump run logs.

## Environment Setup

Create a fresh Conda environment (Python 3.10+ is recommended):

```bash
conda create -n admm-hsi python=3.10
conda activate admm-hsi
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy scikit-learn matplotlib tensorboard torch-scatter
pip install gdal  # required when reading .tif datasets
```

> If you only work with `.mat` datasets you can skip GDAL; if `torch_scatter` is not needed in your experiments you may also omit it.

## Key Dependencies

```text
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==1.24.3
scipy==1.11.3
scikit-learn==1.2.2
pandas==2.1.4
matplotlib==3.7.2
pillow==11.1.0
tensorboard==2.12.1
tqdm==4.66.1
gdal==3.6.2
umap-learn==0.5.7
visdom==0.2.4
kornia==0.7.2
```

For a fully reproducible snapshot you can activate the `spectralnet_backup` environment and run `pip freeze > requirements_full.txt` to capture every installed package.

## Data Preparation

Update the dataset paths inside the entry scripts to match your local layout. The default expectation is:

```
/home/xianlli/dataset/HSI/
    ├── Houston/Houston_corrected.mat, Houston_gt.mat
    ├── trento/Trento.mat, Trento_gt.mat
    └── urban/Urban_corrected.mat, Urban_gt.mat
```

`TOOLS/get_data.py` crops the cube to `[:, 100:400, :]` and performs PCA according to `--n_input`. Adjust the script or CLI flags if you need different spatial extents or channel counts.

## Training Pipeline

1. **Stage 1 – Teacher**: `Teacher.train_stage_1` unfolds ADMM to learn the self-representation matrix `C` and stores weights under `checkpoint/teacher_checkpoint.pt`.
2. **Stage 2 – Student**: `Student.train_stage_2` loads the teacher, builds similarity statistics, and optimizes the lightweight classifier.
3. **Visualization**: the helper saves predicted maps to `code/My_TGRS/result/Ours/<Dataset>/`.

## Single-Run Example

```bash
python Urban.py --device cuda:0
```

Useful flags:

- `--dataset`: dataset name (`Houston`, `Trento`, `Urban`).
- `--n_input`: number of spectral bands after optional PCA.
- `--patch_size`: spatial window size.
- `--num_layer`: number of unfolded ADMM iterations.
- `--alpha`, `--beta`, `--gamma`, `--theta`, `--eta`, `--lamda`: loss weights across stages.
- `--device`: computation device, e.g. `cuda:0` or `cpu`.

Each entry script calls `setup_seed()` before training to keep runs reproducible.

## Batch Scripts

`Urban.sh`, `Trento.sh`, and `houston.sh` iterate through a predefined seed list and redirect console output to `result/<dataset>_2025/result_<seed>.txt`. Extend these scripts if you need grid searches over additional hyperparameters.

## Result Aggregation

Use `Get_average_result.py` to summarise batched runs:

```bash
python Get_average_result.py
```

The script reads from `result/Urban_cos/` by default; adjust `output_dir` to point at the folder you want to aggregate. It reports:

- Mean ACC/NMI/Kappa scores
- Mean per-class accuracy (`ca`)
- Mean elapsed time

The summary is written to `final.txt` inside the chosen directory.

## Logs and Checkpoints

- `runs/`: TensorBoard events written by `TensorBoardLogger`.
- `checkpoint/`: teacher (`teacher_checkpoint.pt`) and student (`model_checkpoint.pt`) weights.

## Customisation Tips

- **Datasets**: add new cases in `TOOLS/train.py` or refactor the loader configuration if you expect to swap datasets frequently.
- **Schedulers**: `Student.train_stage_2` currently uses `ReduceLROnPlateau`; tweak or replace the scheduler as needed.
- **Housekeeping**: functions in `TOOLS/loss_function.py` that are annotated with `# NOTE: currently unused` are safe to delete once you confirm no external dependency relies on them.

Feel free to build on top of these scripts or file issues if you run into problems.
