# CLAUDE.md

## Project Overview

**FMP (Fingerspelling Motion Perception)** — a machine learning project for hand gesture and fingerspelling recognition using computer vision. Models classify ASL fingerspelling letters from 21-point MediaPipe hand landmark data. The primary dataset is Fingerspelling5.

## Tech Stack

- **ML**: PyTorch, PyTorch Lightning, PyTorch Geometric
- **Vision**: MediaPipe (hand landmarks), OpenCV, Kornia (GPU augmentations), Torchvision
- **Experiment Tracking**: MLFlow (primary), AIM, Optuna (HPO)
- **Data**: LMDB caching, pandas, numpy
- **Config/CLI**: LightningCLI + jsonargparse + YAML
- **Dev**: black, isort, mypy, pytest

## Package Installation

The `fmp` package is installed in editable mode via `pyproject.toml`:
```bash
pip install -e .
```

## Common Commands

### Training
```bash
# Current primary training config (mixed multi-source dataset)
python scripts/train_mlflow_basic.py fit --config configs/fingerspelling5_singlehands/train_with_mlflow_mixed.yaml

# Single-source variant
python scripts/train_mlflow_basic.py fit --config configs/fingerspelling5_singlehands/train_with_mlflow.yaml
```

### SWA Continuation
```bash
# Load weights from existing checkpoint, start fresh optimizer, run SWA phase
python scripts/train_mlflow_swa.py fit \
  --config configs/fingerspelling5_singlehands/train_with_mlflow_swa.yaml \
  --model_init_ckpt checkpoints/<experiment>/<run_id>/<checkpoint>.ckpt

# Sweep swa_lr via CLI override (keep tag in sync for MLFlow filtering)
python scripts/train_mlflow_swa.py fit \
  --config configs/fingerspelling5_singlehands/train_with_mlflow_swa.yaml \
  --model_init_ckpt checkpoints/<experiment>/<run_id>/<checkpoint>.ckpt \
  --lr_scheduler.init_args.swa_lr=1e-5 \
  --trainer.logger.init_args.tags.swa_lr="1e-5"
```

### Prediction & Testing
```bash
python scripts/train_mlflow_basic.py predict \
  --config lightning_logs/version_X/config.yaml \
  --ckpt_path lightning_logs/version_X/checkpoints/<checkpoint>.ckpt

python scripts/compute_fingerspelling5_metrics.py predict \
  --config configs/fingerspelling5_singlehands/metrics_scaled.yaml
```

### Hyperparameter Optimization
```bash
python scripts/optuna_hpo.py
```

### MLFlow UI

Experiment tracking is handled by the dedicated `mlflow-infra` server (a
Docker container on this machine — see `~/projects/mlflow-infra/`). The UI
is always up at **http://localhost:5000**; no command needed to launch it.
Verify the server is running:
```bash
docker ps --filter name=mlflow-server
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:5000   # expect 200
```

To browse the pre-migration archive (runs from before 2026-04-11, most with
broken artifact links — only the SWA base run has on-disk artifacts):
```bash
mlflow ui \
  --backend-store-uri sqlite:////home/micha/mlflow-archive/2026-04-11-pre-server/mlruns.db \
  --default-artifact-root file:///home/micha/mlflow-archive/2026-04-11-pre-server/mlruns \
  --port 5050
```

### Tests
```bash
pytest tests/
```

### Data Preparation (Dummy Data)
```bash
python scripts/create_fingerspelling5_dummy.py \
  --dir-dest=data/fingerspelling5 \
  --dataset-name=fingerspelling5_dummy \
  --num-persons=3 --num-samples=4

python scripts/create_fingerspelling5_splits.py \
  --dataset-dir=data/fingerspelling5/fingerspelling5_dummy

python scripts/create_fingerspelling5_dummy_dataquality.py \
  --dataset-dir=data/fingerspelling5/fingerspelling5_dummy
```

## Repository Structure

```
src/fmp/                    # Main package
  datasets/
    fingerspelling5/        # Core dataset (fingerspelling5.py, fingerspelling5_lit.py)
    multi_source/           # Multi-dataset training
    sphere/                 # Sphere-based experiments
  models/
    lit_resnet.py           # ResNetClassifier (primary model, Lightning wrapper)
    lit_mlp.py              # MLP classifier
    multi_source.py         # Multi-source training module
  lit_tools/callbacks/
    mlflow/                 # MLFlow config + checkpoint callbacks
    aim/                    # AIM experiment tracking
  transforms.py             # Custom augmentation transforms
  lr_schedulers.py          # Learning rate schedulers

scripts/                    # Entry points
  train_mlflow_basic.py     # MAIN entry point
  train_mlflow_swa.py       # SWA continuation entry point (SWALightningCLI + --model_init_ckpt)
  train_fingerspelling5_litcli.py
  optuna_hpo.py
  create_fingerspelling5_*.py
  compute_fingerspelling5_metrics.py

configs/
  fingerspelling5_singlehands/  # Main experiment configs
  sphere/                       # Sphere experiment configs
  examples/

tests/fmp/datasets/fingerspelling5/metrics/  # Unit tests

data/                       # Datasets (mounted from ~/data in devcontainer)
checkpoints/                # Saved model checkpoints (local — MLflow artifacts live in the mlflow-infra server's Docker volume)
config_logs/                # MLFlowConfigCallback local config cache, keyed by run_id (used for test/predict layering)
lightning_logs/             # PyTorch Lightning logs
```

## LightningCLI & Experiment Management

`scripts/train_mlflow_basic.py` is the canonical entry point. It wraps `LightningCLI` with a custom `save_config_callback`:

```python
LightningCLI(
    save_config_callback=MLFlowConfigCallback,
    save_config_kwargs={"logdir": "./config_logs", "artifact_path": "configs", ...},
)
```

`torch.set_float32_matmul_precision("high")` is set globally in this script for Ampere GPU performance.

### LightningCLI Subcommands

The same script covers all stages — the subcommand controls the mode.

**Training:**
```bash
python scripts/train_mlflow_basic.py fit --config configs/.../train_with_mlflow.yaml
```

**Test / Predict — layered configs:**
For test and predict, configs are layered: the training config saved by MLFlow is provided first as the base, then a stage-specific config is provided second to override only what needs to change (data splits, checkpoint path, transforms, etc.):

```bash
python scripts/train_mlflow_basic.py test \
  --config config_logs/<mlflow_run_id>/config.yaml \
  --config configs/.../test_config.yaml \
  --ckpt_path checkpoints/<checkpoint>.ckpt

python scripts/train_mlflow_basic.py predict \
  --config config_logs/<mlflow_run_id>/config.yaml \
  --config configs/.../predict_config.yaml \
  --ckpt_path checkpoints/<checkpoint>.ckpt
```

LightningCLI merges multiple `--config` files left-to-right, so keys in the second file override the first. This keeps test/predict configs minimal — they only specify what differs from training.

Any value can also be overridden directly on the CLI:
```bash
python scripts/train_mlflow_basic.py fit --config train_with_mlflow.yaml \
  --trainer.max_epochs=50 --data.batch_size=64
```

### YAML Config Format

Configs use jsonargparse's `class_path` / `init_args` format, which allows full Python class instantiation from YAML. Every component — model, datamodule, optimizer, scheduler, callbacks, logger, transforms — is specified this way:

```yaml
model:
  class_path: fmp.models.ResNetClassifier
  init_args:
    model:
      class_path: fmp.models.ResNet18
      init_args:
        num_classes: 24
```

This means the YAML file is a complete, reproducible description of a run. LightningCLI resolves and instantiates all components before passing them to the `Trainer`.

### SWALightningCLI (`train_mlflow_swa.py`)

`SWALightningCLI` is a `LightningCLI` subclass used exclusively for SWA continuation runs. It adds one extra argument:

- `--model_init_ckpt <path>` — loads `state_dict` from a checkpoint into the already-instantiated model in `before_fit`, **without** restoring optimizer or scheduler state.

This is different from `--ckpt_path`, which resumes full training state (optimizer moments, scheduler step, epoch counter). Use `--model_init_ckpt` when you want "old weights, fresh optimizer trajectory" for the SWA phase.

`ResNetClassifier` uses `save_hyperparameters(ignore=["model"])`, so `load_from_checkpoint` cannot reconstruct the model. The `before_fit` state-dict approach is correct: LightningCLI instantiates the model from the YAML config first, then weights are overwritten before training starts.

### MLFlowConfigCallback

`MLFlowConfigCallback` (extends `SaveConfigCallback`) runs at the start of `fit`/`test` and:

1. **Saves config locally** to `./config_logs/<mlflow_run_id>/config.yaml` — keyed by run ID for traceability
2. **Logs config as MLFlow artifact** under the `configs/` artifact path in the run
3. **Sets MLFlow summary tags** (`config.model`, `config.data`, `config.optimizer`, etc.) for filtering runs in the MLFlow UI

Config hashing (`log_config_hash`) is disabled by default since the MLFlow run ID already provides uniqueness.

The mixed config also uses `MLFlowSystemMetricsLogger` (a custom subclass of `MLFlowLogger`) instead of the plain logger — it adds CPU/GPU/memory tracking and supports `synchronous=true` for real-time UI updates.

The mixed config uses `fmp.lr_schedulers.cosine_annealing_warmup` (with `warmup_iters` and `warmup_factor`) rather than plain `CosineAnnealingLR`, which helps stabilise early training when mixing sources with different statistics.

### MLFlow Callbacks Overview

| Callback | Role |
|----------|------|
| `MLFlowConfigCallback` | Saves + uploads full YAML config; sets summary tags |
| `MLFlowModelCheckpoint` | Top-K checkpoint saving; logs best checkpoints as MLFlow artifacts |
| `MLFlowImageLogger` | Logs training image batches as artifacts (for augmentation debugging) |
| `MLFlowSystemMetricsLogger` | System/GPU metrics logging |

## MLflow Tracking Server

Since 2026-04-11 this project logs to a dedicated MLflow tracking server
(`mlflow-infra`) running as a Docker container on the host. The server lives
at `~/projects/mlflow-infra/` — a sibling project-agnostic directory that
can be reused by any other project on this machine. See
`~/projects/mlflow-infra/README.md` for the full ops manual and
`~/projects/mlflow-infra/CLAUDE.md` for a quick reference.

### Client wiring

`MLFLOW_TRACKING_URI` is exported in `~/.bashrc` and `~/.profile`, so
host-side training just works:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000                  # host
export MLFLOW_TRACKING_URI=http://mlflow-server:5000              # inside a container on mlflow-net
```

Training configs (`train_with_mlflow_mixed.yaml`,
`train_with_mlflow_swa.yaml`, `test_with_mlflow.yaml`) intentionally **omit
both `tracking_uri` and `artifact_location`** from the logger init args —
the env var drives tracking and the server owns artifact storage via
`--serve-artifacts`. Do not re-add `tracking_uri: "sqlite:///..."` or
`artifact_location: "file:..."` to configs; doing so re-introduces the
devcontainer-vs-host path-pollution bug that motivated the migration in
the first place.

A correctly-logged run has `artifact_uri` of the form
`mlflow-artifacts:/<exp_id>/<run_id>/artifacts` — **never** a filesystem
path. Artifacts physically live inside the Docker named volume
`mlflow-infra_mlflow-artifacts`, reachable only via the server API or
`docker exec mlflow-server`. The project-local `./mlruns/` directory
should not exist; if you see it reappear, a client is falling back to
the local file store and the config is wrong.

### Downloading artifacts

Artifacts (checkpoints, configs, etc.) live inside the Docker named volume
and are **not** directly accessible on the host filesystem. Use the MLflow
CLI to download them:
```bash
# Download a specific artifact (e.g., a checkpoint)
mlflow artifacts download \
  -r <run_id> \
  -a checkpoints/<checkpoint>.ckpt \
  -d ./checkpoints

# Download all artifacts for a run
mlflow artifacts download -r <run_id> -d ./artifacts

# Path mapping: mlflow-artifacts:/<exp_id>/<run_id>/artifacts/...
# maps to /mlflow/mlartifacts/<exp_id>/<run_id>/artifacts/... inside the container
```

### Pre-server archive

The pre-migration SQLite store + one high-value run's artifacts are
archived at `~/mlflow-archive/2026-04-11-pre-server/`. The preserved run
`c0b2e0ebd99243cc80e2d44c83a8c602` is the **ResNet18 mixed-dataset base
checkpoint for SWA continuation** — its five top-K checkpoints and two
configs are the only artifacts still on disk. See the archive's own
`README.md` for the full inventory and how to open the archive in a
throwaway MLflow UI. The canonical SWA base checkpoint path (referenced
by the header of `train_with_mlflow_swa.yaml`) is:
```
~/mlflow-archive/2026-04-11-pre-server/mlruns/9/c0b2e0ebd99243cc80e2d44c83a8c602/artifacts/checkpoints/epoch=18-step=15998.ckpt
```

### Devcontainer wiring

`.devcontainer/devcontainer.json` attaches the devcontainer to `mlflow-net`
via `runArgs` and sets `MLFLOW_TRACKING_URI=http://mlflow-server:5000`
via `containerEnv`, so training from inside the devcontainer reaches the
server the same way host-side training does. The `mlflow-net` Docker
network must already exist before the devcontainer starts (it's created
once by `~/projects/mlflow-infra` via `docker network create mlflow-net`).
Inside the container, `localhost` is the container itself — the
service-name form is mandatory.

## Image-Based Pipeline

The project supports two data modalities. The image pipeline (`Fingerspelling5ImageDataModule`) is the primary one for the ResNet18 classifier; the landmark pipeline (`Fingerspelling5LandmarkDataModule`) works with raw 21-point MediaPipe coordinates via PyTorch Geometric.

### Augmentation Stages

Augmentations are split across three stages to balance flexibility and GPU throughput:

**1. CPU transforms** (`train_transforms` in config, applied in the DataLoader worker):
- `FXAALite` — lightweight anti-aliasing (Laplacian edge detection + Gaussian blur on detected edges)
- `ImageSharpening` — Laplacian + Sobel-weighted unsharp mask
- `RandomErasing` — occlusion augmentation
- `ScaleJitter` — uniform random scale; `ScaleJitterXY` scales H/W independently
- `BackgroundImage` — composites the hand crop onto a random LSUN scene (bedroom, kitchen, etc.)
- `CropOrPad` — center-crops or zero-pads to a fixed spatial size
- `PadToSize` — padding-only variant (raises if image is already larger than target)

**2. Kornia train transforms** (`kornia_train_transforms`, run on CPU before GPU transfer):
- `RandomElasticTransform` — elastic deformation

**3. GPU batch transforms** (`gpu_batch_transforms`, applied in `on_after_batch_transfer`):
- Spatial: `RandomHorizontalFlip`, `RandomAffine` (rotation ±30°, translate 10%, shear 10°), `RandomPerspective`
- Color: `ColorJiggle`, `RandomEqualize`, `RandomSolarize`, `RandomPosterize`
- Noise/blur: `RandomMotionBlur`, `RandomGaussianNoise`
- `Normalize` (always last — dataset-specific mean/std)

The model (`ResNetClassifier`) applies these GPU transforms inside `on_after_batch_transfer`, keeping the DataModule and model configs independent.

### Multi-Source Dataset & Per-Source Validation

Combining multiple image sources (Fingerspelling5, Mendeley, Kaggle, Kaggle2, Roboflow) was the key driver for reaching good model performance. Each source has different image statistics, backgrounds, and quality, so several things need to be handled explicitly:

**Per-source validation dataloaders** — `additional_valid_images_files` registers each source as a named validation dataloader. During training, `ResNetClassifier` logs accuracy separately for each one (`acc/valid_mendeley`, `acc/valid_kaggle`, etc.) every validation epoch.

**Per-source normalization** — `additional_valid_image_transform` defines a Kornia `Normalize` (and optionally `PadTo`) for each source. In `train_with_mlflow_mixed.yaml` all sources share a single unified mean/std computed over the combined dataset. In the single-source config each external source keeps its own statistics.

**Checkpoint selection** — `MLFlowModelCheckpoint` saves top-K checkpoints (30 in the mixed config) monitored on `acc/valid_overall`. Because per-source accuracy curves diverge, the MLFlow UI is used after training to inspect per-source accuracy over epochs and select the checkpoint that gives the best trade-off across sources — not necessarily the one with the highest overall score.

**Unified mean/std for the mixed dataset** — since all sources are composited onto LSUN backgrounds using `BackgroundImage` with the same augmentation pipeline, a single normalization applies to all sources in `train_with_mlflow_mixed.yaml`:
```
mean: [0.5342, 0.4953, 0.4969]
std:  [0.2038, 0.2137, 0.2114]
```

### Background Data (LSUN)

`BackgroundImage` uses `torchvision.datasets.LSUN` with six indoor scene categories. The LSUN data root must be accessible at the path configured in `train_transforms` (default: `../../data/lsun`).

### MLFlow Image Logging

`MLFlowImageLogger` (commented out by default in the training config) can log batches of training images as MLFlow artifacts for augmentation debugging. Enable it and set `limit_train_batches`/`limit_val_batches` to run a fast debug pass.

## Configuration System

Configs use YAML with [jsonargparse](https://jsonargparse.readthedocs.io/) via LightningCLI. Key configs:

- `configs/fingerspelling5_singlehands/train_with_mlflow_mixed.yaml` — **current primary** (35 epochs, multi-source dataset, cosine warmup LR, `MLFlowSystemMetricsLogger`, 5 validation sources, top-30 checkpoints)
- `configs/fingerspelling5_singlehands/train_with_mlflow.yaml` — single-source variant (80 epochs, 4 validation sources, per-source normalization stats, `FXAALite`+`ImageSharpening` in CPU transforms)
- `configs/fingerspelling5_singlehands/train_with_mlflow_swa.yaml` — SWA continuation (15 epochs, `WeightAveraging` + `SWALR`, fresh AdamW, sweep `swa_lr`; used with `train_mlflow_swa.py` and `--model_init_ckpt`)
- `configs/sphere/training.yaml` — sphere experiment training

## Docker & DevContainer

- **Dockerfile**: Two targets — `fmp` (production) and `devcontainer` (dev with Node.js + Claude Code CLI)
- **Base image**: `pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime`
- **DevContainer**: GPU passthrough (`--gpus=all`), 8 GB shared memory, port 8050 forwarded for Dash UI
- **Data mount**: `~/data` → `/mnt/data` inside container
