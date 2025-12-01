## Diffusion Model Project

This repository is a **template** for implementing a diffusion model in a clean, modular, and production-ready way.  
There is **no implementation code yet**; instead, this document explains **what you should program in each file**.

The stack is designed around **Python** and **PyTorch**, with clear separation between configuration, data, models, training, sampling, and evaluation.

---

## High-Level Architecture

- **`src/diffusion_model`**: Python package containing all core logic.
- **`configs/`**: YAML configuration files for experiments and training runs.
- **`scripts/`**: Command-line entry points for training, sampling, and evaluation.
- **`tests/`**: Unit and integration tests for the main components.

You will progressively implement each module so that:

- Models are **decoupled** from data and training loops.
- All hyperparameters and paths are controlled by **config files**.
- Training and sampling can be invoked via **CLI scripts**.

---

## Environment & Tooling

### `pyproject.toml`

What you should implement:

- Declare the project metadata (name, version, description, authors).
- Configure the build backend (e.g. `hatchling` or `setuptools`).
- Add a `project.dependencies` section including (at minimum):
  - `torch` (and `torchvision` if you use image datasets),
  - `PyYAML` for config loading,
  - `numpy`,
  - `tqdm` for progress bars,
  - `wandb` or `tensorboard` (optional, for logging),
  - `pytest` and `pytest-cov` for testing.
- Define optional extras (e.g. `dev`, `train`, `docs`) if desired.
- Optionally define console scripts entry points mapping to functions in `scripts/`.

### `.gitignore`

What you should include:

- Ignore typical Python artifacts: `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, `.venv/`, `env/`, `build/`, `dist/`.
- Ignore experiment outputs and logs: `outputs/`, `runs/`, `checkpoints/`, `logs/`.
- Ignore editor and OS files: `.idea/`, `.vscode/`, `*.swp`, `.DS_Store`.

---

## Package Layout (`src/diffusion_model`)

### `src/diffusion_model/__init__.py`

What you should implement:

- Keep it minimal; optionally expose high-level APIs:
  - Factory functions to build models and schedulers from config.
  - Version string (e.g. `__version__`).
- Avoid heavy imports here to keep import time fast.

---

### Configuration

#### `src/diffusion_model/config.py`

What you should implement:

- A configuration system that:
  - Loads YAML files from `configs/`.
  - Merges base configs with experiment-specific overrides.
  - Supports command-line overrides (e.g. `--learning_rate 1e-4`).
- Strongly-typed config objects or dataclasses for:
  - **ModelConfig** (e.g. channels, depth, attention, etc.).
  - **DiffusionConfig** (timesteps, noise schedule, beta range).
  - **TrainingConfig** (batch size, epochs, optimizer, lr schedule).
  - **DataConfig** (dataset path, resolution, augmentations).
  - **LoggingConfig** (log directory, frequency, wandb project, etc.).
- Validation logic (e.g. assert valid ranges, required fields).

---

### Data

#### `src/diffusion_model/data/dataset.py`

What you should implement:

- PyTorch `Dataset` classes for your target data, for example:
  - An image dataset that loads images from a directory or a known dataset (CIFAR-10, ImageNet subset, etc.).
- Dataset responsibilities:
  - Load raw data (paths, labels if needed).
  - Apply preprocessing and transforms.
  - Return tensors in a shape appropriate for your model (e.g. `C x H x W`).
- A factory function that, given a `DataConfig`, returns:
  - Training dataset,
  - Validation dataset (optional),
  - Test dataset (optional).

#### `src/diffusion_model/data/transforms.py`

What you should implement:

- Composable transform pipelines for:
  - Training (random crops, flips, normalization, resizes).
  - Evaluation / sampling (deterministic resizing and normalization).
- Torchvision-style transforms or your own small wrappers.
- A helper to build transforms from config (e.g. specify resize size, normalization mean/std in YAML).

---

### Models

#### `src/diffusion_model/models/unet.py`

What you should implement:

- A U-Net style architecture tailored for diffusion models:
  - Downsampling and upsampling blocks.
  - Residual blocks with group/instance/batch normalization.
  - Optional attention blocks at selected resolutions.
  - Time embedding injected into intermediate layers.
- Flexibility driven by `ModelConfig`:
  - Number of channels at each resolution.
  - Number of residual blocks per stage.
  - Whether to use attention & where.
- A clear forward interface, for example:
  - Inputs: noisy sample, time index, optional conditioning (class labels, text embeddings, etc.).
  - Output: predicted noise (or directly the denoised sample, depending on choice).

#### `src/diffusion_model/models/autoencoder.py` (optional for latent diffusion)

What you should implement:

- Optional autoencoder / VAE module if you choose a **latent diffusion** setup:
  - Encoder network that maps input images to a latent space.
  - Decoder that reconstructs images from latents.
- Loss functions for reconstruction (e.g. MSE, perceptual loss).
- An interface for:
  - Encoding real data into latents for diffusion.
  - Decoding latents back to the image space after sampling.

---

### Schedulers (Diffusion Process)

#### `src/diffusion_model/schedulers/noise_scheduler.py`

What you should implement:

- A class representing the diffusion process (forward and reverse):
  - Definition of betas or alphas schedule over timesteps.
  - Methods to compute:
    - Variances, alphas-cumprod, etc.
    - Noising functions \(q(x_t \mid x_0)\).
    - Reverse step parameters for sampling.
- Support for different schedule types:
  - Linear, cosine, or others (selectable from config).
- Methods that the training loop and sampler will use, for example:
  - Sample random timesteps for training.
  - Add noise to clean samples at a given timestep.
  - Compute the posterior mean and variance for the reverse step.

---

### Training

#### `src/diffusion_model/training/train_loop.py`

What you should implement:

- A high-level training loop that orchestrates:
  - Model, optimizer, scheduler, data loaders.
  - Forward pass, loss computation, and backward pass.
  - Gradient clipping, mixed precision (optional), and logging.
- Core responsibilities:
  - Iterate over epochs and batches.
  - Sample random timesteps and noisy inputs.
  - Call the diffusion model to predict noise / denoised sample.
  - Compute the diffusion loss (e.g. MSE between predicted and true noise).
  - Log metrics and save checkpoints periodically.
- Hooks or callbacks for:
  - Validation / sampling at intervals.
  - Early stopping (optional).
  - Learning rate scheduling.

#### `src/diffusion_model/training/optimizer.py`

What you should implement:

- Utility functions to construct:
  - Optimizers (e.g. Adam, AdamW) from `TrainingConfig`.
  - Learning rate schedulers (cosine decay, step LR, etc.).
- Any parameter grouping logic (e.g. weight decay exclusions for bias and normalization layers).

---

### Sampling

#### `src/diffusion_model/sampling/sampler.py`

What you should implement:

- Sampling utilities that use the trained model and scheduler to:
  - Start from pure noise and iteratively denoise to generate samples.
  - Support multiple sampling methods (DDPM, DDIM-like, or other variants).
- A clear interface that can be used both from:
  - Scripts (e.g. `scripts/sample.py`).
  - Evaluation code.
- Options controlled by config:
  - Number of sampling steps.
  - Guidance scale (for classifier-free guidance, if implemented).
  - Batch size, output directory, and sample resolution.

---

### Utilities

#### `src/diffusion_model/utils/logging.py`

What you should implement:

- High-level logging utilities for:
  - Console logging (Python `logging`).
  - Experiment tracking (e.g. WandB or TensorBoard).
- Typical functions:
  - Initialize logger(s) from config.
  - Log scalar metrics, images, and model graphs where applicable.
  - Save experiment configuration and run metadata.

#### `src/diffusion_model/utils/checkpoint.py`

What you should implement:

- Checkpointing utilities to:
  - Save model, optimizer, scheduler states, and current epoch/step.
  - Load from a checkpoint to resume training or for sampling.
- Handle:
  - Device mapping (CPU/GPU) when loading.
  - Compatibility between slightly different configs if you evolve the code.

---

## Configuration Files (`configs/`)

### `configs/default.yaml`

What you should implement:

- A base configuration capturing common defaults:
  - Model parameters (e.g. channels, depth, attention flags).
  - Diffusion parameters (timesteps, schedule type, beta range).
  - Training parameters (batch size, epochs, learning rate, optimizer type).
  - Data parameters (dataset name/path, resolution, number of workers).
  - Logging parameters (output dir, logging frequency).
- Keep values reasonable for a baseline experiment (e.g. CIFAR-10 resolution).

### Additional config files (optional)

- `configs/experiment_<name>.yaml` for specific experiments:
  - Override defaults to try different model sizes, schedules, or datasets.
  - Configure different logging backends or sampling settings.

---

## CLI Scripts (`scripts/`)

### `scripts/train.py`

What you should implement:

- A CLI entrypoint to start training:
  - Parse arguments such as:
    - `--config` path to YAML file.
    - Optional overrides (e.g. `--learning_rate`, `--batch_size`).
  - Load config using `config.py`.
  - Instantiate:
    - Dataset and data loaders.
    - Model(s) and noise scheduler.
    - Optimizer and LR scheduler.
  - Call the training loop from `training/train_loop.py`.
- Handle:
  - Seeding for reproducibility.
  - Device selection (CPU vs GPU, optional distributed training).

### `scripts/sample.py`

What you should implement:

- A CLI entrypoint to generate samples from a trained model:
  - Parse arguments such as:
    - `--config` path.
    - `--checkpoint` path.
    - `--num_samples`, `--steps`, `--output_dir`, etc.
  - Load config and checkpoint.
  - Build model, noise scheduler, and sampler.
  - Save generated samples as images (e.g. PNG), optionally as grids.

### (Optional) `scripts/evaluate.py`

What you should implement:

- Optional CLI for evaluating generated samples:
  - Compute quantitative metrics (FID, IS, etc.) if you integrate external libs.
  - Summarize and log results to console and logging backends.

---

## Tests (`tests/`)

The `tests/` directory should contain unit tests for each major component. The goal is to ensure that:

- Shapes are correct at each stage.
- Basic invariants of the diffusion process hold.
- Refactors don’t silently break training or sampling.

Example structure (files already created as empty placeholders):

- `tests/test_config.py`
  - Test that configs load correctly, required fields are present, and overrides work.
- `tests/test_dataset.py`
  - Test dataset length, item format, and that transforms are applied as expected.
- `tests/test_unet.py`
  - Test forward pass shapes for the U-Net given typical input sizes and timesteps.
- `tests/test_noise_scheduler.py`
  - Test noise scheduler computations (e.g. shapes and monotonicity of alphas, betas).
- `tests/test_sampler.py`
  - Test that the sampler runs a full denoising chain without errors and returns correctly shaped outputs.

You can expand this with more granular tests (e.g. for optimizer helpers, checkpointing, and logging).

---

## How to Start Implementing

1. **Set up the environment**
   - Complete `pyproject.toml` with your preferred dependencies and install them using your package manager.
2. **Implement configuration & logging**
   - Start with `config.py` and `utils/logging.py` so everything else can rely on a stable config and logging interface.
3. **Implement data pipeline**
   - Implement `data/dataset.py` and `data/transforms.py`, then write basic tests to confirm behavior.
4. **Implement the model & scheduler**
   - Implement `models/unet.py` and `schedulers/noise_scheduler.py`; focus on clear interfaces and unit tests for shapes.
5. **Implement training loop and optimizer helpers**
   - Implement `training/train_loop.py` and `training/optimizer.py`, then wire up `scripts/train.py`.
6. **Implement sampling**
   - Implement `sampling/sampler.py` and `scripts/sample.py`, verify you can generate images after a brief training run.
7. **Add tests and iterate**
   - Flesh out `tests/` with coverage for new features and modules as you add them.

By following this structure, you will end up with a **professional, modular diffusion model implementation** that is easy to maintain, extend, and experiment with.


