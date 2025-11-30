# Copilot instructions for MultimodalAI

This file gives concise, actionable guidance for AI coding agents working in this repository. The project is a **collaborative, production-ready ML solution** for Alzheimer's dementia detection using multimodal AI, with emphasis on team workflows, code quality, data scalability, and portable deployment via Streamlit.

## Project Objectives Alignment

**Apply ML skills in practice:**
- Train & evaluate robust deep learning models (ResNet50, multimodal architectures)
- Full pipeline: raw data → preprocessing → augmentation → training → evaluation → inference

**Team collaboration & Git workflows:**
- Modular architecture with clear separation of concerns (`MultimodalAI/` package)
- Config-driven experiments enable parallel work without code conflicts
- Checkpoints & logging for tracking experiment history
- All paths/parameters in YAML; code changes minimal across team

**Project management (cadrage to réalisation):**
- Config-driven architecture: `configs/config.yaml` captures scope, hyperparameters, data paths
- Reproducible runs: each training logs config + metrics + artifacts
- Notebooks for exploration; scripts for production runs
- Artifacts tracked: models in `models/best/`, logs in `logs/<timestamp>/`

**Real-world solutions with modern tools:**
- PyTorch + TorchVision for deep learning
- Albumentations for advanced image augmentation
- YAML for declarative config (not hardcoded params)
- TensorBoard for experiment visualization
- Streamlit for stakeholder-friendly UI (testing, evaluation, inference)

**Data scalability & portability:**
- Handles large volumes via DataLoader batching + num_workers
- All paths relative; works on any machine with same data structure
- Streamlit app runs standalone: `streamlit run app_streamlit.py`

---

## Core Architecture Layers

**1. Data Layer** (`MultimodalAI/data.py`)
- `AlzheimerDataset`: PyTorch Dataset for train/val/test splits
- `get_transforms()`: Albumentations pipelines (train: augment; val: normalize only)
- `get_data_loaders()`: Returns train/val/test DataLoaders with configurable batch size
- Handles label formats flexibly; supports class-based or file-based labels

**2. Model Layer** (`MultimodalAI/model.py`)
- `create_model()`: Builds ResNet50 (pretrained backbone, configurable dropout, num_classes)
- Supports architecture swap via config (`model.architecture`)
- Frozen/unfrozen backbone option for transfer learning

**3. Training Layer** (`MultimodalAI/train.py`)
- `train_epoch()`: One forward pass, backprop, metrics logging
- `validate_epoch()`: Inference-only validation loop
- Early stopping, checkpointing, learning rate scheduling built-in
- Metrics: loss, accuracy, logged to TensorBoard + console

**4. Orchestration** (`scripts/run_training.py`)
- Entry point: loads config, creates data/model/optimizer, runs training loop
- CLI overrides: `--batch-size`, `--epochs`, `--lr`, `--device`, `--debug`
- Checkpoint management: saves to `models/checkpoints/`, best model to `models/best/`
- Logs everything: config, training curves, history.json

**5. Evaluation & Deployment** (`scripts/evaluate_model.py` + `app_streamlit.py`)
- `evaluate_model.py`: Test set inference, confusion matrix, classification report
- `app_streamlit.py`: Web UI for model testing, evaluation metrics, batch inference
- Streamlit enables non-technical stakeholders to test models

---

## Important Workflows & Commands

### Installation & Setup
```powershell
# From project root
python scripts/setup_environment.py
# or manually
python -m pip install -r requirements.txt
```

### Training
```powershell
# Standard training
python scripts/run_training.py --config configs/config.yaml

# Debug mode (small subset, CPU)
python scripts/run_training.py --config configs/config.yaml --debug --device cpu --epochs 1 --batch-size 8

# Override hyperparameters
python scripts/run_training.py --config configs/config.yaml --batch-size 16 --epochs 50 --lr 0.0005

# Resume from checkpoint
python scripts/run_training.py --config configs/config.yaml --resume models/checkpoints/checkpoint_epoch_10.pth
```

### Evaluation on Test Set
```powershell
python scripts/evaluate_model.py --model models/best/best_model.pth --test-dir data/processed/test
```

### Streamlit App (Testing, Evaluation, Inference)
```powershell
streamlit run app_streamlit.py
# Opens UI at http://localhost:8501
# Features: upload image, get prediction + confidence, compare models, view metrics
```

---

## Code Quality & Structure Expectations

### Modularity & Portability
- **Single Responsibility:** Each module (`data.py`, `model.py`, `train.py`, `utils.py`) does one thing well
- **No hardcoded paths:** All file paths come from `config['paths']`. Use `Path` from `pathlib` for cross-platform compatibility
- **Clear APIs:** Each layer exports clean functions; avoid circular imports
- **Reproducibility:** Config + seed + checkpoint = exact reproduction of any run

### Configuration & Parameters
- **Config-first:** Any tunable parameter lives in `configs/config.yaml`, not in code
- **CLI overrides:** `scripts/run_training.py` allows selective override of config values
- **Experiment variants:** Use `configs/experiments/{baseline.yaml, fine_tuning.yaml, advanced.yaml}` for different scopes

### Documentation & Language
- **French user-facing text:** Prompts, print statements, docstrings in French for consistency
- **Code comments:** Explain *why*, not *what*. Docstrings for all public functions
- **Logging:** Use Python's `logging` module; log to file + console + TensorBoard

### Git Collaboration Practices
- **Config changes only:** Team members modify `configs/config.yaml` to experiment; minimal code conflicts
- **Checkpoints in .gitignore:** Model files (`.pth`) not versioned; logs only tracked in notebooks/reports
- **Branch strategy:** Feature branches for new capabilities (e.g., `feature/streamlit-ui`), merged via PR with code review
- **Reproducible runs:** Commit config + script version; colleague can reproduce exactly

---

## Data Handling for Large Volumes

**Scalability patterns:**
- **DataLoader batching:** `num_workers` in config (4 default); parallel data loading on CPU
- **Memory-efficient:** Images loaded on-demand; not all data in RAM
- **Augmentation on-the-fly:** Albumentations applies augmentations in data pipeline, not upfront
- **Checkpointing:** Mid-training saves; can resume from any epoch without retraining

**Directory structure for large datasets:**
```
data/processed/
├── train/
│   ├── images/       # many .jpg files
│   └── labels/       # corresponding labels file
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Integration & Cross-Component Notes

### Known API Mismatch (Important!)
- **Issue:** `scripts/run_training.py` imports `create_datasets()`, `create_dataloaders()` from `MultimodalAI.data`
- **Reality:** `MultimodalAI/data.py` currently exports `AlzheimerDataset`, `get_data_loaders()`, `get_transforms()`
- **Action required:** Either
  - Add wrapper functions in `data.py`: `def create_datasets(config, debug=False): ...` and `def create_dataloaders(...): ...`
  - OR update `run_training.py` to call `get_data_loaders()` directly
- **Recommendation:** Add wrappers; maintains decoupling

### Config Fields Expected
- `training`: batch_size, num_epochs, learning_rate, optimizer, scheduler, early_stopping
- `paths`: models_dir, checkpoints_dir, best_model_dir, logs_dir, figures_dir
- `logging`: tensorboard, log_interval, save_interval, verbose
- `hardware`: device (auto/cpu/cuda/mps), num_workers, pin_memory

### sys.path Handling
- `run_training.py` inserts repo root into `sys.path` so `MultimodalAI` imports work
- **Always run scripts from project root:** `python scripts/run_training.py` (not from `scripts/`)

---

## Testing & Validation Patterns

**Quick validation (before merging code):**
```powershell
python scripts/run_training.py --config configs/config.yaml --debug --epochs 1 --batch-size 8 --device cpu
```
- Verify: `history.json` created, checkpoint saved, no import errors
- Run time: ~2 min (small dataset, 1 epoch, CPU)

**Automated tests:**
- Minimal test suite in `tests/unit/` and `tests/integration/`
- Use small, deterministic datasets (mock data) for reproducibility
- Test data layer, model creation, training loop separately

**Validation checklist before deployment:**
- [ ] Config complete (no missing required keys)
- [ ] Data splits exist and are correctly labeled
- [ ] Model trains without CUDA/memory errors
- [ ] Best model saved in `models/best/`
- [ ] Streamlit app loads best model and predicts correctly

---

## File Edit Examples (Concrete Patterns)

### Add a CLI Flag to scripts/run_training.py
```python
# In parse_args() function:
parser.add_argument('--weight-decay', type=float, default=None,
                   help='Weight decay for optimizer (override config)')

# In setup_training() function:
if args.weight_decay:
    config['training']['weight_decay'] = args.weight_decay
```

### Add a Compatibility Wrapper in MultimodalAI/data.py
```python
# At bottom of data.py
def create_datasets(config, debug=False):
    """Wrapper for run_training.py compatibility."""
    train_transform, val_transform = get_transforms()
    train_dataset = AlzheimerDataset('train', transform=train_transform)
    val_dataset = AlzheimerDataset('val', transform=val_transform)
    if debug:
        train_dataset.image_paths = train_dataset.image_paths[:16]
        val_dataset.image_paths = val_dataset.image_paths[:8]
    return train_dataset, val_dataset

def create_dataloaders(train_dataset, val_dataset, config, debug=False):
    """Wrapper for run_training.py compatibility."""
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
```

### Add Streamlit Page for Model Comparison
```python
# In app_streamlit.py
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="MultimodalAI Evaluation", layout="wide")

models_dir = Path("models/best")
model_files = list(models_dir.glob("*.pth"))

selected_model = st.selectbox("Select model to evaluate", [m.name for m in model_files])
model_path = models_dir / selected_model
# Load and evaluate...
```

---

## Non-Obvious Pitfalls & Gotchas

1. **Placeholder docstrings:** `MultimodalAI/config.py`, `train.py`, `model.py` contain minimal/placeholder docstrings. Inspect actual implementations before assuming behavior.

2. **Data label formats:** `MultimodalAI/data.py` has comments showing multiple possible label formats (class index, YOLO format, etc.). Confirm which format your `data/processed/.../labels/` files use before transforming.

3. **TensorBoard by default:** `configs/config.yaml` has `logging.tensorboard: true`. On headless servers, either set to `false` or use TensorBoard port forwarding.

4. **Early stopping patience:** Default early_stopping.patience is 10 epochs. If training is unstable, increase it in config or disable early stopping.

5. **Device auto-detection:** `--device auto` tries CUDA first, falls back to CPU. On Mac, will try MPS. Explicitly specify `--device cpu` if debugging.

---

## When to Ask the User

Stop and ask if you encounter:
- **Ambiguous label format:** "What format are the labels in `data/processed/*/labels/`? (class index, one-per-line, YOLO, etc.)"
- **Missing API clarity:** "Should `data.py` or `run_training.py` be the source of truth for dataset creation? Should I add wrappers or refactor imports?"
- **Config conflicts:** "Are there experiment-specific configs in `configs/experiments/` that should override `configs/config.yaml`?"
- **Streamlit scope:** "What should `app_streamlit.py` do first? (inference only, or full evaluation dashboard?)"

---

## Summary: Objectives Met

| Objective | Implementation |
|-----------|-----------------|
| Apply ML skills | PyTorch pipeline from raw data to trained model; ResNet50 + multimodal support |
| Team collaboration | Git-safe config-driven design; checkpoints track experiment history |
| Project management | YAML configs capture scope; reproducible runs; dated logs/artifacts |
| Real-world solutions | Full training loop, evaluation, Streamlit UI for stakeholders |
| Scalability | DataLoader batching, num_workers for large volumes |
| Portability | All paths from config; Streamlit runs anywhere with Python |
