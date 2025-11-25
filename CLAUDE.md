# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a **Supervised Convolutional Autoencoder (CAE)** trained on CIFAR-10. The model performs two tasks simultaneously:
1. **Image Reconstruction**: Compress and reconstruct 64×64 RGB images
2. **Image Classification**: Classify images into 10 CIFAR-10 categories

The architecture learns a shared 64-dimensional latent representation that serves both tasks. Current performance: **83.5% test accuracy** with reconstruction RMSE of ~0.083.

## Repository Structure

```
notebooks/
  └── 02_supervised_cae_64x64_cifar10.ipynb  # Main training notebook
data/
  └── cifar-10-batches-py/                   # Auto-downloaded CIFAR-10 dataset
```

The `.gitignore` excludes `data/`, `checkpoints/`, `trials_nb/`, and `doc/` directories to keep the repository lean.

## Development Commands

### Running the Notebook

The primary workflow is contained in a single Jupyter notebook. Run it with:

```bash
jupyter notebook notebooks/02_supervised_cae_64x64_cifar10.ipynb
```

Or use your preferred notebook interface (JupyterLab, VS Code, etc.)

### Key Python Operations

**Training a model:**
```python
# All hyperparameters are defined in CONFIG dictionary
model = SupervisedAE(
    latent_dim=CONFIG["latent_dim"],
    num_classes=CONFIG["num_classes"],
    lambda_recon=CONFIG["lambda_recon"],
    latent_noise_std=CONFIG["latent_noise_std"],
    classifier_dropout=CONFIG["classifier_dropout"],
)

optimizer = torch.optim.NAdam(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
)

history = train(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    n_epochs=CONFIG["epochs"],
    loss_type=CONFIG["loss_type"],
)
```

**Evaluating the model:**
```python
# Get test set metrics
test_metrics = evaluate(model, test_loader, device, loss_type='mse')

# Generate classification report
report, y_true, y_pred = evaluate_classification_report(
    model=model,
    loader=test_loader,
    class_names=cifar10_classes,
    device=device
)

# Visualize reconstructions
show_reconstructions(model, test_loader, n=10)
```

**Visualizing latent space:**
```python
Z, y = extract_latents(model, test_loader)
plot_latent_2d(Z, y, method="tsne", class_names=cifar10_classes)
plot_latent_2d(Z, y, method="pca", class_names=cifar10_classes)
```

## Dependencies

Install all required packages with:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install tqdm einops
pip install torchmetrics scikit-learn
pip install mlflow optuna  # Optional: for experiment tracking and hyperparameter tuning
```

For Google Colab, prefix each command with `!`.

## Model Architecture

### High-Level Structure

```
Input Image (3, 64, 64)
       ↓
   [Encoder] → Conv layers with stride-2 downsampling
       ↓
Latent Vector (64-dim) ← Optional Gaussian noise during training
       ↓
    ┌──┴──┐
    ↓     ↓
[Decoder] [Classifier]
    ↓        ↓
Reconstructed  Class Logits
Image (3,64,64)  (10 classes)
```

### Component Details

**Encoder:**
- 4 convolutional blocks: 3→32→64→128→128 channels
- Spatial downsampling: 64→32→16→8→4
- Final linear projection to latent_dim (default: 64)
- Gaussian noise injection (std=0.03) during training for regularization

**Decoder:**
- Linear projection from latent_dim to 128×4×4 feature map
- 4 transpose-conv blocks for upsampling: 4→8→16→32→64
- Channels: 128→64→32→16→3
- Sigmoid activation for [0,1] pixel values

**Classifier Head:**
- MLP: latent_dim → 64 (hidden) → num_classes
- BatchNorm + ReLU + Dropout(0.25) regularization

### Loss Function

```python
Total Loss = Classification Loss + λ_recon × Reconstruction Loss + L1 Regularization

where:
  Classification Loss = CrossEntropy(logits, labels)
  Reconstruction Loss = MSE(x_reconstructed, x_original)  # or BCE
  L1 Regularization = 0.05 × mean(|latent_vector|)
```

## Critical Hyperparameters

The `CONFIG` dictionary contains all hyperparameters. Key parameters that significantly affect model behavior:

| Parameter | Default | Purpose | Typical Range |
|-----------|---------|---------|---------------|
| `lambda_recon` | 50 | Weight for reconstruction loss vs classification | 0.1–50 |
| `latent_dim` | 64 | Bottleneck dimension (compression level) | 32, 64, 128 |
| `latent_noise_std` | 0.03 | Gaussian noise std in latent space (regularization) | 0.0–0.1 |
| `classifier_dropout` | 0.25 | Dropout rate in classifier head | 0.0–0.5 |
| `lr` | 3e-4 | Learning rate for NAdam optimizer | 1e-5–1e-3 |
| `loss_type` | 'mse' | Reconstruction loss: 'mse' or 'bce' | — |

**Important Notes:**
- Higher `lambda_recon` prioritizes reconstruction quality over classification
- Lower `latent_noise_std` = cleaner latents, but may overfit
- The model uses gradient clipping (max_norm=1.0) and ReduceLROnPlateau scheduler

## Data Pipeline

### CIFAR-10 Dataset
- **Auto-download**: Dataset automatically downloads to `data/cifar-10-batches-py/` on first run
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split**: 50k train (40k train / 10k validation), 10k test
- **Split ratio**: Controlled by `split_train=0.8` in CONFIG

### Data Augmentation

Training data uses aggressive augmentation (when `augment=True`):
```python
- Resize to 64×64
- RandomCrop(64, padding=4)
- RandomHorizontalFlip()
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
- ToTensor() → [0, 1] range
```

Validation/test data only applies resize and ToTensor (no augmentation).

### DataLoader Configuration
```python
batch_size = 128
num_workers = 4
pin_memory = True          # Faster GPU transfer
persistent_workers = True  # Keep workers alive between epochs
```

## Hyperparameter Optimization

The notebook includes Optuna integration (currently commented out in cells 31-32). To enable hyperparameter search:

1. **Uncomment the Optuna cells** (search for `# ==== Optuna hyperparameter search`)
2. **Choose optimization target**:
   ```python
   selection_metric = "accuracy"   # maximize validation accuracy
   # OR
   selection_metric = "val_rmse"   # minimize reconstruction error
   ```
3. **Run the search**:
   ```python
   study.optimize(objective, n_trials=25, show_progress_bar=True)
   ```
4. **Apply best parameters**:
   ```python
   CONFIG["latent_dim"] = int(study.best_params["latent_dim"])
   CONFIG["lambda_recon"] = float(study.best_params["lambda_recon"])
   CONFIG["latent_noise_std"] = float(study.best_params["latent_noise_std"])
   CONFIG["lr"] = float(study.best_params["lr"])
   ```

**Search Space:**
- `latent_dim`: categorical [32, 64, 128]
- `lambda_recon`: log-uniform [0.05, 50]
- `latent_noise_std`: uniform [0.0, 0.1]
- `lr`: log-uniform [1e-4, 1e-3]

The search runs for fewer epochs (10) than full training to speed up exploration.

## Evaluation and Visualization

The notebook provides comprehensive evaluation tools:

1. **Quantitative Metrics**:
   - Classification: accuracy, precision, recall, F1-score per class
   - Reconstruction: MSE, RMSE (root mean squared error)
   - Combined total loss

2. **Visualizations**:
   - Training curves: loss, accuracy, RMSE over epochs
   - Image reconstructions with per-sample MSE annotations
   - Confusion matrix (normalized and raw counts)
   - Latent space projections (t-SNE, PCA)

3. **Classification Report**:
   ```python
   # CIFAR-10 classes accessed via:
   cifar10_classes = train_loader.dataset.dataset.classes
   # ['airplane', 'automobile', 'bird', 'cat', 'deer',
   #  'dog', 'frog', 'horse', 'ship', 'truck']
   ```

## Current Status and TODOs

**Branch**: `add-latent-space-analysis`

**Completed**:
- Core supervised CAE implementation with type annotations
- Full training pipeline with validation and testing
- Hyperparameter optimization infrastructure (Optuna)
- t-SNE and PCA latent space visualization
- Comprehensive evaluation metrics

**Outstanding TODOs** (marked in notebook):
- UMAP visualization (cell 49: `# TODO: add umap plots`)
- PCA cumulative explained variance ratio (cell 47)
- Results interpretation section (empty section at end)
- Model checkpointing/saving functionality

## Training Best Practices

1. **Reproducibility**: All random operations use `seed=42` (Python, NumPy, PyTorch CPU/CUDA)

2. **Gradient Management**:
   - Gradient clipping: `max_norm=1.0` prevents exploding gradients
   - `optimizer.zero_grad(set_to_none=True)` for memory efficiency

3. **Learning Rate Scheduling**:
   - ReduceLROnPlateau monitors validation accuracy
   - Reduces LR by factor=0.5 after patience=2 epochs without improvement

4. **Device Handling**:
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```
   All operations automatically use GPU when available.

5. **Latent Space Regularization**:
   - Gaussian noise during training: prevents deterministic overfitting
   - L1 penalty on latent activations: encourages sparsity
   - Dropout in classifier head: prevents co-adaptation

## Type Annotations

The codebase uses comprehensive type hints for clarity:
```python
def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    n_epochs: int = 10,
    device: str | None = None,
    max_grad_norm: float = 1.0,
    scheduler_patience: int = 2,
    scheduler_factor: float = 0.5,
    loss_type: str = "mse"
) -> dict[str, list[float]]:
```

This aids in understanding function contracts and catching type-related bugs early.

## Git Workflow

Recent commit history shows iterative development:
```
758dd2b nb02: renamed notebook2
7f0cbb5 nb02: refactored notebook and added type annotation
9659c73 nb02: refactoring
ec77fab updated the config parameters for best accuracy
```

When making changes:
- Keep notebook output cleared for clean diffs (or commit with outputs for documentation)
- Model checkpoints and data should not be committed (already in `.gitignore`)
- Document significant hyperparameter changes in commit messages
