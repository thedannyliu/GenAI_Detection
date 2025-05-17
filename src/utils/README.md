# Utils Module

This module provides utility functions and helpers used throughout the project. It contains tools for logging, metrics calculation, configuration management, and more.

## Key Components

### `metrics.py`

Functions for computing evaluation metrics:

```python
from src.utils.metrics import compute_accuracy, compute_auc, compute_precision_recall_f1

# Calculate accuracy
accuracy = compute_accuracy(predictions, labels)

# Calculate ROC AUC score
auc = compute_auc(predictions, labels)

# Calculate precision, recall, and F1 score
precision, recall, f1 = compute_precision_recall_f1(predictions, labels, threshold=0.5)

# Get all metrics at once
metrics = calculate_all_metrics(predictions, labels)
```

### `logger.py`

Handles logging configuration and provides logging utilities:

```python
from src.utils.logger import setup_logger, get_logger

# Set up logging
setup_logger(log_file="results/logs/experiment.log", level="INFO")

# Get logger and use it
logger = get_logger(__name__)
logger.info("Starting experiment with learning rate: %f", learning_rate)
logger.debug("Loaded %d training samples", len(train_dataset))
```

### `config.py`

Functions for loading and parsing configuration files:

```python
from src.utils.config import load_config

# Load configuration from YAML file
config = load_config("configs/vlm_fine_tune.yaml", default_config_path="configs/default.yaml")

# Access nested configuration values
learning_rate = config.training.learning_rate
batch_size = config.training.batch_size
```

### `visualization.py`

Helper functions for creating visualizations:

```python
from src.utils.visualization import plot_training_history, plot_roc_curve

# Plot training/validation curves
fig = plot_training_history(
    train_losses=[0.8, 0.6, 0.4, 0.3],
    val_losses=[0.9, 0.7, 0.5, 0.4],
    train_accuracies=[0.6, 0.7, 0.8, 0.85],
    val_accuracies=[0.55, 0.65, 0.75, 0.8],
    epochs=4
)
fig.savefig("results/figures/training_history.png")

# Plot ROC curve
fig = plot_roc_curve(y_true, y_pred)
fig.savefig("results/figures/roc_curve.png")
```

### `device.py`

Functions for handling device selection and memory management:

```python
from src.utils.device import get_device, set_memory_efficient_mode

# Get appropriate device (CUDA if available, else CPU)
device = get_device()

# Move model to device
model = model.to(device)

# Set up mixed precision and other memory-saving features
set_memory_efficient_mode()
```

## Utility Functions

### Random Seed

```python
from src.utils.seed import set_seed

# Set random seed for reproducibility
set_seed(42)
```

### File I/O

```python
from src.utils.io import save_json, load_json, ensure_dir

# Create directory if it doesn't exist
ensure_dir("results/figures")

# Save and load JSON data
save_json(metrics, "results/evaluation/metrics.json")
metrics = load_json("results/evaluation/metrics.json")
```

### Progress Tracking

```python
from src.utils.progress import create_progress_bar

# Create progress bar for monitoring training
progress_bar = create_progress_bar(total=len(train_loader), desc="Training")

# Update progress
for i, batch in enumerate(train_loader):
    # Training step
    progress_bar.update(1)
    progress_bar.set_postfix(loss=current_loss, acc=current_acc)
```

### Weights & Biases Integration (Optional)

```python
from src.utils.wandb_logger import init_wandb, log_metrics

# Initialize W&B logging
init_wandb(project="ai-image-detection", config=config)

# Log metrics during training
log_metrics({"loss": loss, "accuracy": acc, "learning_rate": lr})
```

## Common Usage Patterns

### Configuration Management

```python
# In main.py
from src.utils.config import load_config

def main():
    # Parse command-line arguments (e.g., config file path, overrides)
    args = parse_args()
    
    # Load configuration with overrides
    config = load_config(
        args.config_file,
        default_config_path="configs/default.yaml",
        overrides=args.overrides
    )
    
    # Set random seed
    set_seed(config.seed)
    
    # Set up logging
    setup_logger(
        log_file=os.path.join(config.output_dir, "experiment.log"),
        level=config.log_level
    )
    
    # Rest of setup...
```

### Tracking Metrics During Training

```python
# In trainer.py
from src.utils.metrics import calculate_all_metrics
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_epoch(model, loader, optimizer, device):
    # Initialize metrics
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Training loop
    for batch in loader:
        # Training step
        
        # Accumulate predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += loss.item()
    
    # Calculate metrics at epoch end
    metrics = calculate_all_metrics(np.array(all_preds), np.array(all_labels))
    metrics["loss"] = running_loss / len(loader)
    
    # Log metrics
    logger.info(
        "Epoch %d - Loss: %.4f, Accuracy: %.4f, AUC: %.4f",
        epoch, metrics["loss"], metrics["accuracy"], metrics["auc"]
    )
    
    return metrics
```

## Helper Functions for VLMs

### CLIP Processing

```python
from src.utils.clip_helpers import prepare_clip_input

# Process images for CLIP
clip_images = prepare_clip_input(images, device=device)

# Generate text features from prompts
text_features = get_clip_text_features(
    model=clip_model,
    prompts=["a real photo", "an AI-generated image"],
    device=device
)
```

### Prompt Engineering

```python
from src.utils.prompt_utils import create_class_prompts

# Generate class-specific prompts
prompts = create_class_prompts(
    template="a {} image",
    classes=["real", "AI-generated"]
)
```

## Configuration Files

Example configuration file structure:

```yaml
# configs/default.yaml
seed: 42
output_dir: "results"
log_level: "INFO"

data:
  root_dir: "data/genimage/imagenet_ai"
  batch_size: 32
  num_workers: 4
  
model:
  type: "clip"
  name: "openai/clip-vit-base-patch32"
  mode: "fine-tune"
  freeze_backbone: true
  
training:
  num_epochs: 10
  optimizer: "adamw"
  learning_rate: 1e-4
  weight_decay: 0.01
```

## Best Practices

- Use the logger consistently across the project instead of print statements
- Set random seeds for reproducibility
- Keep configuration in YAML files rather than hardcoding values
- Use the provided metrics functions for consistent evaluation
- Handle file paths with platform-independent methods

## Adding New Utilities

To add a new utility function:

1. Identify the appropriate module or create a new one if necessary
2. Implement the function with proper docstrings
3. Add unit tests to verify functionality
4. Import and use the function in your code

Example:

```python
# In src/utils/image_utils.py
def resize_and_crop(image, target_size):
    """Resize an image and apply center crop to reach target size.
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size as (width, height)
    
    Returns:
        PIL.Image: Resized and cropped image
    """
    # Implementation
    
    return processed_image
``` 