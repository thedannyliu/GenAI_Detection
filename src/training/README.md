# Training Module

This module provides tools and utilities for training AI-generated image detection models. It implements the training loop, optimization strategies, and checkpointing.

## Key Components

### `trainer.py`

Contains the core `Trainer` class and related functions:

```python
from src.training.trainer import Trainer
from src.models.vlm_models import CLIPModel
from src.data_processing.custom_dataset import GenImageDataset
from torch.utils.data import DataLoader

# Create model
model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune")

# Set up datasets and dataloaders
train_dataset = GenImageDataset(...)
val_dataset = GenImageDataset(...)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer_type="adamw",
    learning_rate=1e-4,
    weight_decay=0.01,
    scheduler_type="cosine",
    num_epochs=10,
    device="cuda",
    checkpoint_dir="results/saved_models"
)

# Start training
trainer.train()
```

## Training Features

### Optimization Options

- **Optimizers**: AdamW (default), Adam, SGD
- **Learning Rate Schedulers**:
  - Cosine annealing with warmup
  - Step scheduler
  - ReduceLROnPlateau
- **Gradient Management**:
  - Gradient clipping
  - Gradient accumulation
  - Mixed precision training

### Training Configuration

Example YAML configuration:

```yaml
training:
  # Optimization
  optimizer: "adamw"
  learning_rate: 1e-4
  weight_decay: 0.01
  momentum: 0.9  # For SGD

  # Scheduler
  scheduler: "cosine"
  warmup_epochs: 1
  min_lr: 1e-6
  
  # Training loop
  num_epochs: 10
  batch_size: 32
  grad_accumulation_steps: 1
  
  # Checkpointing
  checkpoint_dir: "results/saved_models"
  save_frequency: 1  # Save every epoch
  
  # Miscellaneous
  early_stopping: True
  patience: 3  # Epochs to wait for improvement
  mixed_precision: True
  clip_grad_norm: 1.0
```

## Training Loop Details

The `Trainer` class provides a comprehensive training loop that:

1. Iterates through batches in the training dataloader
2. Performs forward pass through the model
3. Calculates loss using appropriate criterion (e.g., CrossEntropyLoss)
4. Computes gradients and updates model parameters
5. Evaluates model on validation set after each epoch
6. Tracks and logs metrics (loss, accuracy, AUC)
7. Saves model checkpoints based on validation performance

## Logging and Monitoring

The training module integrates with:

- **Console/File Logging**: Progress bars and detailed metrics
- **TensorBoard**: Real-time visualization of metrics
- **Weights & Biases** (optional): Experiment tracking

Example logging output:
```
Epoch 3/10: 100%|████████████████| 500/500 [02:13<00:00, 3.74it/s]
Train Loss: 0.284, Train Acc: 0.869
Val Loss: 0.193, Val Acc: 0.923, Val AUC: 0.976
Saving checkpoint to: results/saved_models/epoch_3.pth
```

## Multi-GPU Training

For distributed training on multiple GPUs:

```python
# Initialize trainer with distributed option
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    distributed=True,
    # other parameters
)

# Train with distributed data parallel
trainer.train()
```

## Resuming Training

To resume training from a checkpoint:

```python
# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    # other parameters
)

# Resume from checkpoint
trainer.resume_from_checkpoint("results/saved_models/epoch_5.pth")
trainer.train()
```

## Training Strategies

### Full Fine-tuning

```python
model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="fine-tune",
    freeze_backbone=False  # Fine-tune all parameters
)

trainer = Trainer(
    model=model,
    learning_rate=5e-5,  # Lower learning rate for full fine-tuning
    # other parameters
)
```

### Partial Fine-tuning

```python
model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="fine-tune",
    freeze_backbone=True  # Only train classifier head
)

trainer = Trainer(
    model=model,
    learning_rate=1e-3,  # Higher learning rate for partial fine-tuning
    # other parameters
)
```

### Prompt Tuning

```python
base_model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="zero-shot")
model = PromptTuningWrapper(model=base_model, prompt_length=10)

trainer = Trainer(
    model=model,
    learning_rate=1e-2,  # Higher learning rate for prompt tuning
    # other parameters
)
```

## Integration with main.py

The training module is designed to be called from `main.py`:

```python
# In main.py
from src.training.trainer import Trainer

def main(config):
    # Set up model, datasets, etc.
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        **config.training
    )
    
    # Train model
    trainer.train()
```

## Best Practices

- Use cosine annealing scheduler for stable training
- Enable mixed precision for faster training and reduced memory
- Start with partial fine-tuning and then consider full fine-tuning
- Save checkpoints regularly
- Monitor validation metrics for overfitting

## Advanced Usage

For specialized training scenarios or custom loss functions, you can extend the `Trainer` class:

```python
class CustomTrainer(Trainer):
    def calculate_loss(self, outputs, targets):
        # Custom loss calculation
        return my_custom_loss(outputs, targets)
```

### `train_instructblip.py`

This script fine-tunes the Salesforce InstructBLIP model (e.g., `Salesforce/instructblip-vicuna-7b`) for binary classification of real versus AI-generated images, primarily utilizing LoRA (Low-Rank Adaptation) for efficient tuning.

**Usage:**

```bash
python src/training/train_instructblip.py --config configs/your_instructblip_config.yaml
```

**Key Features:**

*   **LoRA Fine-tuning:** Leverages PEFT library for LoRA Geschmack anpassen, focusing on adapting the language model part of InstructBLIP.
*   **Binary Image Classification:** Specifically designed to distinguish between authentic and AI-synthesized images.
*   **Custom Data Handling:** Implements `AIImageDataset` for loading images and prompts, and `CustomDataCollator` for batching.
*   **Custom Loss & Metrics:** Includes a `CustomTrainer` to handle potential shape mismatches in labels during loss computation and a `compute_metrics` function for evaluating accuracy, precision, recall, and F1-score.
*   **YAML Configuration:** All aspects of data, model, training, and environment are controlled via a YAML configuration file.

**Core Components:**

1.  **`AIImageDataset(Dataset)`**:
    *   Loads real and AI-generated images.
    *   Constructs prompts for InstructBLIP, combining a question (e.g., "Is this image real or AI-generated? Answer with one word.") with the expected answer ("real" or "fake").
    *   Processes images and text using `InstructBlipProcessor`.
    *   Generates `labels` for the language model, masking out the question part so that loss is computed only on the answer tokens.

2.  **`CustomDataCollator`**:
    *   Pads sequences within a batch to the maximum length.
    *   Specifically handles padding for `labels` using an ignore index (-100).

3.  **`compute_metrics(eval_pred)`**:
    *   Called during evaluation by the `Trainer`.
    *   Decodes token predictions from the language model.
    *   Flattens and filters out ignored labels (-100).
    *   Converts token IDs to binary class labels (0 for real, 1 for AI) for metric calculation. *Note: The current mapping (`predictions % 2`) is a simplification and might need refinement based on actual token IDs for "real" and "fake".*
    *   Calculates accuracy, precision, recall, and F1-score.

4.  **`CustomTrainer(Trainer)`**:
    *   Overrides `compute_loss` to address potential discrepancies between `input_ids` and `labels` shapes that can occur with InstructBLIP. It attempts to reshape or pad `labels` appropriately before the standard loss calculation.

5.  **`main()` Function Flow**:
    *   Parses command-line arguments (primarily the config file path).
    *   Loads the YAML configuration.
    *   Sets up the device (GPU/CPU) and random seeds.
    *   Creates an output directory for results.
    *   Loads and splits image data using `collect_images` and `split_dataset`.
    *   Loads the `InstructBlipProcessor` and `InstructBlipForConditionalGeneration` model.
        *   Crucially uses `device_map={"": device}` for efficient model loading onto the target device.
    *   If LoRA is enabled in the config:
        *   Creates a `LoraConfig` from `peft`.
        *   Applies LoRA to the language model component of InstructBLIP using `get_peft_model`.
    *   Initializes `AIImageDataset` for training and validation.
    *   Sets up `TrainingArguments` (controls epochs, batch size, learning rate, logging, evaluation strategy, etc.).
        *   `prediction_loss_only` is set to `False` to enable `compute_metrics`.
    *   Initializes the `CustomTrainer`.
    *   Starts training if `should_train` is true in the config.
    *   Saves the final model, processor, and LoRA adapter (if used).
    *   Saves the configuration file to the output directory.

**Example YAML Configuration (`your_instructblip_config.yaml`):**

```yaml
data:
  train_path: "path/to/train_data_root"      # Root directory for training data
  val_test_path: "path/to/val_test_data_root" # Root directory for validation/test data
  num_train_samples: 1000
  num_val_samples: 200
  num_test_samples: 200                     # Used for splitting, actual testing not in this script
  seed: 42

model:
  name_pretrained: "Salesforce/instructblip-vicuna-7b"
  finetune_method: "lora"                 # "lora" or potentially "full"
  lora_params:
    r: 8
    alpha: 16
    dropout: 0.05
    target_modules: ["q_proj", "v_proj"]  # Modules to apply LoRA

training:
  epochs: 5
  batch_size: 4
  gradient_accumulation_steps: 1
  learning_rate_lora: 0.0001
  warmup_steps: 50
  weight_decay: 0.01
  max_target_token_length: 64             # Max length for the answer part ("real"/"fake")
  evaluation_strategy: "steps"            # "no", "steps", "epoch"
  eval_steps: 50                          # Evaluate every N steps if strategy is "steps"
  save_strategy: "steps"
  save_steps: 50
  save_total_limit: 2
  load_best_model_at_end: True
  metric_for_best_model: "eval_loss"      # Or "eval_f1", "eval_accuracy"
  greater_is_better: False                # True if metric_for_best_model improves with higher values
  early_stopping_patience: 3
  early_stopping_threshold: 0.001
  should_train: True
  use_bfloat16: True
  gradient_checkpointing: False
  # prediction_loss_only was set to False in the script to enable metrics

output:
  base_results_dir_root: "results/instructblip_runs"

environment:
  gpu_id: 0                               # Specific GPU ID, or null/negative for default
``` 