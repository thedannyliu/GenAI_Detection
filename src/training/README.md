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