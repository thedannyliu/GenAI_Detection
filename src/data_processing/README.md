# Data Processing Module

## Overview
Dataset loaders and preprocessing utilities for GenImage and related datasets.

This module handles data loading and preprocessing for the GenImage dataset. It provides PyTorch Dataset classes and transformation utilities to prepare images for model training and evaluation.

## Key Components

### `custom_dataset.py`

Contains the `GenImageDataset` class that loads and prepares images from the genimage dataset:

```python
from src.data_processing.custom_dataset import GenImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Create data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset
train_dataset = GenImageDataset(
    root_dir="data/genimage/imagenet_ai",
    split="train",
    transform=transform
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate through batches
for images, labels in train_loader:
    # images shape: [batch_size, 3, 224, 224]
    # labels shape: [batch_size], each value is 0 (real) or 1 (AI-generated)
    ...
```

## Dataset Features

- **Split Selection**: Load `train`, `val`, or `test` splits using the `split` parameter. The specified split name is used directly to find the corresponding subdirectory (e.g., `root_dir/test/`).
- **Transformations**: Apply PyTorch transforms for augmentation and preprocessing
- **Generator Selection**: Optionally filter by specific generator with `generator` parameter
- **Class Sampling**: Sample from specific ImageNet classes if needed
- **Flexible Class Directory Mapping**: The `GenImageDataset` now dynamically determines class subdirectories based on the `class_to_idx` parameter provided during initialization. The keys of the `class_to_idx` dictionary (e.g., `{"0_real": 0, "1_fake": 1}` or `{"nature": 0, "ai": 1}`) are used to identify the respective class folders within the `root_dir/split/` path. This allows for compatibility with datasets that use different naming conventions for their class folders (e.g., `0_real`, `1_fake` for the Chameleon dataset, or the default `nature`, `ai`). If `class_to_idx` is not provided or is empty, the dataset defaults to looking for `nature` and `ai` subdirectories.

## Recommended Transforms

### Training Transforms

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Validation/Test Transforms

```python
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## Advanced Features

### Generator-Specific Datasets

To work with a specific generator's images:

```python
midjourney_dataset = GenImageDataset(
    root_dir="data/genimage/Midjourney",  # Point to generator folder
    split="train",
    transform=transform
)
```

### Cross-Generator Testing

For leave-one-out experiments, you can specify included/excluded generators:

```python
# Create dataset that excludes one generator
excluded_generator = "Midjourney"
train_dataset = GenImageDataset(
    root_dir="data/genimage",  # Parent directory containing all generators
    split="train",
    transform=transform,
    exclude_generators=[excluded_generator]
)

# Create dataset for testing on the excluded generator
test_dataset = GenImageDataset(
    root_dir=f"data/genimage/{excluded_generator}",
    split="val",
    transform=transform
)
```

## Example Usage in Training Loop

```python
# Set up datasets
train_dataset = GenImageDataset(
    root_dir=config.data.root_dir,
    split="train",
    transform=train_transform
)

val_dataset = GenImageDataset(
    root_dir=config.data.root_dir,
    split="val",
    transform=val_transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.training.batch_size * 2,  # Larger batches for validation
    shuffle=False,
    num_workers=config.data.num_workers,
    pin_memory=True
)
```
## Notes on Performance

- When working with the full dataset, consider using multiple workers and pinned memory
- For faster loading, SSD storage is recommended over HDD
- If memory constraints are an issue, reduce batch size or use a dataset subset 
