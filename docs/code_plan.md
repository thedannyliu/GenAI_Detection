# Code Design: AI-Generated Image Detection using VLMs

This document outlines the design and architecture of our codebase for detecting AI-generated images using Vision-Language Models. Each section describes one of the core modules and their interactions.

## Module Structure

```
src/
├── data_processing/    <- Data loading and preprocessing
├── models/            <- Model definitions and wrappers
├── training/          <- Training pipeline
├── evaluation/        <- Evaluation and metrics
└── utils/             <- Utility functions
```

## 1. Data Processing Module (`src/data_processing/`)

### Core Components
- **GenImageDataset**: Custom PyTorch Dataset for loading genimage data
- **Data augmentation transformations**: Applied during training

### Key Features
- Handles multiple dataset organizations (by generator or combined)
- Applies appropriate preprocessing and augmentations
- Maintains class balance for training

### Data Flow
1. The dataset locates image files from disk
2. Images are loaded and transformed to model-appropriate format
3. Returns batches of (image, label) pairs for training/evaluation

## 2. Models Module (`src/models/`)

### Core Components
- **Baseline Classifiers**: Traditional CNN models (ResNet-50, etc.)
- **VLM Wrappers**: Classes encapsulating CLIP, BLIP, and other VLMs
- **Prompt/Adapter Modules**: Parameter-efficient adaptation components

### Architecture
- All models follow a common interface with a standard forward() method
- Zero-shot models include specialized prompt-based prediction functions
- Fine-tuning strategies are implemented as model variations

### Model Types
1. **ResNet50Classifier**: Baseline CNN classifier
2. **CLIPModel**: Wrapper for OpenAI's CLIP or HuggingFace implementation
3. **BLIPModel**: Wrapper for Salesforce's BLIP model
4. **PromptTuningWrapper**: Adds learnable prompt parameters to VLMs
5. **AdapterEnhancedModel**: Adds lightweight adapter modules to transformers

## 3. Training Module (`src/training/`)

### Core Components
- **Trainer**: Orchestrates the training loop
- **Loss functions**: Binary cross-entropy or alternatives
- **Optimization**: AdamW with appropriate learning rates

### Training Flow
1. Initialize model, datasets, and optimizer
2. For each epoch:
   - Train over all batches in training set
   - Evaluate on validation set
   - Update learning rate schedules
   - Save checkpoints based on performance

### Features
- Supports both full and partial fine-tuning strategies
- Training resumption from checkpoints
- TensorBoard and/or W&B logging
- Multi-GPU training when available

## 4. Evaluation Module (`src/evaluation/`)

### Core Components
- **Evaluator**: Runs models on test datasets and computes metrics
- **Metrics**: Functions for calculating performance statistics
- **Visualization**: Utilities for creating ROC curves, Grad-CAM, etc.

### Evaluation Types
1. Standard evaluation on validation/test sets
2. Cross-generator evaluation (train on one set, test on another)
3. Robustness evaluation against image manipulations
4. Explainability analysis with attention visualization

### Metrics
- Accuracy, precision, recall, F1-score
- ROC curves and AUC
- Confusion matrices
- Per-generator performance analysis

## 5. Utils Module (`src/utils/`)

### Core Components
- **Logging**: Setup of console and file loggers
- **Metrics**: Common metric calculation functions
- **Visualization**: Utilities for plotting results
- **Configuration**: Config parsing and validation

### Helper Functions
- Random seed setting for reproducibility
- Device detection and management
- File I/O helpers

## Main Execution Flow

The project execution is controlled by the `main.py` script which:

1. Parses configuration from files or command line arguments
2. Sets up the environment (seeds, logging, devices)
3. Initializes the appropriate dataset
4. Creates the model based on configuration
5. Either:
   - Runs the training loop if in training mode
   - Evaluates a checkpoint if in evaluation mode
   - Performs specialized analysis (e.g., Grad-CAM) if requested

## Configuration System

Configurations are stored as YAML files in the `configs/` directory:

- **default.yaml**: Base configuration with shared parameters
- **vlm_zero_shot.yaml**: Settings for zero-shot evaluation
- **vlm_fine_tune.yaml**: Settings for VLM fine-tuning
- **cnn_baseline.yaml**: Settings for baseline CNN training

Parameters include:
- Dataset paths and organization
- Model selection and hyperparameters
- Training settings (learning rates, batch sizes, etc.)
- Evaluation metrics and thresholds

## Integration Points

The modular design allows for:
- Easy swapping of models (change the model class instantiation)
- Testing different datasets (create a new Dataset subclass)
- Adding new evaluation metrics (implement in utils.metrics)
- Supporting new training strategies (extend the Trainer class)

This architecture emphasizes flexibility for research while maintaining reproducibility and clear organization. 