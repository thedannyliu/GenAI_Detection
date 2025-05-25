# Evaluation Module

This module provides tools and functions for evaluating AI-generated image detection models. It handles metrics calculation, performance visualization, and analysis of model behavior.

## Key Components

### `evaluator.py`

Contains the `Evaluator` class for model evaluation:

```python
from src.evaluation.evaluator import Evaluator
from src.models.vlm_models import CLIPModel
from torch.utils.data import DataLoader
from src.data_processing.custom_dataset import GenImageDataset

# Load a trained model
model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune")
model.load_state_dict(torch.load("results/saved_models/best_model.pth"))

# Create evaluation dataset and dataloader
test_dataset = GenImageDataset(root_dir="data/genimage/imagenet_ai", split="val", transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize evaluator
evaluator = Evaluator(
    model=model,
    data_loader=test_loader,
    metrics=["accuracy", "auc", "precision", "recall", "f1"],
    device="cuda"
)

# Run evaluation
results = evaluator.evaluate()
print(results)
```

## Available Metrics

The evaluator supports multiple performance metrics:

- **Accuracy**: Proportion of correct predictions
- **AUC (Area Under ROC Curve)**: Measures discrimination ability across thresholds
- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Counts of true/false positives/negatives
- **Per-Class Accuracy**: Performance breakdown by class
- **Cross-Generator Metrics**: Performance on each generator

## Evaluation Types

### Standard Evaluation

Evaluates a model on a test dataset:

```python
results = evaluator.evaluate()
# returns: {'accuracy': 0.95, 'auc': 0.98, ...}
```

### Cross-Generator Evaluation

Tests generalization to unseen generators:

```python
# Create evaluator with test set from a specific generator
midjourney_dataset = GenImageDataset(
    root_dir="data/genimage/Midjourney",
    split="val",
    transform=val_transform
)
midjourney_loader = DataLoader(midjourney_dataset, batch_size=64, shuffle=False)

# Evaluate
evaluator = Evaluator(model=model, data_loader=midjourney_loader)
results = evaluator.evaluate()
```

### Threshold Optimization

Finds the optimal classification threshold:

```python
# Find threshold that maximizes F1 score
best_threshold, best_f1 = evaluator.optimize_threshold(metric="f1")
print(f"Best threshold: {best_threshold:.3f}, F1: {best_f1:.3f}")
```

### Robustness Testing

Evaluates model performance on transformed images:

```python
# Test robustness to JPEG compression
from src.evaluation.robustness import test_jpeg_robustness

jpeg_results = test_jpeg_robustness(
    model=model,
    data_loader=test_loader,
    quality_levels=[100, 80, 60, 40, 20]
)
```

## Visualizations

### ROC Curve

```python
from src.evaluation.visualization import plot_roc_curve

# Generate ROC curve
fig = evaluator.plot_roc_curve()
fig.savefig("results/figures/roc_curve.png")
```

### Confusion Matrix

```python
from src.evaluation.visualization import plot_confusion_matrix

# Generate confusion matrix visualization
fig = evaluator.plot_confusion_matrix(normalize=True)
fig.savefig("results/figures/confusion_matrix.png")
```

### Grad-CAM Visualization

```python
from src.evaluation.explainability import generate_gradcam

# Generate Grad-CAM for sample images
sample_images, sample_labels = next(iter(test_loader))
gradcam_images = generate_gradcam(
    model=model,
    images=sample_images[:8],
    target_layer="visual.transformer.resblocks.11"  # For CLIP ViT
)

# Save or display visualizations
```

## Batch Evaluation

For evaluating multiple models and comparing their performance:

```python
from src.evaluation.batch_evaluator import compare_models

models = {
    "ResNet-50": resnet_model,
    "CLIP Zero-Shot": clip_zero_shot_model,
    "CLIP Fine-tuned": clip_fine_tuned_model,
    "BLIP Fine-tuned": blip_model
}

comparison = compare_models(
    models=models,
    data_loader=test_loader,
    metrics=["accuracy", "auc", "f1"]
)

# Print comparison table
print(comparison)
```

## Evaluation Configuration

Example YAML configuration:

```yaml
evaluation:
  # Metrics to calculate
  metrics:
    - accuracy
    - auc
    - precision
    - recall
    - f1
    
  # Threshold settings
  threshold: 0.5  # Default classification threshold
  optimize_threshold: True  # Find optimal threshold
  threshold_metric: "f1"  # Metric to optimize
  
  # Output settings
  save_predictions: True
  predictions_file: "results/evaluation/predictions.csv"
  
  # Visualization
  generate_plots: True
  plot_dir: "results/figures"
  
  # Explainability
  generate_gradcam: True
  num_gradcam_samples: 16
  gradcam_layer: "visual.transformer.resblocks.11"
```

## Integration with main.py

The evaluation module is designed to be called from `main.py`:

```python
# In main.py
from src.evaluation.evaluator import Evaluator

def main(config):
    # Load model and dataset
    
    if config.mode == "eval":
        evaluator = Evaluator(
            model=model,
            data_loader=test_loader,
            **config.evaluation
        )
        results = evaluator.evaluate()
        
        # Save and display results
        evaluator.save_results(results, config.evaluation.results_file)
```

## Advanced Analysis

### Error Analysis

```python
# Get misclassified examples
misclassified = evaluator.get_misclassified_samples(n=10)

# Analyze and visualize
for image, true_label, pred_label, confidence in misclassified:
    # Perform detailed analysis
```

### Feature Analysis

```python
# Extract and analyze embeddings
embeddings, labels = evaluator.extract_embeddings(layer="visual.transformer.resblocks.11.attn")

# Perform dimensionality reduction (PCA or t-SNE)
from src.evaluation.visualization import plot_embeddings
plot_embeddings(embeddings, labels, method="tsne")
```

## Evaluation Across Datasets

For evaluating on multiple datasets:

```python
# Create datasets for different generators
datasets = {
    "Midjourney": midjourney_loader,
    "Stable_Diffusion": sd_loader,
    "DALL-E": dalle_loader
}

# Evaluate on each
results = {}
for name, loader in datasets.items():
    evaluator = Evaluator(model=model, data_loader=loader)
    results[name] = evaluator.evaluate()

# Compare performance across generators
```

## Notes on Usage

- For large test sets, consider evaluating in batches to manage memory
- Enable mixed precision during evaluation for faster inference
- Save detailed results for later analysis and visualization
- Use standardized seeds for reproducibility in evaluation 

### Inference with Trained CNN Model (New)

To perform inference using a previously trained CNN model (e.g., `best_model.pth` from a `train_cnn.py` run) on different datasets, you can use the `src/inference.py` script. This script loads the specified model and evaluates it on user-defined image folders, sampling a specified number of images from each.

**Usage:**

```bash
python src/evaluation/eval_cnn.py [OPTIONS]
```

**Key Options:**

*   `--model_path`: Path to the trained CNN model file (e.g., `results/cnn_output_base/resnet50_run1/checkpoints/best_model.pth`). Defaults to this path.
*   `--num_samples_per_folder`: Number of images to randomly sample from each subfolder (e.g., 'ai', 'nature', '1_fake', '0_real'). Defaults to 500.
*   `--seed`: Global random seed for reproducibility of image sampling. Defaults to 42.
*   `--gpu_id`: GPU ID to use for inference if CUDA is available (e.g., 0, 1). Defaults to 0.

The script is pre-configured to evaluate on the following dataset structures:
1.  `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0424_wukong/val/ai/` and `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0424_wukong/val/nature/`
2.  `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_glide/val/ai/` and `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_glide/val/nature/`
3.  `/raid/dannyliu/dataset/GAI_Dataset/Chameleon/test/0_real/` and `/raid/dannyliu/dataset/GAI_Dataset/Chameleon/test/1_fake/`

To use different datasets, you will need to modify the `datasets_to_evaluate` list within the `src/inference.py` script directly.

The script will output the accuracy for each subfolder, each dataset, and an overall accuracy across all processed images.

# Evaluation Scripts Readme

This directory contains scripts for evaluating trained models or specific model capabilities on various datasets.

## CNN Model Evaluation (`eval_cnn.py`)

This script evaluates a trained CNN model (e.g., ResNet50) on multiple image datasets to assess its performance in distinguishing AI-generated images from real (nature) images.

-   **Purpose**: To test a trained CNN classifier across different image datasets, which may vary in terms of generation models, content, or style.
-   **Configuration**: `configs/eval_cnn_config.yaml`
    -   Specifies the path to the pre-trained CNN model (`.pth` file).
    -   Defines general settings like seed and GPU ID.
    -   Lists multiple datasets for evaluation. Each dataset entry includes:
        -   A unique name for the dataset.
        -   Paths to folders containing AI-generated images (`ai_path`) and real/natural images (`nature_path`).
        -   The corresponding integer labels for AI (`ai_label`) and nature (`nature_label`) images (e.g., 0 for nature, 1 for AI).
    -   Sets the number of samples to randomly evaluate from each AI/nature folder within each dataset (`num_samples_per_folder`).
    -   Specifies a base directory for saving evaluation results (`output_base_dir`).
-   **Functionality**:
    1.  Loads the trained CNN model and sets it to evaluation mode.
    2.  Applies the same image transformations used during its training.
    3.  For each dataset defined in the configuration:
        -   Randomly samples the specified number of images from the AI and nature image folders.
        -   Performs inference on these sampled images.
        -   Calculates and prints accuracy for the AI folder, nature folder, and overall for the dataset.
    4.  Calculates and prints an overall accuracy across all processed samples from all datasets.
-   **Outputs**:
    -   Console output detailing accuracy for each folder and dataset, plus an overall summary.
    -   A JSON file named `evaluation_summary.json` saved in a timestamped subdirectory (e.g., `results/cnn_evaluations/config_name_TIMESTAMP/`). This file contains:
        -   Details of the configuration used (config file path, model path, seed, etc.).
        -   For each evaluated dataset: name, paths, labels, number of correct predictions, total samples, and accuracy for AI and nature folders, and overall dataset accuracy.
        -   An overall summary: total correct predictions, total samples, and overall accuracy across all datasets.
-   **Usage**:
    ```bash
    python src/evaluation/eval_cnn.py --config configs/eval_cnn_config.yaml
    ```

## Trained CLIP Linear Probe Evaluation (`eval_clip_linear_probe.py`)

This script evaluates a trained linear classifier (which was trained on top of frozen CLIP image embeddings) on multiple image datasets. It is designed to assess the generalization capability of the CLIP features combined with the simple learned classifier.

-   **Purpose**: To test the performance of a fine-tuned CLIP linear probe (CLIP backbone + trained linear classifier) on diverse image datasets beyond the one it was originally tested on during its training phase.
-   **Configuration**: `configs/eval_clip_linear_probe_config.yaml`
    -   Specifies the path to the trained linear classifier's state dictionary (`.pth` file) obtained from the `clip_linear_probe_train.py` script.
    -   Indicates the CLIP model ID (e.g., `openai/clip-vit-large-patch14`) that was used to extract features for training the linear probe. This ensures consistency.
    -   Includes the architecture configuration of the saved linear classifier (hidden dimensions, number of classes) to correctly reconstruct the model.
    -   Defines general settings like seed, GPU ID, and an output base directory.
    -   Lists multiple datasets for evaluation, similar to `eval_cnn.py`. Each dataset entry includes a name, paths to AI and nature image folders, and their corresponding integer labels.
    -   Sets parameters like the number of samples to evaluate per folder and batch sizes for embedding extraction and classifier inference.
-   **Functionality**:
    1.  Loads the specified pre-trained CLIP model (and its processor) and freezes its parameters.
    2.  Loads the trained linear classifier model and its saved weights.
    3.  For each dataset defined in the configuration:
        -   Optionally, randomly samples a specified number of images from the AI and nature image folders.
        -   Extracts CLIP image embeddings for the selected images in batches.
        -   Feeds these embeddings into the loaded linear classifier to get predictions.
        -   Calculates accuracy, generates a classification report (precision, recall, F1-score), and a confusion matrix for the dataset.
-   **Outputs**:
    -   Console output detailing performance for each dataset.
    -   A JSON file named `cross_dataset_evaluation_summary.json` saved in a timestamped subdirectory (e.g., `results/clip_linear_probe_evaluations/config_name_TIMESTAMP/`). This file contains:
        -   Details of the evaluation configuration (paths, model IDs, seed).
        -   For each evaluated dataset: its name, achieved accuracy, full classification report (as a dictionary), confusion matrix, and the number of samples processed.
-   **Usage**:
    ```bash
    python src/evaluation/eval_clip_linear_probe.py --config configs/eval_clip_linear_probe_config.yaml
    ```
    *Ensure the `linear_classifier_path` in the YAML config points to a valid trained model file.*

---

*More evaluation script details can be added here as the project evolves.*