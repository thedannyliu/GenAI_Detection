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