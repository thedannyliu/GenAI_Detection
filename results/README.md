# Results Directory

This directory stores outputs from experiments, including saved models, logs, evaluation metrics, and visualizations. It is organized by experiment type to keep results clearly separated.

## Directory Structure

```
results/
├── logs/                 <- Training logs and TensorBoard files
├── saved_models/         <- Model checkpoints
├── evaluation/           <- Evaluation reports and metrics
├── figures/              <- Plots, visualizations, and Grad-CAM outputs
├── zero_shot/            <- Results specific to zero-shot experiments
├── fine_tuned/           <- Results from fine-tuned VLM experiments
├── param_efficient/      <- Results from prompt tuning and adapter experiments
└── cnn_baseline/         <- Results from CNN baseline models
```

## Contents

### `logs/`

Contains training logs in various formats:
- Text logs (`.log`) with progress and metrics
- TensorBoard event files for visualization
- CSV files with per-epoch metrics

### `saved_models/`

Stores model checkpoints:
- `{model_name}_best.pth`: Best model according to validation performance
- `{model_name}_epoch_{N}.pth`: Checkpoints at specific epochs
- `{model_name}_final.pth`: Model after training completion

### `evaluation/`

Contains evaluation outputs:
- `{model_name}_metrics.json`: Overall performance metrics
- `{model_name}_predictions.csv`: Per-sample predictions
- `{model_name}_confusion_matrix.csv`: Confusion matrix data
- Cross-generator evaluation reports

### `figures/`

Visualizations generated during training and evaluation:
- Learning curves (loss, accuracy)
- ROC curves and precision-recall curves
- Confusion matrices
- Grad-CAM visualizations showing model attention
- t-SNE or PCA plots of embeddings

## Experiment-Specific Directories

Each experiment type has its own subdirectory (e.g., `zero_shot/`, `fine_tuned/`) with the same internal structure as above. This organization keeps results from different approaches separate for easier comparison and analysis.

## Notes

- Large files in this directory are not tracked in Git
- The directory structure is automatically created as needed during experiments
- For reproducibility, experiment configurations are stored alongside results
- Visualization code can be found in notebooks that use these outputs 