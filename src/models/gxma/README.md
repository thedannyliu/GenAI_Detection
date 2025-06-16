# GXMA Fusion Detector

This module implements a proof of concept for the **GXMA-Fusion Detector**. It
combines frequency based fingerprints with vision-language model semantics
via a simple cross-attention mechanism and finishes with a lightweight MLP
classifier.

## Components

- **frequency_extractors.py** – Contains implementations of three frequency
  feature extraction methods:
  1. Radial Average Spectrum
  2. Block DCT coefficient statistics
  3. Wavelet coefficient histogram
  The extracted vectors are concatenated into a 256‑dimensional descriptor.
- **clip_semantics.py** – Provides `CLIPCLSExtractor` which extracts the global
  `[CLS]` token from OpenAI's CLIP ViT-L/14 model.
- **gxma_fusion_detector.py** – Defines the `GXMAFusionDetector` model that
  fuses the frequency descriptor and CLIP semantics using cross attention and
  predicts real vs. AI-generated images.

The design is intentionally modular so future frequency methods or different
VLM embeddings can be swapped in easily.

## Training

To train the GXMA Fusion Detector, use the `train_gxma.py` script located in `src/training`. You need to provide a configuration file.

```bash
python src/training/train_gxma.py --config configs/gxma_fusion_config.yaml
```

### Configuration

The training process is controlled by a YAML configuration file (`configs/gxma_fusion_config.yaml`). Key parameters include:

- **general**: `output_dir`, `experiment_name`, `seed`, `gpu_id`.
- **model**: `hidden_dim`, `num_heads`, `num_classes` for the detector architecture.
- **data**: `base_data_dir`, sample counts, `batch_size`, `num_workers`.
- **training**: `num_epochs`, `learning_rate`, `optimizer`, and settings for:
    - **scheduler**: Learning rate scheduler (e.g., `cosine_with_warmup`) with `warmup_steps`.
    - **early_stopping**: `monitor` (`val_acc` or `val_loss`), `patience`, and `threshold` to prevent overfitting.

The script will train the model, perform validation at each epoch, save the best model based on the monitored metric, and finally evaluate the best model on the test set. Results, including training history and the final configuration, are saved in the specified output directory.

## Ablation Study

To understand the contribution of each component, two lightweight variants are provided in `src/models/gxma/ablation_detectors.py`:

| Variant | Description | CLI flag |
|---------|-------------|----------|
| **FrequencyOnlyDetector** | Uses the 256-dimensional concatenated frequency descriptor only, followed by an MLP. | `--mode frequency` |
| **SemanticOnlyDetector**  | Uses the CLIP CLS vector only (e.g. 1024-d for ViT-L/14), followed by an MLP. | `--mode semantic` |

All training hyper-parameters, scheduler, and early-stopping behaviour remain identical to the full GXMA Fusion model.

### Running the ablations

```bash
# Full fusion (default)
python src/training/train_gxma.py --config configs/gxma_fusion_config.yaml --mode fusion

# Frequency-only baseline
python src/training/train_gxma.py --config configs/gxma_fusion_config.yaml --mode frequency

# Semantic-only baseline
python src/training/train_gxma.py --config configs/gxma_fusion_config.yaml --mode semantic
```

> Tip: Use different `experiment_name` values in the YAML (e.g. `gxma_fusion_v1`, `gxma_freq_only`, `gxma_sem_only`) to keep results in separate folders.
