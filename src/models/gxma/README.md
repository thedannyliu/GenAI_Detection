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

## Advanced Usage

### 1. Selective Frequency Extraction

`FrequencyFeatureExtractor` now supports **on-demand selection** of the three
available spectral descriptors.  In your YAML you may add:

```yaml
model:
  # Use only Radial FFT and DCT statistics
  freq_methods: ["radial", "dct"]
```

If the key is **omitted**, the extractor defaults to **`["radial", "dct", "wavelet"]`** (all three).
The same field is respected for both `GXMAFusionDetector` and
`FrequencyOnlyDetector`.

### 2. External Test Sets (`extra_tests`)

You can ask the training script to evaluate additional *out-of-distribution*
datasets right after training.  Example configuration snippet:

```yaml
evaluation:
  batch_size: 128
  extra_tests:
    - name: "chameleon_poc"
      base_data_dir: "/path/to/chameleon_poc"
      split_name: ""            # root contains ai/ & nature/
      class_to_idx: {nature: 0, ai: 1}
```

Metrics for each listed dataset are appended under
`results/.../training_results.json -> extra_tests` together with a confusion
matrix.

### 3. Reproducibility

The active YAML file is automatically copied to the run folder as
`config_used.yaml`, ensuring the exact hyper-parameters can be recovered.

### 4. GPU Memory Notes

* The CLIP vision encoder is loaded to the same device as the main detector
  (`model.to(device)`).  Because its parameters are frozen and wrapped in
  `torch.no_grad()`, the forward pass holds only weights (+ minor activations),
  so VRAM usage is modest (~2–3 GB for ViT-L/14).
* Spectral feature extraction (FFT/DCT/Wavelet) currently runs on **CPU** via
  NumPy/SciPy/PyWavelets.  Porting these kernels to CUDA would require
  replacing them with `torch.fft` / custom GPU ops and is left as future
  work.
* To fine-tune CLIP (thus increasing GPU utilisation) simply remove the
  `requires_grad = False` loop and the `torch.no_grad()` guard in
  `CLIPCLSExtractor`.
