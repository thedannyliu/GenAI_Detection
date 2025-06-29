# GXMA Fusion Detector

This module implements a proof of concept for the **GXMA-Fusion Detector**. It
combines frequency based fingerprints with vision-language model semantics
via a simple cross-attention mechanism and finishes with a lightweight MLP
classifier.

> 2025-06-16  Update — **Strategy B: Parallel Attention Streams** has been implemented.  See
> `ParallelCrossAttentionFusion` in `gxma_fusion_detector.py` and the new YAML
> configs described below.

> 2025-06-29  **Performance v1.1** — Dataset-side frequency extraction, 8-worker DataLoader, `float16` CLIP + AMP and new `forward(images, freq_feat)` API deliver ~2-4× faster epochs on A100-40GB while preserving metrics.

The design is intentionally modular so future frequency methods or different
VLM embeddings can be swapped in easily.

## New: Parallel Attention Streams (Strategy B)

In addition to the original *1-to-1* cross-attention (single K/V stream), the
detector now supports **Parallel Attention Streams** where *one attention head
is built for each frequency descriptor* (Radial FFT, DCT, Wavelet).  The CLIP
semantic vector is used as a *shared Query* while each frequency vector is
projected to its own Key/Value space:

```
             ┌──────────────────────────┐
   F_radial ─►  Attn(q, k_r, v_r) ─┐    │
             ├─────────────────────┤    │  summed ➜ fused repr.
   F_dct    ─►  Attn(q, k_d, v_d) ─┼──► Σ │
             ├─────────────────────┤    │
   F_wavelet ─►  Attn(q, k_w, v_w) ─┘    │
             └──────────────────────────┘
```

Output vectors from each stream are **summed** (default) or concatenated
(aggregated="concat") before the MLP classifier.  This yields a stronger yet
light-weight fusion without introducing heavy transformer blocks.

To enable the parallel strategy simply add the following to your YAML config
under the `model:` section:

```yaml
model:
  fusion_strategy: "parallel"  # "single" (default) or "parallel"
```

Two ready-to-run configs are provided:

| Config Path | Description |
|-------------|-------------|
| `configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml` | Parallel fusion, frozen CLIP |
| `configs/gxma/poc_stage1/gxma_parallel_endtoend_finetune.yaml` | Parallel fusion + LoRA fine-tuning of CLIP |

Launch example:

```bash
# Parallel fusion with frozen CLIP
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml \
  --mode fusion

# Parallel fusion + LoRA fine-tuning (batch size 64)
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_endtoend_finetune.yaml \
  --mode fusion
```

**Note:** Frequency extraction is still executed on CPU via NumPy/ SciPy /
PyWavelets.  The additional attention streams therefore add negligible GPU
memory overhead compared to the single-stream version.

## Components

- **frequency_extractors.py** – Contains implementations of three frequency
  feature extraction methods:
  1. Radial Average Spectrum
  2. Block DCT coefficient statistics
  3. Wavelet coefficient histogram
  The extracted vectors are concatenated into a 256-dimensional descriptor.
- **clip_semantics.py** – `CLIPCLSExtractor` loads the vision tower of
  OpenAI CLIP ViT-L/14 and optionally injects *LoRA* adapters via PEFT.
- **gxma_fusion_detector.py** – Houses both fusion strategies:
  - `CrossAttentionFusion` (*Strategy A*, 1-to-1)
  - `ParallelCrossAttentionFusion` (*Strategy B*, 1-to-3)
  plus the high-level `GXMAFusionDetector` wrapper.

## Training Quick-Start

See the *Configuration* section above for details.  In short:

```bash
# Single-stream baseline (Tier-1)
python src/training/train_gxma.py --config configs/gxma/poc_stage1/gxma_fusion_config.yaml --mode fusion

# Parallel streams (Tier-2, current SOTA baseline in this repo)
python src/training/train_gxma.py --config configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml --mode fusion
```

All configs inherit sane defaults; feel free to modify `hidden_dim`, learning
rate, dataset paths, etc.  Each run will create
`results/gxma_runs/<experiment_name>/` containing checkpoints, TensorBoard
logs and JSON metrics.

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

* The CLIP vision encoder now defaults to **`float16` + AMP** when CUDA is
  available, reducing VRAM by ~40 % and offering ~1.3 × inference speed-up.
* Spectral features are **pre-computed by DataLoader workers** (CPU) and passed
  as a single tensor, avoiding costly GPU→CPU copies during training.
* On-the-fly extraction logic remains as a fallback when `freq_feat` is not
  supplied (e.g., for legacy inference scripts).
