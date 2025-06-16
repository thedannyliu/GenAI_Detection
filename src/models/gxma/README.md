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

For an example training setup see `configs/gxma_fusion_config.yaml` and the
training script `src/training/train_gxma.py`.
