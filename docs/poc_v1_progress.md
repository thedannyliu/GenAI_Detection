# GXMA PoC – Progress Log (v1.0)

> Last update: **2025-06-29**

This document tracks the implementation status of the **GXMA – Frequency ✕ Semantics Fusion** proof-of-concept.  All notes below are written in **English** to facilitate paper drafting.

---

## 0. Goals at a Glance
1. **Hypothesis:** *"Combining frequency fingerprints with VLM semantics via attention can effectively detect AI-generated images."*
2. Scope of PoC-V1:
   • **Frequency**: Radial FFT, DCT statistics, Wavelet statistics  
   • **Semantics**: CLIP ViT-L/14 `[CLS]` token (optionally LoRA-fine-tuned)  
   • **Fusion**: Tier-1 (*single* cross-attention) and Tier-2 **Parallel Attention Streams**.

---

## 1. Completed ✅
• Frequency extractors (`frequency_extractors.py`) – 256-dim concatenated vector.  
• Semantic extractor (`clip_semantics.py`) with optional PEFT-LoRA.  
• Tier-1 fusion (`CrossAttentionFusion`).  
• Training pipeline (`train_gxma.py`) with early-stopping & extra-test evaluation.  
• Baseline YAML configs (`gxma_sem_only_config.yaml`, `gxma_fusion_config.yaml`, `gxma_endtoend_finetune.yaml`).

### New in v1.0 (this commit)
• **ParallelCrossAttentionFusion** implemented (Tier-2).  
• Detector now supports `fusion_strategy: parallel`.  
• Two new configs:  
  1. `gxma_parallel_fusion_config.yaml` (frozen CLIP)  
  2. `gxma_parallel_endtoend_finetune.yaml` (LoRA fine-tune).
• **Resume / Checkpointing** – training now saves `last.pth` (full state) every epoch and can resume via `--resume <path>`.
• **Auto-merge logs** – when resuming, the script now (i) loads the previous `training_results.json` to append new epoch metrics, (ii) re-uses `config_used.yaml` from the run directory when `--config` is omitted.  TensorBoard continues seamlessly.
• **Perf v1.1** – Speed-oriented refactor.
    1. **CPU → DataLoader workers**: frequency vectors computed inside each worker; no more per-batch CPU fallback.
    2. **GPU Copy Removed**: pre-computed frequency tensor is forwarded directly, eliminating `img.cpu()` round-trips.
    3. **I/O Pipeline**: default `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=4`.
    4. **CLIP Vision Encoder**: runs in `float16` with `torch.cuda.amp.autocast()` for inference; LoRA path supported.
    5. **Vectorised Forward**: `GXMAFusionDetector.forward(images, freq_feat)` signature keeps backward-compatibility when `freq_feat` is `None`.
    6. **Config-free**: all optimisations activate automatically; YAML can still override `data.num_workers` or disable `include_freq_features` for ablation.

---

## 2. TODO 📝
| Status | Item |
|--------|------|
| ⬜ | **Meta-Gate (Strategy C)** – softmax weights *g1-g3* on top of parallel streams |
| ⬜ | **Patch-level semantics** – feed CLIP patch tokens + Transformer decoder (Tier-3) |
| ⬜ | Expand datasets & ablation notebooks (Florence-2, unseen generators) |
| ⬜ | Inference CLI for single-image detection |
| ⬜ | Full docstrings, type hints & unit tests |

---

## 3. Milestones
| Version | Date | Highlights |
|---------|------|------------|
| v0.1 | 2025-06-05 | Frequency & Semantic extractors + Tier-1 fusion |
| v1.0 | 2025-06-16 | **Parallel Attention Streams, updated docs & configs** |

---

> Feel free to append progress updates directly in this file and open a PR. 