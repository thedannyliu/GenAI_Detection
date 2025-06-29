# GXMA – PoC Stage-1 Configuration Guide

This folder contains **ready-to-run YAML files** for the GXMA Fusion Detector
experiments described in the PoC-V1 roadmap.  Each file encapsulates a
complete training recipe – simply point the training script to the YAML and
select the desired *mode* (fusion / frequency / semantic).

## 1. Config Matrix

| File | Fusion Strategy | CLIP | LoRA | Description |
|------|-----------------|------|------|-------------|
| `gxma_fusion_config.yaml` | single (Tier-1) | frozen | ✗ | Baseline 1-to-1 cross-attention (FFT + Semantics) |
| `gxma_parallel_fusion_config.yaml` | parallel (Tier-2) | frozen | ✗ | Parallel attention streams (FFT / DCT / Wavelet) summed |
| `gxma_parallel_endtoend_finetune.yaml` | parallel (Tier-2) | trainable | ✓ | Parallel streams **+ LoRA** fine-tuning of CLIP |
| `gxma_hierarchical_endtoend_finetune.yaml` | hierarchical (Tier-2 ✕ Meta-Gate) | trainable | ✓ | Parallel streams + **Meta-Gate** + LoRA |
| `gxma_endtoend_finetune.yaml` | single (Tier-1) | trainable | ✓ | 1-to-1 fusion with LoRA fine-tuning |
| `gxma_sem_only_config.yaml` | — | frozen | ✗ | Semantic-only ablation (uses CLIP CLS vector only) |

> Tip The *frequency-only* ablation does **not** need a separate YAML – simply
> set `--mode frequency` when using any fusion config.

## 2. Running Examples

```bash
# Tier-1 baseline (single-stream, frozen CLIP)
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_fusion_config.yaml \
  --mode fusion

# Tier-2 parallel streams (frozen CLIP)
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml \
  --mode fusion

# Tier-2 parallel streams + LoRA fine-tune
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_endtoend_finetune.yaml \
  --mode fusion

# Tier-2 hierarchical Meta-Gate + LoRA fine-tune
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_hierarchical_endtoend_finetune.yaml \
  --mode fusion

# Semantic-only ablation (shared hyper-params)
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_sem_only_config.yaml \
  --mode semantic

# Frequency-only ablation
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_fusion_config.yaml \
  --mode frequency
```

## 3. Customising a Config

1. **Dataset paths** – edit `data.base_data_dir` (and optional `class_to_idx`).
2. **GPU selection** – modify `general.gpu_id`.
3. **Batch size / workers** – `data.batch_size`, `data.num_workers`.
4. **LoRA hyper-parameters** – `model.lora.*` (set `enable: false` to disable).
5. **Early stopping** – `training.early_stopping.*`.

All other defaults (optimiser, scheduler, AMP, checkpointing) are handled by
the training script.  For more advanced overrides refer to
`src/models/gxma/README.md`. 

## 4. YAML Field Reference

Below tables explain key parameters you may wish to tweak.  Fields not listed
here generally retain robust defaults in the training script.

### general
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| output_dir | str | "results/gxma_runs" | Root folder for all experiment outputs |
| experiment_name | str | (required) | Sub-folder under `output_dir` for this run |
| seed | int | 42 | Global RNG seed (Python / NumPy / PyTorch) |
| gpu_id | int | 0 | Index of CUDA device to use |

### model
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| hidden_dim | int | 256 | Hidden size H for attention & classifier |
| num_heads | int | 4 | Heads in each `nn.MultiheadAttention` |
| num_classes | int | 2 | Classes for final MLP |
| freq_methods | list[str] | ["radial","dct","wavelet"] | Spectral descriptors to extract |
| fusion_strategy | str | "single" | "single", "parallel", or "hierarchical" |

#### model.lora (optional)
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| enable | bool | false | Enable LoRA on CLIP ViT |
| r | int | 8 | LoRA rank |
| alpha | int | 16 | Scaling factor (≈ 2×r recommended) |
| dropout | float | 0.1 | LoRA dropout |
| target_modules | list[str] | ["q_proj","v_proj"] | Layers to inject adapters into |

### data
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| base_data_dir | str | (required) | Dataset root containing split folders |
| train_split_name | str | "train" | Sub-folder name for training images |
| val_split_name | str | "val" | Sub-folder name for validation images |
| test_split_name | str | "test" | Sub-folder name for test images |
| train_samples_per_class | int\|null | null | Randomly subsample per class (null → all) |
| batch_size | int | 128 | Batch size per GPU |
| num_workers | int | 4 | DataLoader worker threads |
| class_to_idx | dict | {nature:0, ai:1} | Label mapping if folder names differ |

### training
| Key | Type | Default | Description |
|-----|------|---------|-------------|
| optimizer | str | "adamw" | Currently supports AdamW only |
| learning_rate | float | 1e-4 | Initial LR |
| weight_decay | float | 0.0 | L2 regularisation |
| num_epochs | int | 2000 | Max training epochs |

#### training.scheduler
| Key | Default | Description |
|-----|---------|-------------|
| name | "cosine_with_warmup" | Scheduler type (omit for constant LR) |
| warmup_steps | 15 | Warm-up steps before cosine decay |

#### training.early_stopping
| Key | Default | Description |
|-----|---------|-------------|
| monitor | "val_auc" | Metric to watch (val_acc / val_auc …) |
| patience | 40 | Stop if no improvement for X epochs |
| threshold | 0.001 | Minimum metric gain to reset patience |

### evaluation
| Key | Default | Description |
|-----|---------|-------------|
| batch_size | Same as data.batch_size | Evaluation batch size |
| extra_tests | list | [] | Additional OOD datasets; see YAML example |

For any advanced field not covered, consult inline comments in
`train_gxma.py` or the main project README. 