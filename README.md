# Detecting AI-Generated Images using Vision-Language Models (VLMs) on the genimage Dataset

## Project Overview

This repository contains the implementation of our research project on detecting AI-generated images using Vision-Language Models (VLMs). We leverage large-scale VLMs such as CLIP, BLIP, and Flamingo to detect AI-created images in the genimage benchmark, a million-scale dataset of real and fake image pairs.

## Key Features

- Zero-shot classification using pre-trained VLMs
- Fine-tuning strategies for improved detection
- Prompt engineering and analysis
- Parameter-efficient adaptation methods
- Prompt tuning with InstructBLIP via `src/training/prompt_tuning_instructBLIP.py`
- Comprehensive evaluation metrics
- Explainability analysis using Grad-CAM
- Robustness testing against image manipulations

## Repository Structure

```
project-root/
├── README.md                        <- Project overview and setup instructions
├── LICENSE                          <- Project license
├── requirements.txt                 <- Python package requirements
├── configs/                         <- YAML configuration files
├── data/                           <- Dataset directory
├── docs/                           <- Documentation
├── notebooks/                      <- Jupyter notebooks
├── results/                        <- Experiment outputs
├── src/
│   ├── data_processing/    # Scripts for dataset handling, preprocessing, augmentation
│   ├── evaluation/         # Scripts for evaluating trained models
│   ├── experiments/        # Main zero-shot experiment scripts (and older/other experiments)
│   ├── models/             # Model definitions (CNNs, VLMs wrappers, etc.)
│   │   └── gxma/           # GXMA-Fusion Detector implementation
│   ├── prompts/            # Prompt engineering strategies and definitions
│   ├── training/           # Scripts for model training (e.g. CNNs, VLM fine-tuning, linear probes)
│   └── utils/              # Utility functions (config loading, logging, etc.)
├── tests/                          <- Test suite
```

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/genimage-vlm-detection.git
   cd genimage-vlm-detection
   ```

2. **Create Environment:**
   ```bash
   conda create -n vlm-detection python=3.9 -y
   conda activate vlm-detection
   ```

3. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Dataset:**
   Follow instructions in `data/README.md` to download and prepare the genimage dataset.

5. **Dataset Notes:**
   - **General Structure**: Most datasets are expected to follow a structure like `dataset_root/split_name/class_name/image.jpg`, where `class_name` is typically `nature` or `ai`.
   - **Chameleon Dataset**: The Chameleon dataset has a slightly different structure for its class folders, using `0_real` and `1_fake` instead of `nature` and `ai` (e.g., `/raid/dannyliu/dataset/GAI_Dataset/Chameleon/test/0_real/...`). 
   - **Dataset Loading (`GenImageDataset`)**: The `src.data_processing.custom_dataset.GenImageDataset` class has been updated to handle these variations. It now dynamically determines the class subdirectories based on the `class_to_idx` mapping provided in the dataset configuration (e.g., in `configs/vlm_zero_shot_custom.yaml`). The keys from `class_to_idx` (like `"0_real"` or `"nature"`) are used to locate the respective class folders. If `class_to_idx` is not specified for a dataset, it defaults to looking for `nature` and `ai` folders.

6. **Configuration:**
   Review and adjust YAML files in `configs/` as needed.

## Usage

### Training a Model

**For VLM Models (Fine-tuning InstructBLIP):**

To fine-tune an InstructBLIP model (e.g., `Salesforce/instructblip-vicuna-7b`) for AI-generated image detection, use the `train_instructblip.py` script.
(Details as previously, ensure paths in `src/training/train_instructblip.py` and configs are correct)

To run the training:
1.  **Prepare your configuration file**: (e.g., `configs/my_instructblip_vicuna_train_config.yaml`)
2.  **Run the training script**:
    ```bash
    python src/training/train_instructblip.py --config configs/my_instructblip_vicuna_train_config.yaml
    ```

**Prompt Tuning with InstructBLIP:**

To adapt InstructBLIP using a small set of learnable prompt tokens:
1.  Prepare a configuration (e.g., `configs/prompt_tuning_instructblip_config.yaml`).
2.  Run the prompt tuning script:
    ```bash
    python src/training/prompt_tuning_instructBLIP.py --config configs/prompt_tuning_instructblip_config.yaml
    ```

**For CNN-based Classification:**

To train a ResNet50-based CNN classifier, use the `train_cnn.py` script with a YAML configuration file (e.g., `configs/cnn_baseline.yaml`).
```bash
python src/training/train_cnn.py --config configs/cnn_baseline.yaml
```

**For CLIP Linear Probing Training:**

This approach trains a linear classifier on top of frozen image embeddings extracted from a pre-trained CLIP model.
1.  **Configure**: Edit `configs/clip_linear_probe_config.yaml` to set dataset paths, model ID, sampling parameters, and training settings.
2.  **Run training script**:
    ```bash
    python src/training/clip_linear_probe_train.py --config configs/clip_linear_probe_config.yaml
    ```
    This will save the best linear classifier, training history, and test set evaluation results.

**For GXMA Fusion Detector Training:**

The GXMA detector now supports two fusion modes:

| Tier | Fusion Strategy | YAML Config | Notes |
|------|-----------------|-------------|-------|
| 1 | Single cross-attention | `configs/gxma/poc_stage1/gxma_fusion_config.yaml` | Baseline |
| 2 | **Parallel Attention Streams** | `configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml` | New default |
| 2-FT | Parallel + LoRA fine-tune | `configs/gxma/poc_stage1/gxma_parallel_endtoend_finetune.yaml` | End-to-end training |

Run examples:

```bash
# Tier-1 baseline
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_fusion_config.yaml \
  --mode fusion

# Tier-2 parallel fusion (frozen CLIP)
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_fusion_config.yaml \
  --mode fusion

# Tier-2 with LoRA fine-tuning
python src/training/train_gxma.py \
  --config configs/gxma/poc_stage1/gxma_parallel_endtoend_finetune.yaml \
  --mode fusion
```

Each run dumps checkpoints, TensorBoard logs, and JSON metrics to
`results/gxma_runs/<experiment_name>/`.

### C-VFiD (Hierarchical Router + Multi-Expert)

We provide a lightweight training script that fine-tunes **C-VFiD** on the AIGC Detection Benchmark.

```bash
# Example (single-GPU)
python -m src.training.train_cvfid \
    --train_dir AIGCDetectionBenchMark/progan_train \
    --val_dir   AIGCDetectionBenchMark/progan_val \
    --output_dir results/cvfid_run1 \
    --batch_size 16 \
    --epochs 5 \
    --gating_mode sigmoid  # multi-hot router
```

The script will:
1. Load images using `BenchmarkImageDataset` (a thin wrapper of `torchvision.datasets.ImageFolder`).
2. Initialise `RvfidModel` with hierarchical query head, sigmoid router, and three low-level experts.
3. Train with AdamW, save the **best** checkpoint (highest validation AUC) under `<output_dir>/best_cvfid.pt`.
4. Dump per-epoch metrics to `<output_dir>/history.json`.

Feel free to adjust `--num_experts`, learning rate, and other hyper-parameters.

### Evaluation

**Zero-shot VLM Evaluation (Custom Script):**

For flexible zero-shot evaluation of various VLMs (CLIP, BLIP, InstructBLIP, LLaVA, etc.):
1.  **Configure**: Create or modify a YAML configuration file (e.g., `configs/vlm_zero_shot_custom.yaml`).
2.  **Run evaluation script**:
    ```bash
    python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml
    ```

**Evaluating a Fine-tuned VLM Model (InstructBLIP):**

To evaluate a fine-tuned InstructBLIP model:
1.  Ensure trained model artifacts exist.
2.  Configure `src/evaluation/eval_instructblip.py` with the correct `RUN_ID`.
3.  **Run evaluation script**:
    ```bash
    python src/evaluation/eval_instructblip.py
    ```

**Evaluating a Trained CNN Model:**

1.  **Prepare configuration**: Create or modify `configs/eval_cnn_config.yaml` specifying the model path and datasets.
2.  **Run evaluation script**:
    ```bash
    python src/evaluation/eval_cnn.py --config configs/your_eval_cnn_config.yaml
    ```

**Evaluating a Trained CLIP Linear Probe:**

To evaluate a trained CLIP linear probe on various datasets:
1.  **Update Configuration**: After running the CLIP linear probe training, note the path to the saved `best_linear_classifier.pth`. Open `configs/eval_clip_linear_probe_config.yaml` and update the `linear_classifier_path` field with this correct path.
2.  **Configure Datasets**: Ensure the `datasets` section in `configs/eval_clip_linear_probe_config.yaml` lists all datasets you want to test against.
3.  **Run evaluation script**:
    ```bash
    python src/evaluation/eval_clip_linear_probe.py --config configs/eval_clip_linear_probe_config.yaml
    ```

### Running Tests

```bash
pytest
```

## Documentation

- Detailed experiment design: `docs/experiment_plan.md`
- Code design: `docs/code_plan.md`
- Dataset instructions: `data/README.md`
- Training script details: `src/training/README.md`
- Evaluation script details: `src/evaluation/README.md`
- Experiment (Zero-shot) script details: `src/experiments/README.md`

## Contributing

Please read our contribution guidelines in `CONTRIBUTING.md` before submitting pull requests.

## License

[License to be determined]

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{your-paper,
  title={Detecting AI-Generated Images using Vision-Language Models},
  author={Your Name and Co-authors},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

We thank the creators of the genimage dataset and the open-source community for their valuable contributions.

## CLIP Linear Probing (Consolidated from previous duplicate content)

-   **Description**: This approach involves using a pre-trained CLIP model, freezing its weights, and training only a simple linear classifier (or a shallow MLP) on top of the image embeddings extracted from CLIP. It's an efficient way to adapt powerful pre-trained models to specific tasks like AI vs. Nature image detection.
-   **Methodology**:
    1.  **Feature Extraction**: Use the frozen image encoder of a CLIP model (e.g., `openai/clip-vit-large-patch14`) to generate fixed-size vector embeddings for input images.
    2.  **Classifier Training**: Train a linear classifier using these embeddings as input and the image labels (AI-generated or Nature) as targets.
-   **Scripts**:
    -   Training Linear Probe: `src/training/clip_linear_probe_train.py`
        -   Configuration: `configs/clip_linear_probe_config.yaml`
        -   This script handles data sampling, CLIP feature extraction, training the linear classifier with early stopping, and evaluating on a held-out test set.
    -   Evaluating Trained Linear Probe: `src/evaluation/eval_clip_linear_probe.py`
        -   Configuration: `configs/eval_clip_linear_probe_config.yaml`
        -   This script takes the trained linear classifier and evaluates its performance on multiple, potentially different, datasets to test generalization.
-   **Key Features**: Computationally efficient fine-tuning, leverages strong CLIP features, allows for robust evaluation across datasets.
-   **Further Details**: See documentation within `src/training/README.md` and `src/evaluation/README.md`.

## Troubleshooting and Setup Notes for InstructBLIP LoRA 8-bit Fine-tuning

This section documents the steps and issues encountered while setting up LoRA fine-tuning for the `Salesforce/instructblip-vicuna-7b` model with 8-bit quantization, aiming to enable training on a single GPU with limited memory (e.g., 40GB A100).

### Initial Problem
The primary goal was to enable a previously中断 (interrupted) training process (`src/training/train_instructblip.py`) to run reliably from start to finish, especially for large datasets and epochs. The initial attempts with full precision or even standard LoRA with `bfloat16` on a 40GB A100 GPU led to CUDA Out-of-Memory (OOM) errors, even with very small batch sizes.

### Debugging Steps and Solutions Attempted

1.  **Reducing Data and Training Scale:**
    *   Modified `configs/train_instructblip_config.yaml`:
        *   `data.num_train_samples`, `data.num_val_samples`, `data.num_test_samples` were significantly reduced (e.g., to 10).
        *   `training.epochs` reduced (e.g., to 2).
        *   `training.early_stopping_patience` reduced.
    *   **Outcome:** Still encountered OOM errors, indicating the base model size was the main issue.

2.  **Investigating `DynamicCache` Error with `pad_across_processes`:**
    *   An error `TypeError: Unsupported types (<class 'transformers.cache_utils.DynamicCache'>) passed to \`_pad_across_processes\`` occurred during evaluation.
    *   **Solution Attempt 1:** Ensured `model.config.use_cache = False` was set after loading the model.
    *   **Solution Attempt 2:** Overrode `prediction_step` in the custom `Trainer` to ensure only logits tensors were passed, not complex model outputs containing `DynamicCache`.
    *   **Outcome:** These changes helped resolve the `DynamicCache` error, but OOM errors persisted.

3.  **Addressing `AttributeError: 'TrainingArguments' object has no attribute 'predict_with_generate'`:**
    *   This error appeared in the custom `prediction_step` due to an incorrect check for `self.args.predict_with_generate`.
    *   **Solution:** Simplified `prediction_step` to rely on `compute_loss` for obtaining logits, removing the problematic attribute check.
    *   **Outcome:** Resolved the `AttributeError`, but OOM remained the primary blocker.

4.  **CUDA Out of Memory (OOM) on Specified GPU:**
    *   Despite setting `CUDA_VISIBLE_DEVICES=2` and the script confirming usage of the target GPU, OOM errors (referring to PyTorch's internal `GPU 0`, which was correctly mapped to the physical GPU 2) persisted.
    *   **Initial thought:** Confusion about GPU indexing.
    *   **Clarification:** The script *was* running on the correct physical GPU (GPU 2), but the model itself, even with a batch size of 10 or 4, was too large for 40GB VRAM in `bfloat16`.
    *   **Solution Attempts:**
        *   Reduced `training.batch_size` in `configs/train_instructblip_config.yaml` progressively from 16 down to 4, then to 1.
        *   Set environment variable `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
        *   Ensured `environment.gpu_id` in the config also pointed to the correct GPU ID (e.g., 2) as a fallback.
    *   **Outcome:** Even with `batch_size: 1`, OOM errors continued, indicating the need for more aggressive memory-saving techniques. The error traces often pointed to operations like `logits.float()` during loss computation or `softmax` within model components.

5.  **Introducing 8-bit Quantization with `bitsandbytes`:**
    *   This was identified as a key strategy to significantly reduce memory footprint.
    *   **Modification 1 (Initial 8-bit setup):**
        *   Added `bitsandbytes` to requirements.
        *   In `src/training/train_instructblip.py`, modified model loading:
            ```python
            model = InstructBlipForConditionalGeneration.from_pretrained(
                model_config['name_pretrained'],
                low_cpu_mem_usage=True,
                device_map="auto",
                load_in_8bit=True
            )
            ```
    *   **Encountered `ValueError: You cannot perform fine-tuning on purely quantized models...`**:
        *   This error arises because the `Trainer` detects a fully quantized model but expects PEFT adapters for fine-tuning.
    *   **Modification 2 (Adding `prepare_model_for_kbit_training`):**
        *   Imported `prepare_model_for_kbit_training` from `peft`.
        *   Called `model = prepare_model_for_kbit_training(model)` after loading the 8-bit model and before applying LoRA.
        *   Initially included `use_gradient_checkpointing` argument, then removed it as it's deprecated and handled by `TrainingArguments`.
    *   **Encountered `NameError: name 'peft' is not defined`**:
        *   Caused by an incorrect `hasattr(peft, ...)` check.
        *   **Solution:** Removed the `hasattr` check and called `prepare_model_for_kbit_training` directly.
    *   **Outcome (`ValueError` persisted):** The `ValueError` about fine-tuning purely quantized models remained even after `prepare_model_for_kbit_training`. Debug logs showed:
        *   `Is model a PEFT model after LoRA? False`
        *   `Is model.language_model a PEFT model after LoRA? True`
        This indicated that the top-level `model` object passed to the `Trainer` was not recognized as a `PeftModel`, even though its `language_model` sub-component was correctly wrapped by PEFT.

6.  **Explicitly Setting `requires_grad = False` for Non-LoRA Parameters:**
    *   To further ensure the `Trainer` understands that only LoRA adapters are trainable:
        ```python
        if model_config['finetune_method'] == 'lora':
            # ... (apply LoRA to model.language_model) ...
            logger.info("Iterating over model parameters to set requires_grad for non-LoRA parts...")
            for name, param in model.named_parameters():
                if '.lora_' not in name:
                    param.requires_grad = False
            logger.info("Finished setting requires_grad for non-LoRA parameters.")
        ```
    *   **Outcome (`ValueError` persisted):** This explicit step, while good practice, did not resolve the `Trainer`'s `ValueError`.

7.  **Current Approach: Using `BitsAndBytesConfig` for Quantization (Following Deprecation Warnings):**
    *   The `load_in_8bit` argument is deprecated. The recommended way is to use `quantization_config` with a `BitsAndBytesConfig` object.
    *   Modified model loading in `src/training/train_instructblip.py`:
        ```python
        from transformers import BitsAndBytesConfig
        # ...
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_8bit_compute_dtype=torch.bfloat16 # Can be added if mixed precision compute is desired
        )
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_config['name_pretrained'],
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        ```
    *   **Status:** This is the current state. The next step is to test this configuration to see if it resolves the `ValueError` and allows training to proceed without OOM errors.

### Key Learnings and Next Steps

*   Fine-tuning large models like InstructBLIP-Vicuna-7B requires aggressive memory optimization, with 8-bit quantization being essential for single GPU (40GB) operation.
*   The interaction between `transformers.Trainer`, PEFT-modified models (especially when only a sub-module is adapted), and k-bit quantization can be tricky. The `Trainer` needs to correctly identify that the model, despite its base being quantized, has trainable adapter layers.
*   Ensuring the top-level model object passed to the `Trainer` is somehow recognized or treated as a PEFT-compatible model for fine-tuning is crucial.
*   If the `ValueError` persists, further investigation into how `Trainer` checks for "purely quantized models" versus PEFT-adapted quantized models will be needed. This might involve looking at PEFT's `PeftModel` class hierarchy or specific flags/attributes the `Trainer` expects.

*(The content below this line seems to be a duplicate from a merge or an older version and has been consolidated above or is part of the general project description. It will be removed for clarity.)*

---
*The previous detailed sections on "Available Experiments and Evaluations" and "Running Experiments" have been integrated into the main "Usage" and specific model sections above for better flow. The redundant "CLIP Linear Probing" section at the very end is also consolidated.* 