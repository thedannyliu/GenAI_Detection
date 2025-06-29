# Experiment Plan: Detecting AI-Generated Images using VLMs

## Research Questions

* **RQ1: VLM Detection Performance** – How well can pre-trained vision-language models distinguish AI-generated images from real images? For example, what is the accuracy of CLIP or BLIP in zero-shot detection, and how does fine-tuning affect their performance?
* **RQ2: Zero-Shot vs Fine-Tuning** – What are the trade-offs between zero-shot detection (no additional training, using prompts) and fine-tuned detection (adapting the model on labeled data)? Can prompt engineering alone yield competitive results, or is model fine-tuning necessary for high accuracy?
* **RQ3: Generalization to Unseen Generators** – Do models trained on images from certain generators (e.g. Stable Diffusion, BigGAN) generalize to **unseen generators** like newer Midjourney versions or others not in the training set? How do VLM-based detectors compare to conventional CNNs in cross-generator generalization?
* **RQ4: Parameter-Efficient Adaptation** – Can techniques like **prompt tuning** (learning soft text prompts) or **adapter modules** yield performance close to full fine-tuning while using far fewer trainable parameters? What impact do these methods have on in-distribution accuracy vs. out-of-distribution robustness?
* **RQ5: Explainability and Cues** – What **visual cues or features** do the models use to make decisions? Through Grad-CAM and attention map analysis, can we identify whether the detector focuses on specific artifacts (e.g. texture, noise patterns) indicative of generative models? Are these cues consistent with known differences between AI-generated and natural images?
* **RQ6: Robustness and Countermeasures** – How robust are our detection models to common transformations or "laundering" of images (downscaling, compression, slight noise) applied to hide AI origins? Additionally, how might an adversary fool a VLM-based detector, and how can we make the detectors more resilient?

## Methodology Overview

Our approach consists of several stages and strategies to comprehensively evaluate the use of VLMs for AI-generated image detection:

### 1. Baseline Models

We begin with baseline detectors for reference:

* **Standard CNN Baseline:** Train a convolutional neural network (e.g. ResNet-50, using the script `src/training/train_cnn.py`) from scratch or fine-tuned from ImageNet weights on the genimage dataset. The dataset will be sampled as per the specific requirements (10k train, 1k val, 1k test, 1:1 class balance from specified data paths).
* **Zero-Shot VLM Baselines:** Evaluate models like CLIP, BLIP, GIT, and Flamingo in a zero-shot manner (no training on genimage).
* **Implementation Details:** We'll use public APIs or implementations (e.g., Hugging Face Transformers) and evaluate based on accuracy, ROC-AUC, and other metrics.

### 2. Fine-Tuned VLM Models

Next, we fine-tune VLMs on the genimage training data:

* **CLIP Fine-Tuning:** Using CLIP's image encoder as a backbone with both partial and full fine-tuning approaches.
* **Other VLM Fine-Tuning:** Where feasible, fine-tune models like BLIP or BLIP-2, casting the detection task as a binary classification or question-answering problem.
* **Training Strategy:** Use standard practices with binary cross-entropy loss, data augmentation, and early stopping based on validation performance.

### 3. Prompt Engineering Analysis

For zero-shot approaches, we'll systematically explore prompt engineering:

* Testing simple prompts like "a real photo" vs "an AI-generated image" as well as more descriptive variants.
* Using a held-out subset of training data as a prompt validation set to measure which phrasings yield the highest accuracy.
* Documenting performance differences observed with different prompt formulations.

### 4. genimage Dataset Utilization

The dataset will be organized as follows:

* Use the provided training set from multiple generators, following the authors' recommendation to combine all generators' data.
* Ensure class balance (fake vs real) remains 1:1 in the training data.
* Use the combined validation set as our test set, with leave-one-generator-out tests to evaluate generalization.

### 5. Evaluation Methodology

We will use a consistent evaluation protocol:

* **Metrics:** Accuracy, Precision/Recall, F1-score, and ROC-AUC for binary classification.
* **Logging & Checkpointing:** Track training and validation metrics per epoch using TensorBoard and/or Weights & Biases.
* **Reproducibility:** Set random seeds for all operations and provide configuration files for each experiment.
* **Statistical Analysis:** For comparing methods, run multiple trials and perform significance testing when differences are subtle.

### 6. Advanced Experimentation

Beyond the main comparisons, we plan several advanced experiments:

* **Prompt Tuning (Soft Prompts):** Implement learnable prompt vectors for CLIP's text encoder while keeping the model weights frozen.
* **Adapter-Based Fine-Tuning:** Explore lightweight adapter modules inserted into the VLM's architecture, comparing performance to full fine-tuning.
* **Explainability Analysis:** Use Grad-CAM on our best-performing models to visualize regions influencing predictions.
* **Robustness Testing:** Simulate attempts to "fool" detectors with image manipulations like JPEG compression, noise, or blurring.

### 7. GXMA Fusion PoC Workflow (2025-06-16)

This section documents the **GXMA Fusion** proof-of-concept pipeline that combines
frequency fingerprints (Radial FFT + DCT + DWT) with CLIP semantics via
cross-attention.  The reference implementation lives in:

* Model: `src/models/gxma/`
* Training script: `src/training/train_gxma.py`
* Default config: `configs/gxma_fusion_config.yaml`

#### 7.1 Datasets

| Dataset | Purpose | Split(s) used | Size | Path |
|---------|---------|---------------|------|------|
| **genimage_poc** | Main training/validation/test | train / val / test | 64k / 8k / 8k | `/raid/dannyliu/dataset/GAI_Dataset/genimage/genimage_poc` |
| **chameleon_poc** | Out-of-distribution test | — (root has `ai/` & `nature/`) | 10k | `/raid/dannyliu/dataset/GAI_Dataset/chameleon_poc` |

Both datasets are constructed with fixed seed `42`; see the data README for
exact generation steps.

#### 7.2 Configuration Highlights

```yaml
general:
  experiment_name: "gxma_fusion_poc_genimage"
data:
  base_data_dir: "/…/genimage_poc"
  train_split_name: train
  val_split_name: val
  test_split_name: test
  class_to_idx: {nature: 0, ai: 1}
model:
  freq_methods: ["radial", "dct"]   # choose any of radial, dct, wavelet
evaluation:
  batch_size: 128
  extra_tests:            # new field handled by train_gxma.py
    - name: "chameleon_poc"
      base_data_dir: "/…/chameleon_poc"
      split_name: ""      # root dir contains ai/ & nature/
      class_to_idx: {nature: 0, ai: 1}
training:
  early_stopping:
    monitor: val_auc      # primary metric
```

Key points:
* **extra_tests** allows an arbitrary list of external datasets that will be
  evaluated *after* the main training loop using the best checkpoint.
* Early stopping now supports `val_auc` & `val_f1` in addition to accuracy
  or loss.
* The training script copies the exact YAML into
  `<output_dir>/config_used.yaml` for full reproducibility.

#### 7.3 Output Artefacts

After a run you should find the following structure (example GPU-1 run):

```text
results/gxma_runs/gxma_fusion_poc_genimage/
├── best_model.pth                # checkpoint with highest val AUC
├── config_used.yaml              # frozen copy of the YAML
├── tensorboard/                  # TB event files (loss/acc/auc/f1)
├── training_results.json         # metrics & curves
│   ├─ history.*                  # per-epoch arrays
│   ├─ test_* (genimage_poc.test) # loss/acc/auc/f1 + confusion matrix
│   └─ extra_tests/               # dict per external dataset
│       └─ chameleon_poc/ …       # loss/acc/auc/f1 + confusion matrix
└── …
```

#### 7.4 Re-running on a Different Dataset

1. Duplicate the YAML file.
2. Change `experiment_name` and the `data.*` paths.
3. (Optional) Edit `extra_tests` to include additional OOD datasets.

Then launch:

```bash
python src/training/train_gxma.py --config configs/your_new.yaml --mode fusion
```

#### 7.5 Latest Updates (2025-06-16)

* **Parallel Attention Streams implemented** – `ParallelCrossAttentionFusion` now
  active; use `fusion_strategy: parallel` in YAML.  Ready-made configs are
  available under `configs/gxma/poc_stage1/`.
* New LoRA fine-tune config to train the semantic branch end-to-end while
  keeping frequency extractors fixed.

#### 7.6 Roadmap

* Add Florence-2 semantics or patch-level tokens (Tier-3 experiments).
* Implement Meta-Gate (Strategy C) for dynamic weighting of frequency experts.
* Support mixed-precision training and multi-GPU.

## Experiment Schedule

| Week | Tasks |
|------|-------|
| 1    | Dataset preparation, baseline CNN implementation, and zero-shot VLM evaluation |
| 2    | CLIP fine-tuning (partial and full) and prompt engineering analysis |
| 3    | Adapter and prompt tuning experiments, cross-generator generalization tests |
| 4    | Explainability analysis, robustness testing, and final evaluation |
| 5    | Analysis, documentation, and preparation of results |

## Success Metrics

We will consider our experiments successful if we can:

1. Quantify the performance gap between zero-shot and fine-tuned VLMs for AI-generated image detection
2. Identify which prompt engineering strategies are most effective for zero-shot detection
3. Determine whether parameter-efficient methods can match full fine-tuning performance
4. Assess generalization capabilities to unseen generators
5. Provide insights into what visual cues the models use for detection

All findings will be thoroughly documented with quantitative metrics and qualitative visualizations for interpretability. 