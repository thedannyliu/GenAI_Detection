# Detecting AI-Generated Images using Vision-Language Models (VLMs) on the genimage Dataset

## Project Overview

This repository contains the implementation of our research project on detecting AI-generated images using Vision-Language Models (VLMs). We leverage large-scale VLMs such as CLIP, BLIP, and Flamingo to detect AI-created images in the genimage benchmark, a million-scale dataset of real and fake image pairs.

## Key Features

- Zero-shot classification using pre-trained VLMs
- Fine-tuning strategies for improved detection
- Prompt engineering and analysis
- Parameter-efficient adaptation methods
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

*(The content below this line seems to be a duplicate from a merge or an older version and has been consolidated above or is part of the general project description. It will be removed for clarity.)*

---
*The previous detailed sections on "Available Experiments and Evaluations" and "Running Experiments" have been integrated into the main "Usage" and specific model sections above for better flow. The redundant "CLIP Linear Probing" section at the very end is also consolidated.* 