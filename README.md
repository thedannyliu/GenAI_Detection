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
├── src/                            <- Source code
└── tests/                          <- Test suite
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

5. **Configuration:**
   Review and adjust YAML files in `configs/` as needed.

## Usage

### Training a Model

**For VLM Models (Fine-tuning InstructBLIP):**

To fine-tune an InstructBLIP model (e.g., `Salesforce/instructblip-vicuna-7b`) for AI-generated image detection, use the `train_instructblip.py` script. This script handles data preparation, LoRA or full fine-tuning, and saves model artifacts. It supports evaluation at each epoch, saving the best and recent checkpoints, and early stopping.

All training parameters are managed through a YAML configuration file. A template is provided in `configs/train_instructblip_config.yaml`. You should copy and modify this file.

To run the training:
1.  **Prepare your configuration file**: Copy `configs/train_instructblip_config.yaml` (e.g., to `configs/my_instructblip_vicuna_train_config.yaml`) and edit it. Key parameters include:
    *   `data`: Paths and sample counts.
    *   `model`: `name_pretrained` (e.g., `"Salesforce/instructblip-vicuna-7b"`), `finetune_method` ("lora" or "full"), and `lora_params` (ensure `target_modules` are appropriate for the chosen model if using LoRA).
    *   `training`: `epochs` (max epochs), `batch_size`, learning rates, `evaluation_strategy` (set to `"epoch"` for per-epoch validation), `save_strategy` (set to `"epoch"` to save checkpoints per epoch), `save_total_limit` (e.g., 3, to keep recent checkpoints), `early_stopping_patience` (e.g., 50, epochs to wait for improvement before stopping), and `early_stopping_threshold`.
    *   `output`: `base_results_dir_root`.
    *   `environment` (New): `gpu_id` (e.g., `0`, `1` for specific GPU, `-1` for CPU, or leave blank/null for auto-selection).

2.  **Run the training script**:
    ```bash
    python src/training/train_instructblip.py --config configs/my_instructblip_vicuna_train_config.yaml
    ```
    Replace with your actual configuration file path.

The script will create a `RUN_ID` based on the configuration. 
    *   The **best model** (based on validation loss) will be saved in `<output.base_results_dir_root>/<RUN_ID>/final_model_artifacts/`.
    *   **Recent checkpoints** (including the last one if training completes or stops early) will be saved in `<output.base_results_dir_root>/<RUN_ID>/trainer_checkpoints/`. The number of checkpoints is limited by `save_total_limit`.
    *   Training logs for TensorBoard will be in `<output.base_results_dir_root>/<RUN_ID>/training_logs/`.

**For CNN-based Classification (New):**

To train a ResNet50-based CNN classifier for detecting AI-generated images, use the `train_cnn.py` script with a YAML configuration file. 
A sample configuration is provided in `configs/cnn_baseline.yaml`.

```bash
python src/training/train_cnn.py --config configs/cnn_baseline.yaml
```

Before running, ensure that `configs/cnn_baseline.yaml` is configured correctly. Key sections in the YAML include:
- `general`: 
    - `output_dir`: Base directory for all experiment outputs (e.g., `results/cnn_base_runs`). A sub-directory named after `model.name` will be automatically created here to store all artifacts for a specific run (e.g., `results/cnn_base_runs/resnet50_experiment1/`).
    - `seed`: Random seed for reproducibility.
    - `gpu_id`: Specific GPU ID to use for training.
- `model`: Architecture settings like `type`, `name` (this name also defines the output subfolder), `pretrained`, `freeze_backbone`, `num_classes`.
- `data`: `base_data_dir` (e.g., `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/`), `batch_size`, `num_workers`, and sample counts (`train_samples_per_class`, `val_samples_per_class`, `test_samples_per_class` for the 10k/1k/1k split).

This script now supports loading all configurations from the YAML file, saves both the best and last model weights, and includes early stopping.

### Zero-shot Evaluation

```bash
python src/main.py --config configs/vlm_zero_shot.yaml --mode eval
```

### Evaluating a Fine-tuned VLM Model (InstructBLIP)

To evaluate a fine-tuned InstructBLIP model (trained using `src/training/train_instructblip.py`) on a test set, use the `eval_instructblip.py` script. This script loads the specified test data, the trained model artifacts (based on a `RUN_ID` matching the training run), performs inference, and calculates evaluation metrics.

1.  **Ensure your trained model artifacts exist**: The training script saves artifacts to `results/instructblip_finetune/<RUN_ID>/final_model_artifacts/`.
2.  **Configure `eval_instructblip.py`**: Modify the main block of `src/evaluation/eval_instructblip.py` to set the correct `RUN_ID` parameters (e.g., `FINETUNE_METHOD`, `NUM_TRAIN_SAMPLES_CONFIG`, `MODEL_NAME_PRETRAINED_CONFIG`, etc.) to match the training run you want to evaluate. This ensures the script loads the correct model and test data.
3.  **Run the evaluation script**:

    ```bash
    python src/evaluation/eval_instructblip.py
    ```
    Evaluation results, including detailed metrics, classification reports, and confusion matrices, will be saved in a subdirectory like `results/instructblip_finetune/<RUN_ID>/evaluation_on_test_set/`.

### Custom Zero-shot VLM Evaluation (New)

For more flexible zero-shot evaluation of various VLMs (CLIP, BLIP, InstructBLIP, LLaVA, etc.) using the new extensible VLM and prompt strategy framework, use the `zero_shot_vlm_eval.py` script directly. This script allows detailed configuration of models, prompts, specific GenImage generator datasets, and output paths through a YAML configuration file.

1.  **Configure your evaluation**: 
    Create or modify a YAML configuration file (e.g., `configs/vlm_zero_shot_custom.yaml`). Specify:
    *   `eval_gpu_id`: GPU to use.
    *   `dataset`: `root_dir` (path to the specific GenImage sub-dataset, e.g., `.../stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/`), `eval_split`.
    *   `vlm.model_config`: Module, class, and parameters (like `model_id` and `config` for the VLM wrapper) for the VLM to test.
    *   `vlm.prompt_config`: Module, class, and parameters for the prompt strategy, including `prompt_to_class_map` or `keyword_to_class_map` for interpreting VLM outputs.
    *   `output_dir`: Where to save metrics and results.

    Refer to `configs/vlm_zero_shot_custom.yaml` for a detailed template.

2.  **Run the evaluation script**:
    ```bash
    python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml
    ```
    Replace `configs/vlm_zero_shot_custom.yaml` with the path to your specific configuration file.
    Results, including detailed metrics and raw predictions, will be saved to the specified `output_dir`.

### Running Tests

```bash
pytest
```

## Documentation

- Detailed experiment design: `docs/experiment_plan.md`
- Code design: `docs/code_plan.md`
- Dataset instructions: `data/README.md`

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