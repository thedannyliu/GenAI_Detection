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

**For VLM Models (Fine-tuning):**

```bash
python src/experiments/vlm_finetune_and_infer.py --config path_to_your_vlm_config.yaml 
# (Note: This script was identified as a likely candidate for main.py functionality for VLMs)
# Or, if main.py is indeed the entry point and uses configs like vlm_fine_tune.yaml:
# python src/main.py --config configs/vlm_fine_tune.yaml 
```

**For CNN-based Classification (New):**

To train a ResNet50-based CNN classifier for detecting AI-generated images, use the `train_cnn.py` script with a YAML configuration file. 
A sample configuration is provided in `configs/cnn_baseline.yaml`.

```bash
python src/training/train_cnn.py --config configs/cnn_baseline.yaml
```

Before running, ensure that `configs/cnn_baseline.yaml` is configured correctly, especially the following fields:
- `general.output_dir`: Directory to save model checkpoints and training logs.
- `general.gpu_id`: GPU to use for training.
- `data.base_data_dir`: Root directory containing `train` and `val` subfolders for the dataset (e.g., `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/`).
- `data.train_samples_per_class`, `data.val_samples_per_class`, `data.test_samples_per_class`: Should be set to 5000, 500, and 500 respectively for the 10k/1k/1k split.
- `model.*`: Model architecture settings (type, pretrained, freeze_backbone, etc.).
- `training.*`: Training hyperparameters (epochs, batch_size, optimizer, learning_rate, early_stopping, etc.).

This script now supports loading all configurations from the YAML file, saves both the best and last model weights, and includes early stopping.

### Zero-shot Evaluation

```bash
python src/main.py --config configs/vlm_zero_shot.yaml --mode eval
```

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