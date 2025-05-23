# Experiments Module

This directory (`src/experiments/`) contains scripts for running various experiments, such as model training, evaluation, and analysis related to AI-generated image detection.

## Key Scripts

- **`zero_shot_vlm_eval.py`**: 
  This script is designed to perform zero-shot evaluation of Vision-Language Models (VLMs) on image datasets, particularly for tasks like distinguishing real vs. AI-generated images.

  **Features**:
  - Supports multiple VLMs (e.g., CLIP, BLIP, InstructBLIP, LLaVA) through a common `BaseVLM` interface.
  - Utilizes configurable prompt strategies for different VLM types (discriminative vs. generative).
  - Evaluates on datasets like `genimage`, allowing selection of specific data generators and splits.
  - Calculates metrics such as accuracy, precision, recall, F1-score, and provides a detailed classification report.
  - Saves evaluation metrics and raw per-sample prediction results to JSON files.
  - Allows running evaluations for multiple VLM configurations sequentially from a single YAML configuration file.
  - Supports setting a random seed for reproducible dataset sampling and other stochastic processes.
  - Outputs results into model-specific subdirectories.

  **Configuration**:
  The script is configured using a YAML file (e.g., `configs/vlm_zero_shot_custom.yaml`). Key configuration options include:
  - `eval_gpu_id`: GPU to use for evaluation.
  - `random_seed`: Seed for reproducibility.
  - `batch_size`, `num_workers`: DataLoader parameters.
  - `output_dir_base`: Base directory where results for each model run will be saved in a subdirectory named after the model run.
  - `dataset`:
      - `root_dir`: Path to the specific dataset folder (e.g., the `imagenet_ai_XXXX_YYYY` subfolder within a `genimage` generator directory).
      - `eval_split`: Dataset split to use (e.g., "val").
      - `num_samples_eval`: Number of samples to randomly evaluate from the dataset. If null, uses all samples.
      - `class_to_idx`: Mapping from class names (e.g., "nature", "ai") to integer labels.
  - `vlm`:
      - `model_configs`: A list of VLM configurations to run. Each item in the list should define:
          - `name`: A unique name for this model run (e.g., "CLIP-L-14-OpenAI"). This name will be used for the output subdirectory.
          - `model_config`: Configuration for the VLM wrapper, including:
              - `module`: Path to the VLM wrapper module (e.g., `src.models.vlm.clip_model`).
              - `class`: Class name of the VLM wrapper (e.g., `CLIPModelWrapper`).
              - `params`: Parameters for the VLM wrapper, including:
                  - `model_name`: An internal name for the VLM (e.g., "CLIP-L-14").
                  - `config`: VLM-specific parameters passed to the wrapper (e.g., `model_id` for Hugging Face models).
      - `prompt_config`: Configuration for the prompt strategy (shared across all model runs in the config file).
          - `module`: Path to the prompt strategy module (e.g., `src.prompts.generic_prompts`).
          - `class`: Class name of the prompt strategy (e.g., `GenImageDetectPrompts`).
          - `params`: Parameters for the prompt strategy, including mappings like `prompt_to_class_map` and `keyword_to_class_map`.
  - `tie_breaking_label_for_generative`, `default_label_for_generative_no_match`: Labels for handling ambiguous or non-matching predictions from generative VLMs.

  **Usage**:
  ```bash
  python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml
  ```
  This will run the evaluation for all VLM configurations listed in `vlm_zero_shot_custom.yaml`. Results for each VLM will be saved in a separate subdirectory under the specified `output_dir_base`.

- **`vlm_finetune_and_infer.py`**:
  (Details about this script can be added here once its functionality is finalized and documented.)

## Configuration

Experiment configurations are typically managed using YAML files stored in the `configs/` directory. Each script in this module should clearly document the configuration parameters it expects.

## Running Experiments

Refer to the specific script's documentation or command-line help for instructions on how to run it.

## Output

Experiment outputs, such as trained model checkpoints, evaluation metrics, logs, and visualizations, are generally saved to the `results/` directory, often in subdirectories named after the experiment or configuration. 