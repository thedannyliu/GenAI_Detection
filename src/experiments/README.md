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

  **Key Implementation Details in `zero_shot_vlm_eval.py`**:
  - **Custom Collate Function**: The script utilizes a custom `vlm_collate_fn` for the `DataLoader`. This is necessary because the `VLMGenImageDataset` yields PIL Image objects (as expected by many VLM preprocessors) and integer labels. The default `DataLoader` collate function cannot batch PIL Images directly. `vlm_collate_fn` handles this by keeping images as a list of PIL objects and batching labels into a tensor.
  - **Model Evaluation Mode**: To set a VLM wrapper (e.g., `CLIPModelWrapper`) to evaluation mode, the script now attempts to call `wrapper_instance.model.eval()`. This is because the actual PyTorch `nn.Module` (which has the `eval()` method) is typically stored within a `.model` attribute of the VLM wrapper. If `.model.eval()` is not found, it falls back to trying `wrapper_instance.eval()`. This ensures that operations like dropout are correctly disabled during evaluation.

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
  This will run the evaluation for all VLM configurations listed in `vlm_zero_shot_custom.yaml` and for all datasets specified in `evaluation_datasets`. Results for each model-dataset combination will be saved in a separate subdirectory under `output_dir_base / <model_name> / <dataset_name>`. 

  **Multi-Dataset Evaluation Configuration**:
  The `configs/vlm_zero_shot_custom.yaml` (and similar configurations) can be set up to evaluate models on multiple datasets in a single run. This is achieved through two main keys at the root of the YAML:
  - `dataset_defaults`: This section defines common parameters that will be applied to all datasets unless overridden. It typically includes:
    - `eval_split`: Default split (e.g., "val", "test").
    - `num_samples_eval`: Default number of samples to evaluate from the entire dataset (sum across classes). For example, 1000 means 1000 images will be randomly sampled from the specified split of the dataset.
    - `class_to_idx`: Default mapping from class folder names to integer labels (e.g., `nature: 0`, `ai: 1`).
  - `evaluation_datasets`: This is a list, where each item represents a specific dataset to evaluate.
    - `name`: A unique name for the dataset (e.g., "ImageNet-SDv4_Val"). This name is used in output paths.
    - `root_dir`: The root directory of the dataset. The script expects subfolders named according to `class_to_idx` keys within the `root_dir/eval_split/` path.
    - `eval_split` (optional): Overrides the default split for this specific dataset.
    - `num_samples_eval` (optional): Overrides the default number of samples for this specific dataset.
    - `class_to_idx` (optional): Overrides the default class mapping. This is particularly useful for datasets with different folder naming conventions. For example, for a dataset like Chameleon with folders `0_real` and `1_fake` inside the `test` split, you might set `eval_split: "test"` and `class_to_idx: {"0_real": 0, "1_fake": 1}`. The script will then look for images in `<root_dir>/test/0_real/` and `<root_dir>/test/1_fake/` and assign them labels 0 and 1 respectively.

  The script iterates through each model defined in `vlm.model_configs` and then, for each model, iterates through each dataset in `evaluation_datasets`, applying the merged default and specific dataset configurations.

  **Robust Dataset Configuration Handling**:
  To better support diverse dataset structures (like Chameleon with its unique folder names and split) and ensure that specific dataset configurations correctly override defaults, the script's logic for resolving parameters like `eval_split`, `class_to_idx`, and `num_samples_eval` for each dataset has been made more explicit. It now clearly prioritizes settings defined directly under a dataset entry in `evaluation_datasets` over those in `dataset_defaults`.

- **`vlm_finetune_and_infer.py`**:
  (Details about this script can be added here once its functionality is finalized and documented.)

## Configuration

Experiment configurations are typically managed using YAML files stored in the `configs/` directory. Each script in this module should clearly document the configuration parameters it expects.

## Running Experiments

Refer to the specific script's documentation or command-line help for instructions on how to run it.

## Output

Experiment outputs, such as trained model checkpoints, evaluation metrics, logs, and visualizations, are generally saved to the `results/` directory, often in subdirectories named after the experiment or configuration. 

### Zero-Shot VLM DataLoader and Collate Function

For zero-shot VLM evaluation using `zero_shot_vlm_eval.py`, the `DataLoader` is configured with a custom `collate_fn` (named `vlm_collate_fn` within the script). This is essential because:
- The `VLMGenImageDataset` is designed to return a tuple of `(PIL.Image.Image, label)` for each sample.
- Many VLM preprocessing pipelines (e.g., from Hugging Face Transformers) expect PIL Image objects as input, not PyTorch tensors.
- The default `torch.utils.data.dataloader.default_collate` function cannot handle batching of PIL Image objects directly and would raise a `TypeError`.

The `vlm_collate_fn` addresses this by:
- Gathering all PIL Image objects from the batch into a Python list.
- Collecting all integer labels from the batch and converting them into a single PyTorch tensor.
- Returning a tuple `(list_of_pil_images, labels_tensor)`.

This structure is then correctly processed by the evaluation loop in `zero_shot_vlm_eval.py`.

**Troubleshooting `default_collate` Errors:**
If you encounter an error message similar to `TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>` while working with or modifying `zero_shot_vlm_eval.py`, ensure that the `DataLoader` is indeed initialized with `collate_fn=vlm_collate_fn`.

### Prompt-to-Class Mapping Completeness

This section has been moved and expanded in `src/prompts/README.md`. Please refer to that file for detailed guidance on configuring `prompt_to_class_map` in your YAML files, especially when using `GenImageDetectPrompts`.

If you see warnings about prompts not being found in the map during the execution of `zero_shot_vlm_eval.py`, it means the `prompt_to_class_map` in your experiment's YAML configuration is incomplete. You should update the YAML to include all possible prompts that your VLM might select, along with their corresponding class labels (e.g., 0 for real, 1 for AI).

---

*More experiment details can be added here as the project evolves.* 