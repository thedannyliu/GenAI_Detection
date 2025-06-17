# Configuration Files

This directory stores YAML configuration files for various scripts in the project, particularly for experiments and model training/evaluation.

## `vlm_zero_shot_custom.yaml`

This is the primary configuration file for the `src/experiments/zero_shot_vlm_eval.py` script. It allows for detailed setup of zero-shot evaluations for Vision-Language Models (VLMs).

### Key Configuration Sections:

-   **`eval_gpu_id`**: (Integer) Specifies the GPU ID to be used for the evaluation (e.g., 0, 1, 2, 3). If set to `null` or not provided, the script will use the default CUDA device or CPU if CUDA is unavailable.
-   **`random_seed`**: (Integer) A seed value for random number generators (Python's `random`, `numpy.random`, `torch.manual_seed`). This ensures reproducibility of experiments, especially for dataset sampling. Default is 42 if not specified in the script's loading logic.
-   **`batch_size`**: (Integer) Batch size for the DataLoader. For VLM zero-shot evaluation where images are often processed one by one with PIL inputs, this is typically set to 1.
-   **`num_workers`**: (Integer) Number of worker processes for data loading.
-   **`output_dir_base`**: (String) The base directory where results for each VLM run will be stored. The script will create a subdirectory within this path for each configured VLM, named after the `name` field in its configuration (e.g., `results/zero_shot_eval/dataset_name/CLIP-L-14-OpenAI/`).

-   **`dataset`**: Configuration for the dataset to be used.
    -   `root_dir`: (String) Path to the root directory of the specific dataset version being evaluated (e.g., `/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/`). This should point to the folder containing `train` and `val` subdirectories.
    -   `eval_split`: (String) Specifies which split of the dataset to use for evaluation (e.g., "val", "test").
    -   `num_samples_eval`: (Integer or `null`) The number of images to randomly sample from the `eval_split`. If `null` or not provided, all samples in the split will be used. Sampling is performed with the global `random_seed`.
    -   `class_to_idx`: (Dictionary) Maps class names (e.g., "nature", "ai") to integer labels (e.g., `nature: 0`, `ai: 1`). This should align with the dataset structure and the labels used internally by the evaluation script.

-   **`vlm`**: Contains configurations related to the Vision-Language Models.
    -   **`model_configs`**: (List of Dictionaries) This is a list where each dictionary defines a VLM to be evaluated sequentially. Each entry requires:
        -   `name`: (String) A unique identifier for this specific model run (e.g., "CLIP-L-14-OpenAI", "BLIP-ITM-Large-Salesforce"). This name is used to create the subdirectory under `output_dir_base` for storing results.
        -   `model_config`: (Dictionary) Configuration for the VLM wrapper, passed to `get_instance_from_config`.
            -   `module`: (String) The Python module path to the VLM wrapper class (e.g., `src.models.vlm.clip_model`).
            -   `class`: (String) The class name of the VLM wrapper (e.g., `CLIPModelWrapper`).
            -   `params`: (Dictionary) Parameters passed to the constructor of the VLM wrapper class.
                -   `model_name`: (String) An internal, often descriptive name for the model (e.g., "CLIP-L-14").
                -   `config`: (Dictionary) Model-specific parameters, such as `model_id` (Hugging Face model identifier like `openai/clip-vit-large-patch14`), `task` for BLIP models, or `max_new_tokens` for generative models.
    -   **`prompt_config`**: (Dictionary) Configuration for the prompt strategy, which is shared across all VLM runs defined in `model_configs`. This is passed to `get_instance_from_config`.
        -   `module`: (String) Python module path to the prompt strategy class (e.g., `src.prompts.generic_prompts`).
        -   `class`: (String) Class name of the prompt strategy (e.g., `GenImageDetectPrompts`).
        -   `params`: (Dictionary) Parameters for the prompt strategy constructor. This often includes:
            -   `prompt_to_class_map`: (Dictionary) For discriminative VLMs, maps the text of prompts (e.g., "a real photograph") to class labels (0 or 1).
            -   `keyword_to_class_map`: (Dictionary) For generative VLMs, maps keywords expected in the model's response (e.g., "real photograph") to class labels.
            -   Other strategy-specific parameters like `discriminative_prompts`, `generative_question`, `generative_response_keywords` can be set here to override defaults in the prompt strategy class.

-   **`tie_breaking_label_for_generative`**: (Integer, 0 or 1) If a generative VLM detects keywords for both real and AI classes with equal confidence (or if both are present), this label is used as the prediction.
-   **`default_label_for_generative_no_match`**: (Integer, 0 or 1) If a generative VLM fails to detect any relevant keywords in its output, this label is used as the prediction.

### Example Snippet for `model_configs`:
```yaml
vlm:
  model_configs:
    - name: "CLIP-L-14-OpenAI"
      model_config:
        module: "src.models.vlm.clip_model"
        class: "CLIPModelWrapper"
        params:
          model_name: "CLIP-L-14"
          config:
            model_id: "openai/clip-vit-large-patch14"
    
    - name: "BLIP-ITM-Large-Salesforce"
      model_config:
        module: "src.models.vlm.blip_model"
        class: "BlipModelWrapper"
        params:
          model_name: "BLIP-ITM-L"
          config:
            model_id: "Salesforce/blip-itm-large-eval"
            task: "image-text-matching"
  # ... (prompt_config follows)
```

## Other Configuration Files

- `cnn_baseline.yaml`: Example configuration for a baseline CNN model.
- `default.yaml`: May contain default parameters for the project.
- `prompt_tuning.yaml`: Example configuration for prompt tuning experiments.
- `prompt_tuning_instructblip_config.yaml`: Configuration for InstructBLIP prompt tuning.
- `vlm_fine_tune.yaml`: Example configuration for VLM fine-tuning.
- `vlm_zero_shot.yaml`: An older or alternative configuration for zero-shot VLM evaluation, possibly for a single model run.
- `gxma_fusion_config.yaml`: Configuration for training the GXMA Fusion Detector proof of concept.
- `gxma_fusion_parallel.yaml`: Uses the parallel attention fusion strategy (Strategy B).
- `gxma_fusion_gated.yaml`: Uses the hierarchical gating fusion strategy (Strategy C).

These files should be documented similarly as their functionalities are developed or integrated into experimental workflows. 