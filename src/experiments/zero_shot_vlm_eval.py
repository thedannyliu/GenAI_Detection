import argparse
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import random # Added for seed setting
import shutil # Added for copying config

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adjust import paths based on your project structure
from src.utils.config_utils import load_yaml_config, get_instance_from_config
from src.data_processing.custom_dataset import GenImageDataset # Assuming GenImageDataset is suitable

# Define labels for clarity, these should match your dataset's class_to_idx or be configurable
LABEL_REAL = 0  # Typically 'nature'
LABEL_AI = 1    # Typically 'ai'

def vlm_collate_fn(batch):
    # batch is a list of tuples, where each tuple is (PIL.Image, label)
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    # For batch_size=1 (common for VLMs in this script), 'images' will be a list containing a single PIL Image.
    # The evaluation loop is set up to take images[i], which works fine.
    return images, labels_tensor

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def evaluate_predictions(all_predictions, all_labels, class_names_for_eval):
    """
    Calculates accuracy and other metrics.
    Args:
        all_predictions (list): List of predicted labels (0 or 1).
        all_labels (list): List of true labels (0 or 1).
        class_names_for_eval (list): List of class names, e.g., ['nature', 'ai']
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    if not all_predictions or not all_labels:
        return {"error": "No predictions or labels to evaluate."}
    if len(all_predictions) != len(all_labels):
        return {"error": "Mismatch in number of predictions and labels."}

    predictions_arr = np.array(all_predictions)
    labels_arr = np.array(all_labels)

    accuracy = np.mean(predictions_arr == labels_arr)
    
    metrics = {"accuracy": accuracy, "num_samples": len(labels_arr)}

    try:
        from sklearn.metrics import classification_report
        report = classification_report(labels_arr, predictions_arr, target_names=class_names_for_eval, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        metrics["precision_real"] = report[class_names_for_eval[LABEL_REAL]]["precision"]
        metrics["recall_real"] = report[class_names_for_eval[LABEL_REAL]]["recall"]
        metrics["f1_real"] = report[class_names_for_eval[LABEL_REAL]]["f1-score"]
        metrics["precision_ai"] = report[class_names_for_eval[LABEL_AI]]["precision"]
        metrics["recall_ai"] = report[class_names_for_eval[LABEL_AI]]["recall"]
        metrics["f1_ai"] = report[class_names_for_eval[LABEL_AI]]["f1-score"]
    except ImportError:
        print("scikit-learn not installed. Skipping classification report.")
    except Exception as e:
        print(f"Error generating classification report: {e}")

    return metrics

def run_evaluation_for_model(model_run_config, global_cfg, prompt_strategy, eval_dataset_for_loader, device, class_names_for_eval, idx_to_class):
    """
    Runs the evaluation loop for a single VLM configuration.
    """
    vlm_wrapper_config = model_run_config['model_config']
    model_run_name = model_run_config.get('name', vlm_wrapper_config.get('params', {}).get('model_name', 'unknown_model'))
    
    print(f"\n--- Starting evaluation for model: {model_run_name} ---")

    # 1. Initialize VLM Model
    print(f"Initializing VLM model: {model_run_name}...")
    vlm = get_instance_from_config(vlm_wrapper_config)
    print(f"VLM Model {vlm.get_model_name()} loaded for run {model_run_name}.")
    # Ensure VLM's internal model is on the correct device (our wrappers should handle this).

    batch_size = global_cfg.get('batch_size', 1)
    num_workers = global_cfg.get('num_workers', 2)
    eval_loader = DataLoader(
        eval_dataset_for_loader,
        batch_size=batch_size, # From config
        shuffle=False, # Usually false for evaluation
        num_workers=num_workers, # From config
        pin_memory=True, # If using GPU, can speed up host-to-device transfers
        collate_fn=vlm_collate_fn # Use the custom collate function
    )
    print(f"DataLoader ready for {model_run_name} with {len(eval_dataset_for_loader)} samples.")

    # 2. Run Evaluation Loop
    all_predictions = []
    all_labels = []
    raw_results = [] 

    is_generative_vlm = vlm.model_name.lower() in ["llava", "instructblip"] 
    
    prompts_for_vlm = prompt_strategy.get_prompts(class_names=class_names_for_eval)
    keywords_for_vlm = prompt_strategy.get_keywords_for_response_check()
    
    if is_generative_vlm:
        generative_question = prompt_strategy.get_vlm_question()
        if generative_question: 
            if hasattr(vlm, 'config') and isinstance(vlm.config, dict):
                question_key = f"{vlm.model_name.lower()}_question" 
                vlm.config[question_key] = generative_question
                print(f"Using question from prompt strategy for {vlm.model_name}: '{generative_question}'")
            else:
                print(f"Warning: Could not set question for {vlm.model_name} from prompt strategy.")

    print(f"Starting evaluation loop for {model_run_name} on {len(eval_loader)} batches...")
    for batch_idx, (images, labels) in enumerate(tqdm(eval_loader, desc=f"Evaluating {model_run_name}")):
        for i in range(len(images)): 
            image = images[i] 
            true_label = labels[i].item()
            
            current_sample_index_in_dataloader = batch_idx * eval_loader.batch_size + i
            
            underlying_dataset = eval_loader.dataset
            if isinstance(underlying_dataset, torch.utils.data.Subset):
                original_dataset_index = underlying_dataset.indices[current_sample_index_in_dataloader]
                image_path = underlying_dataset.dataset.image_paths[original_dataset_index]
            else: 
                image_path = underlying_dataset.image_paths[current_sample_index_in_dataloader]

            sample_result = {"image_path": image_path, "true_label": true_label, "true_class": idx_to_class[true_label]}

            prompt_strategy_config = global_cfg['vlm']['prompt_config'] # Moved here for access

            if is_generative_vlm:
                raw_prediction_scores = vlm.predict(image, keywords_for_vlm)
                sample_result["raw_scores"] = raw_prediction_scores
                
                keyword_to_class_map = prompt_strategy_config.get('params', {}).get('keyword_to_class_map', {})
                predicted_label = -1 
                
                score_real = raw_prediction_scores.get(keywords_for_vlm[0], 0.0) if keywords_for_vlm and len(keywords_for_vlm)>0 else 0.0
                score_ai = raw_prediction_scores.get(keywords_for_vlm[1], 0.0) if keywords_for_vlm and len(keywords_for_vlm)>1 else 0.0

                if score_real > score_ai: 
                    predicted_label = LABEL_REAL
                elif score_ai > score_real: 
                    predicted_label = LABEL_AI
                elif score_real == 1.0 and score_ai == 1.0: 
                     predicted_label = global_cfg.get("tie_breaking_label_for_generative", LABEL_AI) 
                     sample_result["ambiguous_prediction"] = True
                else: 
                     predicted_label = global_cfg.get("default_label_for_generative_no_match", LABEL_AI) 
                     sample_result["no_keyword_match"] = True
                reasoning_kw_0 = keywords_for_vlm[0] if keywords_for_vlm and len(keywords_for_vlm)>0 else "N/A"
                reasoning_kw_1 = keywords_for_vlm[1] if keywords_for_vlm and len(keywords_for_vlm)>1 else "N/A"
                sample_result["predicted_label_reasoning"] = f"Scores - RealKW ('{reasoning_kw_0}'): {score_real}, AIKW ('{reasoning_kw_1}'): {score_ai}"
            else: 
                raw_prediction_scores = vlm.predict(image, prompts_for_vlm)
                sample_result["raw_scores"] = raw_prediction_scores
                
                prompt_to_class_map = prompt_strategy_config.get('params', {}).get('prompt_to_class_map')
                if not prompt_to_class_map:
                     prompt_to_class_map = {
                         prompts_for_vlm[0]: LABEL_REAL,
                         prompts_for_vlm[1]: LABEL_AI
                     } if len(prompts_for_vlm) >= 2 else {}

                best_prompt = None
                max_score = -float('inf')
                for prompt, score in raw_prediction_scores.items():
                    if score > max_score:
                        max_score = score
                        best_prompt = prompt
                
                if best_prompt and best_prompt in prompt_to_class_map:
                    predicted_label = prompt_to_class_map[best_prompt]
                else:
                    predicted_label = LABEL_AI # Default, consider making this configurable
                    if prompts_for_vlm and len(prompts_for_vlm) >=2 : # Simplified fallback
                        if best_prompt == prompts_for_vlm[0]: predicted_label = LABEL_REAL
                        elif best_prompt == prompts_for_vlm[1]: predicted_label = LABEL_AI
                    print(f"Warning for {model_run_name}: Best prompt '{best_prompt}' not in prompt_to_class_map or map is empty for {image_path}. Defaulting to {predicted_label}.")


                sample_result["best_prompt"] = best_prompt
                sample_result["max_score"] = max_score
            
            all_predictions.append(predicted_label)
            all_labels.append(true_label)
            sample_result["predicted_label"] = predicted_label
            sample_result["predicted_class"] = idx_to_class.get(predicted_label, "unknown")
            raw_results.append(sample_result)

    # 3. Calculate and Print Metrics
    print(f"Calculating metrics for {model_run_name}...")
    eval_metrics = evaluate_predictions(all_predictions, all_labels, class_names_for_eval)
    
    print(f"--- Evaluation Metrics for {model_run_name} ---")
    for metric, value in eval_metrics.items():
        if metric == "classification_report":
            print(f"{metric}:")
            for k, v in value.items():
                 print(f"  {k}: {v}")
        else:
            print(f"{metric}: {value}")
    print("--------------------------")

    # 4. Save Results
    output_dir_base = global_cfg.get("output_dir_base", "results/zero_shot_eval")
    # Create a subdirectory for this specific model run
    model_specific_output_dir = os.path.join(output_dir_base, model_run_name)
    os.makedirs(model_specific_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use model_run_name in filenames to distinguish if multiple configs are run (though timestamp helps)
    
    metrics_filename = os.path.join(model_specific_output_dir, f"metrics_{model_run_name}_{timestamp}.json")
    with open(metrics_filename, 'w') as f:
        json.dump(eval_metrics, f, indent=4)
    print(f"Evaluation metrics for {model_run_name} saved to: {metrics_filename}")

    raw_results_filename = os.path.join(model_specific_output_dir, f"raw_results_{model_run_name}_{timestamp}.json")
    with open(raw_results_filename, 'w') as f:
        json.dump(raw_results, f, indent=4)
    print(f"Raw per-sample results for {model_run_name} saved to: {raw_results_filename}")
    
    # Save the specific model_run_config used for this run (or part of the main config)
    # For simplicity, we still copy the main config, but label it with model_run_name
    copied_config_filename = os.path.join(model_specific_output_dir, f"config_used_for_{model_run_name}_{timestamp}.yaml")
    # To save only the relevant part:
    # with open(copied_config_filename, 'w') as f:
    #    json.dump({"model_run_config": model_run_config, "dataset_config": global_cfg['dataset'], "prompt_config": global_cfg['vlm']['prompt_config']}, f, indent=4)
    # For now, copy the whole config_path from main function context.
    # This requires passing config_path to this function or handling it in the main loop.
    # Let's adjust run_zero_shot_evaluation to handle this.

    print(f"--- Finished evaluation for model: {model_run_name} ---\n")
    return model_specific_output_dir # Return for potential further use (like copying config)

def run_zero_shot_evaluation(config_path: str):
    """
    Main function to run zero-shot evaluation for one or more VLMs.
    """
    # 1. Load Configuration
    print(f"Loading configuration from: {config_path}")
    cfg = load_yaml_config(config_path)

    # Set seed for reproducibility
    seed = cfg.get("random_seed", 42) # Default seed if not specified
    set_seed(seed)
    print(f"Random seed set to: {seed}")

    # GPU Configuration
    gpu_id = cfg.get("eval_gpu_id", 3) 
    if torch.cuda.is_available():
        if gpu_id is not None:
            try:
                torch.cuda.set_device(f"cuda:{gpu_id}")
                print(f"Using GPU: {gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
            except Exception as e:
                print(f"Could not set GPU to cuda:{gpu_id} ({e}). Using default GPU or CPU.")
        else: # User explicitly set to None or not specified, PyTorch will use default or CPU
             print(f"GPU ID not specified or set to null. Using default CUDA device or CPU if no CUDA.")
        if not torch.cuda.is_available(): # Check again after trying to set
             print("CUDA not available. Running on CPU.")
    else:
        print("CUDA not available. Running on CPU.")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu")

    # 2. Initialize Prompt Strategy (shared for all models in this run)
    print("Initializing shared prompt strategy...")
    prompt_strategy_config = cfg['vlm']['prompt_config']
    prompt_strategy = get_instance_from_config(prompt_strategy_config)
    print(f"Shared Prompt Strategy {type(prompt_strategy).__name__} loaded.")

    # 3. Prepare Dataset (shared for all models in this run)
    print("Preparing shared dataset...")
    dataset_config = cfg['dataset']
    # GenImageDataset specific: ensure class_to_idx matches our LABEL_REAL, LABEL_AI
    # The dataset_config should ideally have a class_to_idx that maps 'nature' to 0 and 'ai' to 1
    # For evaluation, we usually use the 'val' or a 'test' split.
    # The dataset root should be the specific generator folder you want to test, e.g., stable_diffusion_v_1_5/
    dataset_root = dataset_config['root_dir'] # This should point to the specific generator like /path/to/stable_diffusion_v_1_5
    eval_split = dataset_config.get('eval_split', 'val') # or 'test'
    num_samples_to_eval = dataset_config.get('num_samples_eval', None) # New: for sampling

    # The GenImageDataset takes the *parent* of train/val (e.g. imagenet_ai_0424_sdv5)
    # and then split='val' will look for imagenet_ai_0424_sdv5/val/ai and imagenet_ai_0424_sdv5/val/nature
    # So, dataset_root should be e.g. "/raid/dannyliu/dataset/GAI_Dataset/genimage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/"
    
    # We need to define the class names for evaluation metrics and for prompt strategy if it uses them
    # This should align with dataset_config['class_to_idx']
    # Default: {'nature': 0, 'ai': 1}
    class_to_idx = dataset_config.get('class_to_idx', {'nature': LABEL_REAL, 'ai': LABEL_AI})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names_for_eval = [idx_to_class[LABEL_REAL], idx_to_class[LABEL_AI]] # e.g. ['nature', 'ai']

    # The transform for VLM predict methods should ideally be minimal or identity,
    # as VLMs often have their own internal processors.
    # The GenImageDataset's default transform resizes and normalizes, which might be okay
    # or might need to be adjusted/removed if the VLM processor handles it.
    # For now, we use a minimal transform that just converts to PIL Image if not already.
    # Our VLM wrappers expect PIL.Image.Image.
    # GenImageDataset already loads PIL images and returns (transformed_tensor, label)
    # The VLM wrappers expect PIL images. So, we might need a custom dataset or modify GenImageDataset for this script
    # to return (PIL.Image, label) for zero-shot, or adapt the VLM wrappers.
    # For simplicity, let's assume we'll get PIL images from the dataset for now.
    # This might require a custom transform or a flag in GenImageDataset.
    
    # Let's create a version of the dataset that does not apply ToTensor or Normalize for VLM input
    # We will define a simple transform that only ensures it's RGB.
    from torchvision import transforms
    pil_transform = transforms.Compose([
        # transforms.Resize((224,224)), # VLMs might have their own size, CLIP does, LLaVA too
        # The VLM processors will handle resizing.
    ])

    eval_dataset = GenImageDataset(
        root_dir=dataset_root, 
        split=eval_split, 
        transform=pil_transform, # Pass minimal transform; VLM's processor will handle specifics
        class_to_idx=class_to_idx
    )
    # We need to modify GenImageDataset or use a different one if it doesn't return PIL images
    # when transform is minimal. Let's assume __getitem__ returns (PIL.Image, label) if transform is basic.
    # Forcing GenImageDataset to return PIL Images for VLM:
    # A quick fix would be to modify its __getitem__ or pass a special transform.
    # Let's assume we have a way to get (PIL.Image, label).
    # For this script, we'll override the transform in GenImageDataset to return PIL.
    # A cleaner way: Add a parameter to GenImageDataset like `return_pil_image=True`.
    # For now, we assume the loaded eval_dataset yields (PIL.Image, int_label)

    # Hacky way to ensure PIL images if GenImageDataset doesn't support it directly:
    # Wrap dataset or modify its __getitem__ logic for this script
    class VLMGenImageDataset(GenImageDataset):
        # __init__ can be inherited from GenImageDataset if no special init logic is needed for VLMGenImageDataset itself.
        # The GenImageDataset is initialized with transform=None by the calling code for this VLM path.
        def __getitem__(self, idx):
            img_path = self.image_paths[idx] # Populated by GenImageDataset's __init__
            label = self.labels[idx]         # Populated by GenImageDataset's __init__
            
            try:
                image_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {img_path} in VLMGenImageDataset: {e}.")
                # Propagate the error so DataLoader can handle it (e.g., skip if num_workers > 0 and it doesn't crash the worker)
                # Or raise a more specific error / return a placeholder if the main loop is robust to it.
                raise RuntimeError(f"Failed to load image {img_path}: {e}")
            
            # Return PIL image and label, as VLM wrappers expect PIL images.
            # Collate function will handle batching these.
            return image_pil, label

    full_eval_dataset = VLMGenImageDataset( # Changed variable name here
        root_dir=dataset_root,
        split=eval_split,
        transform=None, # No PyTorch transforms, VLM processor handles it
        class_to_idx=class_to_idx
    )

    if len(full_eval_dataset) == 0:
        print(f"Error: Full dataset is empty. Check path: {os.path.join(dataset_root, eval_split)}")
        return
    
    # Apply sampling if num_samples_to_eval is set
    if num_samples_to_eval is not None and num_samples_to_eval > 0 and num_samples_to_eval < len(full_eval_dataset):
        print(f"Randomly sampling {num_samples_to_eval} images from the full dataset ({len(full_eval_dataset)} total) using seed {seed}.")
        # Ensure torch.randperm uses the seed set earlier
        indices = torch.randperm(len(full_eval_dataset))[:num_samples_to_eval].tolist()
        eval_dataset_for_loader = torch.utils.data.Subset(full_eval_dataset, indices)
    else:
        eval_dataset_for_loader = full_eval_dataset
        if num_samples_to_eval is not None and num_samples_to_eval >= len(full_eval_dataset):
            print(f"Requested num_samples_eval ({num_samples_to_eval}) is >= total samples ({len(full_eval_dataset)}). Using all samples from full dataset.")
        elif num_samples_to_eval is None:
            print("num_samples_eval not specified. Using all samples from full dataset.")


    if len(eval_dataset_for_loader) == 0:
        print(f"Error: Dataset for loader is empty. Check path: {os.path.join(dataset_root, eval_split)} and sampling config.")
        return

    # Adjust print statement to reflect the loader's dataset size
    print(f"Dataset for DataLoader: {len(eval_dataset_for_loader)} samples (split: '{eval_split}', root: {dataset_root}).")

    # 4. Loop through each VLM configuration and run evaluation
    vlm_configs_to_run = cfg['vlm'].get('model_configs')
    if not vlm_configs_to_run:
        print("Error: No model configurations found in 'vlm.model_configs' in the YAML.")
        # Backward compatibility: try to run a single model if old format is detected
        if 'model_config' in cfg['vlm']:
            print("Found 'vlm.model_config'. Attempting to run as a single model.")
            # Create a dummy name for the run
            model_name_from_params = cfg['vlm']['model_config'].get('params',{}).get('model_name', 'default_model_run')
            vlm_configs_to_run = [{'name': model_name_from_params, 'model_config': cfg['vlm']['model_config']}]
        else:
            return

    for model_run_config in vlm_configs_to_run:
        model_specific_output_dir = run_evaluation_for_model(
            model_run_config=model_run_config,
            global_cfg=cfg,
            prompt_strategy=prompt_strategy,
            eval_dataset_for_loader=eval_dataset_for_loader,
            device=device, # device is passed but VLM wrappers manage their own internal device.
            class_names_for_eval=class_names_for_eval,
            idx_to_class=idx_to_class
        )
        
        # Save the main config file used for this run into the model-specific directory
        if model_specific_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Re-generate for unique config copy name
            model_run_name = model_run_config.get('name', 'unknown_model')
            copied_config_filename = os.path.join(model_specific_output_dir, f"config_used_MAIN_{model_run_name}_{timestamp}.yaml")
            shutil.copy(config_path, copied_config_filename)
            print(f"Main configuration file copied to: {copied_config_filename} for model {model_run_name}")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zero-Shot VLM Evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    run_zero_shot_evaluation(args.config)

# Example Usage (from terminal):
# python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml 