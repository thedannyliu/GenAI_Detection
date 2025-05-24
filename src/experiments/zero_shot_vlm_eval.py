import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizer parallelism to avoid deadlocks with DataLoader workers
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import random # Added for seed setting
import shutil # Added for copying config
from sklearn.model_selection import train_test_split
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adjust import paths based on your project structure
from src.utils.config_utils import load_yaml_config, get_instance_from_config
from src.data_processing.custom_dataset import GenImageDataset # Assuming GenImageDataset is suitable

# Define labels for clarity, these should match your dataset's class_to_idx or be configurable
LABEL_REAL = 0  # Typically 'nature'
LABEL_AI = 1    # Typically 'ai'

# --- Define VLMGenImageDataset at the top level ---
class VLMGenImageDataset(GenImageDataset):
    # This dataset wrapper ensures that __getitem__ returns a PIL Image and its label,
    # which is expected by the VLM evaluation loop.
    # It inherits __init__ from GenImageDataset.
    # When used, it should be initialized with transform=None or a transform that
    # does not convert to PyTorch tensor, as VLM processors handle that.
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path} in VLMGenImageDataset: {e}.")
            # Depending on DataLoader behavior (e.g. num_workers > 0), errors here might be suppressed
            # or could halt execution. Raising RuntimeError ensures it's noticeable.
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        return image_pil, label
# --- End of VLMGenImageDataset definition ---

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

def evaluate_predictions(all_predictions, all_labels, model_name_unique, prompt_strategy, global_cfg_dict, model_specific_output_dir, all_prompt_score_outputs):
    # The signature of evaluate_predictions was changed in a previous step and seems to have been reverted partially in the latest file content.
    # Let's use the one that takes model_name_unique, prompt_strategy etc.
    # Also, `class_names_for_eval` is needed for classification_report target_names.
    # This needs to be passed or retrieved from global_cfg_dict and dataset_cfg_dict.
    
    dataset_cfg_dict = global_cfg_dict['dataset'] # Assuming dataset config is here
    class_to_idx = dataset_cfg_dict.get('class_to_idx', {'nature': LABEL_REAL, 'ai': LABEL_AI})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Ensure class_names_for_eval are in the correct order [REAL, AI]
    class_names_for_eval = [idx_to_class.get(LABEL_REAL, "real"), idx_to_class.get(LABEL_AI, "ai")]

    if not all_predictions or not all_labels:
        print(f"No predictions/labels for {model_name_unique}. Skipping metrics.")
        # Create a minimal metrics dict indicating failure/skip
        metrics = {"accuracy": 0, "num_samples": 0, "error": "No predictions or labels available."}
    elif len(all_predictions) != len(all_labels):
        print(f"Mismatch in predictions/labels count for {model_name_unique}. Skipping metrics.")
        metrics = {"accuracy": 0, "num_samples": 0, "error": "Mismatch in prediction/label count."}
    else:
        predictions_arr = np.array(all_predictions)
        labels_arr = np.array(all_labels)

        # print(f"Debug - Model: {model_name_unique}")
        # print(f"Debug: Unique predicted labels in evaluate_predictions: {np.unique(predictions_arr, return_counts=True)}")
        # print(f"Debug: True labels in evaluate_predictions ({len(labels_arr)} total): {np.unique(labels_arr, return_counts=True)}")

    accuracy = np.mean(predictions_arr == labels_arr)
    metrics = {"accuracy": accuracy, "num_samples": len(labels_arr)}
    try:
        from sklearn.metrics import classification_report
        report = classification_report(labels_arr, predictions_arr, target_names=class_names_for_eval, output_dict=True, zero_division=0)
        metrics["classification_report"] = report
        # Ensure class_names_for_eval[LABEL_REAL] and class_names_for_eval[LABEL_AI] correctly access the report keys
        if class_names_for_eval[LABEL_REAL] in report and class_names_for_eval[LABEL_AI] in report:
            metrics["precision_real"] = report[class_names_for_eval[LABEL_REAL]].get("precision", 0)
            metrics["recall_real"] = report[class_names_for_eval[LABEL_REAL]].get("recall", 0)
            metrics["f1_real"] = report[class_names_for_eval[LABEL_REAL]].get("f1-score", 0)
            metrics["precision_ai"] = report[class_names_for_eval[LABEL_AI]].get("precision", 0)
            metrics["recall_ai"] = report[class_names_for_eval[LABEL_AI]].get("recall", 0)
            metrics["f1_ai"] = report[class_names_for_eval[LABEL_AI]].get("f1-score", 0)
        else:
            print(f"Warning: Class names for report ({class_names_for_eval[LABEL_REAL]}, {class_names_for_eval[LABEL_AI]}) not found in classification_report keys: {list(report.keys())}")
    except ImportError:
        print("scikit-learn not installed. Skipping classification report.")
    except Exception as e:
        print(f"Error generating classification report for {model_name_unique}: {e}")
    
    # Save metrics
    metrics_filename = os.path.join(model_specific_output_dir, f"metrics_{model_name_unique}.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {model_name_unique} saved to {metrics_filename}")

    # Save raw predictions and scores
    raw_output_filename = os.path.join(model_specific_output_dir, f"predictions_and_scores_{model_name_unique}.json")
    with open(raw_output_filename, 'w') as f:
        json.dump(all_prompt_score_outputs, f, indent=4)
    print(f"Raw predictions and scores for {model_name_unique} saved to {raw_output_filename}")
    return metrics # Or whatever this function is supposed to return

def setup_output_directory(output_dir_base: str, model_name_unique: str, dataset_identifier: str) -> str:
    # Sanitize dataset_identifier to be a valid path component
    sanitized_dataset_id = dataset_identifier.replace('/', '_').replace('\\', '_') # Basic sanitization
    # Truncate if too long
    max_len_dataset_id = 50
    if len(sanitized_dataset_id) > max_len_dataset_id:
        sanitized_dataset_id = sanitized_dataset_id[:max_len_dataset_id]
        
    model_specific_output_dir = os.path.join(output_dir_base, sanitized_dataset_id, model_name_unique)
    os.makedirs(model_specific_output_dir, exist_ok=True)
    print(f"Output directory for {model_name_unique} on {sanitized_dataset_id}: {model_specific_output_dir}")
    return model_specific_output_dir

def run_evaluation_for_model(model_name_unique: str, model_config_dict: Dict, prompt_strategy_config_dict: Dict, dataset_cfg_dict: Dict, global_cfg_dict: Dict):
    print(f"\nStarting evaluation for model: {model_name_unique}")
    device = f"cuda:{global_cfg_dict['eval_gpu_id']}" if global_cfg_dict.get('eval_gpu_id') is not None and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Setup Model
    print("Initializing model (without passing device to constructor initially)...")
    model = get_instance_from_config(model_config_dict) # Always instantiate without device first
    
    # Then, move to the target device if necessary
    if hasattr(model, 'to') and callable(model.to):
        current_model_device_type = "unknown"
        if hasattr(model, 'device') and isinstance(model.device, torch.device):
             current_model_device_type = model.device.type

        # device is a string like "cuda:0" or "cpu". We need torch.device(device).type for comparison.
        target_device_obj = torch.device(device)
        if current_model_device_type != target_device_obj.type:
            try:
                print(f"Moving model to device {device} after instantiation.")
                model.to(device)
            except Exception as e_move:
                print(f"Error moving model to device {device}: {e_move}. Model might remain on {current_model_device_type}.")
        else:
            print(f"Model already on target device {device} ({current_model_device_type}) or handles device internally.")
    else:
        print(f"Warning: Model does not have a 'to' method. Device placement ({device}) might not be applied.")
    
    print(f"Model {model_name_unique} initialized.")

    # 2. Setup Prompt Strategy
    print("Initializing prompt strategy...")
    prompt_strategy = get_instance_from_config(prompt_strategy_config_dict)
    model.prompt_strategy = prompt_strategy # Assign prompt_strategy to the model instance
    print("Prompt strategy initialized.")

    # 3. Setup Dataset and DataLoader
    # `eval_split` is defined in the dataset_cfg_dict from YAML
    eval_split = dataset_cfg_dict.get('eval_split', 'val') 
    print(f"Setting up dataset and loader for split: {eval_split}...")
    eval_dataset, eval_loader = setup_dataset_and_loader(dataset_cfg_dict, global_cfg_dict, split=eval_split)

    if eval_dataset is None or eval_loader is None:
        print(f"Error: Could not setup dataset/loader for {model_name_unique} on split '{eval_split}'. Skipping evaluation.")
        return
    print(f"Dataset and DataLoader for '{eval_split}' ready.")

    all_predictions = []
    all_true_labels = []
    all_prompt_score_outputs = [] # To store detailed scores for each sample

    # Generate prompts once if they are static for all images (e.g., discriminative)
    # For generative prompts from GenImageDetectPrompts, this is more dynamic / per image, handled inside loop if needed.
    # Here, we assume get_prompts_for_image(None) gives the general set for discriminative VLMs.
    current_prompts = prompt_strategy.get_prompts_for_image(image=None) 
    if current_prompts is None:
        print(f"Warning: Prompt strategy for {model_name_unique} returned None for prompts. Using an empty list.")
        current_prompts = []
    elif not isinstance(current_prompts, list):
        print(f"Warning: Prompt strategy for {model_name_unique} returned a non-list ({type(current_prompts)}) for prompts. Using an empty list.")
        current_prompts = []
    print(f"Using prompts: {current_prompts}")

    # model.eval() # Set model to evaluation mode
    # Attempt to set the underlying model to evaluation mode
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'eval') and callable(model.model.eval):
            print(f"Calling model.model.eval() for {model_name_unique}")
            model.model.eval()
        elif hasattr(model, 'eval') and callable(model.eval): # Fallback if wrapper itself has eval
            print(f"Calling model.eval() for {model_name_unique}")
            model.eval()
        else:
            print(f"Warning: {model_name_unique} and its underlying model do not have a standard eval method. Skipping.")
    except Exception as e_eval:
        print(f"Exception while trying to set {model_name_unique} to eval mode: {e_eval}. Continuing...")

    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Evaluating {model_name_unique} on '{eval_split}' split"):
            batch_images_pil, batch_labels_tensor = batch_data
            
            # Assuming batch_size=1 from config for VLMs, so batch_images_pil is a list with one PIL image
            if not batch_images_pil:
                print(f"Warning: Empty image list in batch {i}. Skipping.")
                continue
            current_image_pil = batch_images_pil[0]
            current_label_int = batch_labels_tensor[0].item()

            try:
                # VLM's predict_batch should return List[Dict[str, float]]
                # For bsize=1, it's List with 1 Dict: [{prompt: score, ...}]
                prompt_scores_list = model.predict_batch([current_image_pil], current_prompts) 

                if not prompt_scores_list or not prompt_scores_list[0]:
                    print(f"Warning: predict_batch for {model_name_unique} returned empty or invalid scores for image {i}. Using default label.")
                    predicted_label = prompt_strategy.default_label_if_no_match
                    prompt_scores_output = {}
                else:
                    prompt_scores = prompt_scores_list[0]
                    prompt_scores_output = prompt_scores # For saving
                    
                    if not prompt_scores: # Should be caught above, but double check
                        best_prompt_text = None
                    else:
                        best_prompt_text = max(prompt_scores, key=prompt_scores.get)
                    
                    predicted_label = None
                    if best_prompt_text:
                        # This is where GenImageDetectPrompts.get_class_for_prompt (with its debug logs) will be called.
                        predicted_label = prompt_strategy.get_class_for_prompt(best_prompt_text)
                    
                    if predicted_label is None: # Fallback if prompt not in map or no best_prompt
                        if best_prompt_text:
                            # The warning about specific prompt not mapping is now inside get_class_for_prompt.
                            # Here we just note that we are using a fallback.
                            print(f"Info (run_evaluation_for_model): For {model_name_unique}, best prompt was '{best_prompt_text}'. Fallback label used after get_class_for_prompt returned None.")
                        else:
                            print(f"Info (run_evaluation_for_model): For {model_name_unique}, no best prompt from scores. Fallback label used.")
                        predicted_label = prompt_strategy.default_label_if_no_match

            except Exception as e:
                print(f"Error during prediction for {model_name_unique}, image {i}: {e}. Using default label.")
                # import traceback
                # traceback.print_exc()
                predicted_label = prompt_strategy.default_label_if_no_match
                prompt_scores_output = {"error": str(e)}
            
            all_predictions.append(predicted_label)
            all_true_labels.append(current_label_int)
            all_prompt_score_outputs.append({"image_idx": i, "true_label": current_label_int, "predicted_label": predicted_label, "scores": prompt_scores_output})

    # After loop, evaluate predictions
    print(f"\nFinished predictions for {model_name_unique}.")
    if not all_predictions or not all_true_labels:
        print(f"No predictions made for {model_name_unique}. Skipping metrics calculation.")
        return

    output_dir_base = global_cfg_dict.get('output_dir_base', 'results/zero_shot_eval')
    model_specific_output_dir = setup_output_directory(output_dir_base, model_name_unique, dataset_cfg_dict.get('root_dir', 'unknown_dataset'))
    
    evaluate_predictions(all_predictions, all_true_labels, model_name_unique, prompt_strategy, global_cfg_dict, model_specific_output_dir, all_prompt_score_outputs)

def setup_dataset_and_loader(dataset_cfg, global_cfg, split='val'):
    print(f"Setting up dataset for split: {split}")
    data_root_dir = dataset_cfg['root_dir']
    class_to_idx = dataset_cfg.get('class_to_idx')
    
    # Ensure the split-specific directory exists (e.g., data_root_dir/val)
    split_dir = os.path.join(data_root_dir, split)
    if not os.path.exists(split_dir) or not os.path.isdir(split_dir):
        print(f"Error: Split directory {split_dir} does not exist or is not a directory.")
        return None, None

    full_dataset = VLMGenImageDataset(
        root_dir=data_root_dir, 
        split=split, 
        class_to_idx=class_to_idx,
        transform=None 
    )

    if not full_dataset.image_paths:
        print(f"Error: Full dataset for split '{split}' is empty after initialization. Check dataset structure and paths.")
        return None, None
    print(f"Full dataset for '{split}' initialized with {len(full_dataset.image_paths)} images.")
    # print(f"Example labels from full_dataset: {full_dataset.labels[:10]}")
    # print(f"Class to index map used by dataset: {full_dataset.class_to_idx}")
    # --- Added detailed debug prints for dataset contents ---
    # print(f"Debug Dataset Check: Root directory used: {full_dataset.root_dir}")
    # print(f"Debug Dataset Check: Split used: {full_dataset.split}")
    # print(f"Debug Dataset Check: Class to index map in dataset: {full_dataset.class_to_idx}")
    # if hasattr(full_dataset, 'image_paths') and full_dataset.image_paths:
    #     print(f"Debug Dataset Check: Example image paths (first 3): {full_dataset.image_paths[:3]}")
    # else:
    #     print("Debug Dataset Check: full_dataset.image_paths is empty or not available.")
    # if hasattr(full_dataset, 'labels') and full_dataset.labels:
    #     print(f"Debug Dataset Check: Example labels (first 3): {full_dataset.labels[:3]}")
    #     unique_labels_debug, counts_debug = np.unique(np.array(full_dataset.labels), return_counts=True)
    #     print(f"Debug Dataset Check: Initial class distribution in full_dataset (from VLMGenImageDataset directly): {dict(zip(unique_labels_debug, counts_debug))}")
    # else:
    #     print("Debug Dataset Check: full_dataset.labels is empty or not available for initial distribution check.")
    # --- End of added debug prints ---

    eval_dataset_for_loader = full_dataset
    num_samples_eval = dataset_cfg.get('num_samples_eval', None)

    if num_samples_eval is not None and num_samples_eval > 0 and num_samples_eval < len(full_dataset):
        print(f"Attempting to sample {num_samples_eval} images from '{split}' split ({len(full_dataset)} total)...")
        labels_for_stratification = np.array(full_dataset.labels)
        unique_labels, counts = np.unique(labels_for_stratification, return_counts=True)
        print(f"Class distribution in full '{split}' set: {dict(zip(unique_labels, counts))}")

        # Conditions for attempting stratification:
        # 1. More than one class present.
        # 2. Desired sample size is less than total samples.
        # 3. Each class for stratification must have at least 2 samples if num_samples_eval is a fraction, 
        #    or more generally, enough samples for train_test_split to work without error.
        #    A common threshold is that n_splits (implicitly 1 for test_size) must be <= n_samples_per_class.
        #    So, each class should have at least 1 sample, but practically sklearn might need more for some scenarios.
        
        can_stratify = True
        if len(unique_labels) < 2:
            print("Warning: Only one class present in the dataset. Cannot perform stratified sampling.")
            can_stratify = False
        elif any(c < 1 for c in counts): # Should not happen if dataset loaded correctly
            print("Warning: At least one class has zero samples. Cannot perform stratified sampling.")
            can_stratify = False
        # If num_samples_eval is very small relative to class count, stratification might not be meaningful or possible
        # For train_test_split, stratify requires at least 2 members for any class if it's to be split.
        # If we are taking num_samples_eval, then remaining is len(full_dataset) - num_samples_eval.
        # Both parts need to respect class counts for stratify. Min count for any class for stratify is typically 2.
        # If num_samples_eval means we are selecting a small subset, we need to ensure min(counts) is large enough.
        # A simpler check: if any(counts < 2) and we are trying to stratify, it might be an issue.
        # For our purpose, we are forming one subset (the sampled one). 
        # sklearn stratify needs n_samples >= n_classes for y, and for test_size, it means each class in y must have at least n_splits (implicitly 1) examples.
        # The most restrictive is if a class has only 1 sample, it cannot be split. So it must go entirely to train or test.

        if can_stratify:
            try:
                full_indices = np.arange(len(full_dataset))
                _ , sampled_indices = train_test_split(
                    full_indices, 
                    test_size=int(num_samples_eval), 
                    stratify=labels_for_stratification, 
                    random_state=global_cfg.get('random_seed', None)
                )
                eval_dataset_for_loader = torch.utils.data.Subset(full_dataset, sampled_indices)
                print(f"Stratified sampling successful. Sampled {len(eval_dataset_for_loader)} images.")
                sampled_labels = [full_dataset.labels[i] for i in sampled_indices]
                unique_sampled, counts_sampled = np.unique(sampled_labels, return_counts=True)
                print(f"Sampled subset class distribution: {dict(zip(unique_sampled, counts_sampled))}")
            except ValueError as e:
                print(f"Warning: Stratified sampling failed ({e}). Falling back to random sampling.")
                can_stratify = False # Mark to fallback
        
        if not can_stratify: # Fallback to random sampling
            print(f"Using random sampling for {num_samples_eval} samples.")
            indices = np.random.choice(len(full_dataset), int(num_samples_eval), replace=False)
            eval_dataset_for_loader = torch.utils.data.Subset(full_dataset, indices)

    elif num_samples_eval is not None and num_samples_eval >= len(full_dataset):
        print(f"num_samples_eval ({num_samples_eval}) is >= total dataset size ({len(full_dataset)}). Using full dataset for split '{split}'.")
    elif num_samples_eval is None or num_samples_eval <= 0:
        print(f"num_samples_eval not specified or invalid. Using full dataset for split '{split}'.")

    if len(eval_dataset_for_loader) == 0:
        print(f"Error: Dataset for loader (split '{split}') is empty after sampling/selection. Cannot create DataLoader.")
        return None, None

    print(f"Final dataset size for loader for split '{split}': {len(eval_dataset_for_loader)}")
    
    batch_size = global_cfg.get('batch_size', 1)
    num_workers = global_cfg.get('num_workers', 2)
    data_loader = DataLoader(
        eval_dataset_for_loader,
        batch_size=batch_size, 
        shuffle=False, # Usually false for evaluation
        num_workers=num_workers, 
        pin_memory=True,
        collate_fn=vlm_collate_fn 
    )
    print(f"DataLoader created for split '{split}' with batch_size={batch_size}, num_workers={num_workers}.")
    return eval_dataset_for_loader, data_loader

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
    gpu_id = cfg.get("eval_gpu_id", 0) 
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
    # prompt_strategy instance is now created inside run_evaluation_for_model

    dataset_config = cfg['dataset']
    
    # --- Define class_to_idx and idx_to_class here ---
    # This should align with dataset_config['class_to_idx'] from YAML
    # Default: {'nature': 0, 'ai': 1}
    class_to_idx = dataset_config.get('class_to_idx', {'nature': LABEL_REAL, 'ai': LABEL_AI})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # class_names_for_eval will be derived inside evaluate_predictions based on these
    # --- End definition ---
    
    # The VLMGenImageDataset definition is now at the top level of the file.
    # The local instantiation of dataset/dataloader was also moved into run_evaluation_for_model.
    # The main loop will call run_evaluation_for_model which internally calls setup_dataset_and_loader.
    
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
        # Note: prompt_strategy is now initialized inside run_evaluation_for_model
        model_specific_output_dir = run_evaluation_for_model(
            model_name_unique=model_run_config['name'],
            model_config_dict=model_run_config['model_config'],
            prompt_strategy_config_dict=prompt_strategy_config, # Pass the config dict here
            dataset_cfg_dict=dataset_config,
            global_cfg_dict=cfg
        )
        
        if model_specific_output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_run_name = model_run_config.get('name', 'unknown_model') # get name again for safety
            copied_config_filename = os.path.join(model_specific_output_dir, f"config_used_MAIN_{model_run_name}_{timestamp}.yaml")
            try:
                shutil.copy(config_path, copied_config_filename)
                print(f"Main configuration file copied to: {copied_config_filename} for model {model_run_name}")
            except Exception as e_copy:
                print(f"Error copying config file for {model_run_name}: {e_copy}")

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zero-Shot VLM Evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    run_zero_shot_evaluation(args.config)

# Example Usage (from terminal):
# python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml 