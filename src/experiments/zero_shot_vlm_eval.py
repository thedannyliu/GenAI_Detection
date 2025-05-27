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

def evaluate_predictions(all_predictions, all_labels, model_name_unique, dataset_name, current_dataset_config, global_cfg_dict, specific_output_dir, all_prompt_score_outputs):
    # The signature of evaluate_predictions was changed in a previous step and seems to have been reverted partially in the latest file content.
    # Let's use the one that takes model_name_unique, prompt_strategy etc.
    # Also, `class_names_for_eval` is needed for classification_report target_names.
    # This needs to be passed or retrieved from global_cfg_dict and dataset_cfg_dict.
    
    class_to_idx = current_dataset_config.get('class_to_idx', {'nature': LABEL_REAL, 'ai': LABEL_AI})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # Ensure class_names_for_eval are in the correct order [REAL, AI]
    class_names_for_eval = [idx_to_class.get(LABEL_REAL, "real"), idx_to_class.get(LABEL_AI, "ai")]

    if not all_predictions or not all_labels:
        print(f"No predictions/labels for {model_name_unique} on {dataset_name}. Skipping metrics.")
        # Create a minimal metrics dict indicating failure/skip
        metrics = {"accuracy": 0, "num_samples": 0, "error": "No predictions or labels available."}
    elif len(all_predictions) != len(all_labels):
        print(f"Mismatch in predictions/labels count for {model_name_unique} on {dataset_name}. Skipping metrics.")
        metrics = {"accuracy": 0, "num_samples": 0, "error": "Mismatch in prediction/label count."}
    else:
        predictions_arr_orig = np.array(all_predictions)
        labels_arr_orig = np.array(all_labels)

        # Count and report samples predicted as class 2 (unknown/unclassified)
        unknown_predictions_count = np.sum(predictions_arr_orig == 2)
        metrics = {
            "num_total_samples": int(len(labels_arr_orig)), 
            "num_unknown_predictions": int(unknown_predictions_count)
        }

        # Filter out predictions and labels that are 2 for standard classification metrics
        mask_known = (predictions_arr_orig != 2) & (labels_arr_orig != 2) # Also filter labels if they could be 2
        # Actually, for zero-shot, true labels (labels_arr_orig) should only be 0 or 1.
        # We only expect predictions_arr_orig to potentially contain 2.
        mask_for_report = (labels_arr_orig != 2) # Should always be true for true labels 0 or 1
        
        # We evaluate metrics on samples where prediction was 0 or 1.
        # If a prediction is 2, it's an "unable to classify" case from our side.
        # The accuracy should reflect how well it did on the N-unknown_predictions_count samples.
        
        # Let's define "effective" predictions/labels as those not predicted as 2
        predictions_arr_eff = predictions_arr_orig[predictions_arr_orig != 2]
        labels_arr_eff = labels_arr_orig[predictions_arr_orig != 2] # Match the filtering

        if len(labels_arr_eff) > 0:
            accuracy_eff = np.mean(predictions_arr_eff == labels_arr_eff)
            metrics["accuracy_on_classified"] = float(accuracy_eff) # np.mean can return np.float64
            metrics["num_classified_samples"] = int(len(labels_arr_eff))

            # For classification report, use only the effectively classified samples
            # Ensure that the labels_arr_eff and predictions_arr_eff are what classification_report expects
            # (i.e., containing only labels 0 and 1 if class_names_for_eval is for 0 and 1)
            try:
                from sklearn.metrics import classification_report
                # Filter out any 2s from predictions if they somehow slipped through for the report against 0,1 labels
                report_predictions = predictions_arr_eff[np.isin(predictions_arr_eff, [LABEL_REAL, LABEL_AI])]
                report_labels = labels_arr_eff[np.isin(predictions_arr_eff, [LABEL_REAL, LABEL_AI])] # Match filtering based on predictions

                if len(report_labels) > 0:
                    report = classification_report(report_labels, report_predictions, target_names=class_names_for_eval, output_dict=True, zero_division=0)
                    metrics["classification_report"] = report
                    if class_names_for_eval[LABEL_REAL] in report and class_names_for_eval[LABEL_AI] in report:
                        metrics["precision_real"] = report[class_names_for_eval[LABEL_REAL]].get("precision", 0)
                        metrics["recall_real"] = report[class_names_for_eval[LABEL_REAL]].get("recall", 0)
                        metrics["f1_real"] = report[class_names_for_eval[LABEL_REAL]].get("f1-score", 0)
                        metrics["precision_ai"] = report[class_names_for_eval[LABEL_AI]].get("precision", 0)
                        metrics["recall_ai"] = report[class_names_for_eval[LABEL_AI]].get("recall", 0)
                        metrics["f1_ai"] = report[class_names_for_eval[LABEL_AI]].get("f1-score", 0)

                    # Calculate and add confusion matrix for classified samples
                    try:
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(report_labels, report_predictions, labels=[LABEL_REAL, LABEL_AI])
                        metrics["confusion_matrix_classified_0_1"] = cm.tolist() # Convert numpy array to list for JSON
                    except Exception as e_cm:
                        print(f"Error generating confusion matrix for {model_name_unique}: {e_cm}")
                        metrics["confusion_matrix_classified_0_1"] = "Error generating confusion matrix."
                else:
                    print(f"Warning: No samples left for classification report after filtering for model {model_name_unique}.")
                    metrics["classification_report"] = "No samples for report after filtering."

            except ImportError:
                print("scikit-learn not installed. Skipping classification report.")
            except Exception as e:
                print(f"Error generating classification report for {model_name_unique}: {e}")
        else:
            print(f"Warning: No samples classified as 0 or 1 for model {model_name_unique}. Accuracy is 0.")
            metrics["accuracy_on_classified"] = 0.0
            metrics["num_classified_samples"] = 0
        
        # Overall accuracy (counting 2 as incorrect) - this might be too harsh if 2 means "don't know"
        # For now, let's keep accuracy_on_classified as the primary accuracy.
        # We can add another accuracy: (correct_0_1_predictions) / total_samples
        correct_0_1_predictions_val = np.sum(predictions_arr_orig[labels_arr_orig != 2] == labels_arr_orig[labels_arr_orig != 2])
        metrics["accuracy_overall_vs_total"] = float(correct_0_1_predictions_val / len(labels_arr_orig)) if len(labels_arr_orig) > 0 else 0.0
        metrics["num_correctly_classified_0_1"] = int(correct_0_1_predictions_val) # Add this for clarity

    # Save metrics
    metrics_filename = os.path.join(specific_output_dir, f"metrics_{model_name_unique}_{dataset_name}.json")
    with open(metrics_filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {model_name_unique} on {dataset_name} saved to {metrics_filename}")

    # Save raw predictions and scores
    raw_output_filename = os.path.join(specific_output_dir, f"predictions_and_scores_{model_name_unique}_{dataset_name}.json")
    with open(raw_output_filename, 'w') as f:
        json.dump(all_prompt_score_outputs, f, indent=4)
    print(f"Raw predictions and scores for {model_name_unique} on {dataset_name} saved to {raw_output_filename}")
    return metrics # Or whatever this function is supposed to return

def setup_output_directory(output_dir_base: str, model_name_unique: str, dataset_name: str) -> str:
    # Sanitize dataset_identifier to be a valid path component
    sanitized_dataset_name = dataset_name.replace('/', '_').replace('\\', '_') # Basic sanitization
    # Truncate if too long
    max_len_dataset_id = 50
    if len(sanitized_dataset_name) > max_len_dataset_id:
        sanitized_dataset_name = sanitized_dataset_name[:max_len_dataset_id]
        
    # New structure: output_dir_base / model_name_unique / sanitized_dataset_name
    model_and_dataset_specific_output_dir = os.path.join(output_dir_base, model_name_unique, sanitized_dataset_name)
    os.makedirs(model_and_dataset_specific_output_dir, exist_ok=True)
    print(f"Output directory for {model_name_unique} on {dataset_name}: {model_and_dataset_specific_output_dir}")
    return model_and_dataset_specific_output_dir

def run_evaluation_for_model_on_dataset(model_name_unique: str, model_config_dict: Dict, prompt_strategy_config_dict: Dict, current_dataset_config: Dict, global_cfg_dict: Dict, config_path_for_copy: str):
    print(f"\nStarting evaluation for model: {model_name_unique} on dataset: {current_dataset_config['name']}")
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
    if hasattr(model, 'prompt_strategy'): # Check if model expects prompt_strategy to be set
        model.prompt_strategy = prompt_strategy # Assign prompt_strategy to the model instance
    else:
        print(f"Warning: Model {model_name_unique} does not have a 'prompt_strategy' attribute. Prompt strategy might not be used as expected by the model wrapper.")
    print("Prompt strategy initialized.")

    # 3. Setup Dataset and DataLoader
    # `eval_split` is defined in the current_dataset_config (after merging defaults)
    eval_split = current_dataset_config.get('eval_split', 'val') 
    print(f"Setting up dataset and loader for split: {eval_split} using dataset config: {current_dataset_config['name']}...")
    eval_dataset, eval_loader = setup_dataset_and_loader(current_dataset_config, global_cfg_dict, split=eval_split)

    if eval_dataset is None or eval_loader is None:
        print(f"Error: Could not setup dataset/loader for {model_name_unique} on dataset {current_dataset_config['name']}, split '{eval_split}'. Skipping evaluation for this dataset.")
        return
    print(f"Dataset and DataLoader for '{current_dataset_config['name']}' (split '{eval_split}') ready.")

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
        print(f"No predictions made for {model_name_unique} on {current_dataset_config['name']}. Skipping metrics calculation.")
        return

    output_dir_base = global_cfg_dict.get('output_dir_base', 'results/zero_shot_eval')
    # Specific output dir for this model and this dataset
    specific_output_dir = setup_output_directory(output_dir_base, model_name_unique, current_dataset_config['name'])
    
    evaluate_predictions(all_predictions, all_true_labels, model_name_unique, current_dataset_config['name'], current_dataset_config, global_cfg_dict, specific_output_dir, all_prompt_score_outputs)
    
    # Copy the main config file to this specific output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    copied_config_filename = os.path.join(specific_output_dir, f"config_used_MAIN_{model_name_unique}_{current_dataset_config['name']}_{timestamp}.yaml")
    try:
        shutil.copy(config_path_for_copy, copied_config_filename)
        print(f"Main configuration file copied to: {copied_config_filename} for model {model_name_unique} on dataset {current_dataset_config['name']}")
    except Exception as e_copy:
        print(f"Error copying main config file for {model_name_unique} on dataset {current_dataset_config['name']}: {e_copy}")

def setup_dataset_and_loader(dataset_cfg_for_current_run, global_cfg, split='val'):
    print(f"Setting up dataset for split: {split} from dataset config: {dataset_cfg_for_current_run.get('name', 'Unknown Dataset')}")
    data_root_dir = dataset_cfg_for_current_run['root_dir']
    class_to_idx = dataset_cfg_for_current_run.get('class_to_idx')
    
    # Get num_samples_per_class for dataset initialization
    # This new parameter allows specifying how many samples to load *per class* directly from GenImageDataset
    num_samples_per_class_to_load = dataset_cfg_for_current_run.get('num_samples_per_class_eval', None)

    split_dir = os.path.join(data_root_dir, split)
    if not os.path.exists(split_dir) or not os.path.isdir(split_dir):
        print(f"Error: Split directory {split_dir} does not exist or is not a directory.")
        return None, None

    full_dataset = VLMGenImageDataset(
        root_dir=data_root_dir, 
        split=split, 
        class_to_idx=class_to_idx,
        transform=None,
        num_samples_per_class=num_samples_per_class_to_load, # Pass this to GenImageDataset
        seed=global_cfg.get('random_seed', 42) # Ensure GenImageDataset also uses the seed for its sampling
    )

    if not full_dataset.image_paths:
        print(f"Error: Full dataset for split '{split}' is empty after initialization. Check dataset structure and paths.")
        return None, None
    print(f"Full dataset for '{split}' initialized with {len(full_dataset.image_paths)} images.")
    if num_samples_per_class_to_load:
        print(f"  (Targeted {num_samples_per_class_to_load} samples per class during initial load)")

    eval_dataset_for_loader = full_dataset
    
    # num_samples_eval is the total number of samples for the final loader, potentially after a second round of sampling.
    # If num_samples_per_class_to_load was used, full_dataset already reflects the desired sampled size based on per-class counts.
    # In this case, we should generally use all of full_dataset unless num_samples_eval is explicitly set to something *smaller*
    # than len(full_dataset), which would be an unusual override.
    
    num_samples_eval_total = dataset_cfg_for_current_run.get('num_samples_eval', None) 

    # If num_samples_per_class_to_load was specified and resulted in a dataset,
    # we typically want to use all of those loaded samples.
    # The subsequent block for num_samples_eval_total should only apply if num_samples_per_class_to_load was NOT used,
    # OR if num_samples_eval_total is meant to further subsample the already class-sampled dataset (less common).
    
    perform_secondary_sampling = True
    if num_samples_per_class_to_load is not None and len(full_dataset.image_paths) > 0:
        # If GenImageDataset already did per-class sampling, we likely don't want to sample again
        # unless num_samples_eval_total is explicitly set to a *smaller* value.
        print(f"Dataset was initialized with num_samples_per_class. Total loaded: {len(full_dataset)}.")
        if num_samples_eval_total is not None and num_samples_eval_total < len(full_dataset):
            print(f"  num_samples_eval_total ({num_samples_eval_total}) is set and is less than loaded. Will perform secondary stratified sampling.")
            # Proceed with secondary sampling
        elif num_samples_eval_total is not None and num_samples_eval_total >= len(full_dataset):
            print(f"  num_samples_eval_total ({num_samples_eval_total}) is >= loaded. Using all loaded samples.")
            perform_secondary_sampling = False
        else: # num_samples_eval_total is None
            print(f"  num_samples_eval_total is not set. Using all {len(full_dataset)} loaded samples (from per-class sampling).")
            perform_secondary_sampling = False
    
    if perform_secondary_sampling and num_samples_eval_total is not None and num_samples_eval_total > 0 and num_samples_eval_total < len(full_dataset):
        print(f"Attempting secondary sampling for {num_samples_eval_total} images from '{split}' split of {dataset_cfg_for_current_run.get('name')} ({len(full_dataset)} total currently loaded)...")
        labels_for_stratification = np.array(full_dataset.labels)
        unique_labels, counts = np.unique(labels_for_stratification, return_counts=True)
        print(f"Class distribution in full '{split}' set: {dict(zip(unique_labels, counts))}")

        # Conditions for attempting stratification:
        # 1. More than one class present.
        # 2. Desired sample size is less than total samples.
        # 3. Each class for stratification must have at least 2 samples if num_samples_eval_total is a fraction, 
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
        # If num_samples_eval_total is very small relative to class count, stratification might not be meaningful or possible
        # For train_test_split, stratify requires at least 2 members for any class if it's to be split.
        # If we are taking num_samples_eval_total, then remaining is len(full_dataset) - num_samples_eval_total.
        # Both parts need to respect class counts for stratify. Min count for any class for stratify is typically 2.
        # If num_samples_eval_total means we are selecting a small subset, we need to ensure min(counts) is large enough.
        # A simpler check: if any(counts < 2) and we are trying to stratify, it might be an issue.
        # For our purpose, we are forming one subset (the sampled one). 
        # sklearn stratify needs n_samples >= n_classes for y, and for test_size, it means each class in y must have at least n_splits (implicitly 1) examples.
        # The most restrictive is if a class has only 1 sample, it cannot be split. So it must go entirely to train or test.

        if can_stratify:
            try:
                full_indices = np.arange(len(full_dataset))
                _ , sampled_indices = train_test_split(
                    full_indices, 
                    test_size=int(num_samples_eval_total), 
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
            print(f"Using random sampling for {num_samples_eval_total} samples.")
            indices = np.random.choice(len(full_dataset), int(num_samples_eval_total), replace=False)
            eval_dataset_for_loader = torch.utils.data.Subset(full_dataset, indices)

    elif perform_secondary_sampling and (num_samples_eval_total is None or num_samples_eval_total <= 0): # Check perform_secondary_sampling here
        print(f"num_samples_eval_total not specified or invalid. Using full loaded dataset for split '{split}'.")
    # If num_samples_per_class_to_load was used and we decided not to do secondary sampling, this path is also hit.
    # The eval_dataset_for_loader is already full_dataset.

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
    
    # device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu") # Device is set inside run_evaluation_for_model_on_dataset

    # 2. Initialize Prompt Strategy (shared for all models in this run)
    print("Initializing shared prompt strategy config...")
    prompt_strategy_config = cfg['vlm']['prompt_config']
    
    # Get dataset defaults and list of datasets to evaluate
    dataset_defaults = cfg.get('dataset_defaults', {})
    evaluation_datasets = cfg.get('evaluation_datasets', [])

    if not evaluation_datasets:
        print("Error: No 'evaluation_datasets' found in the YAML configuration.")
        return
        
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
        model_name_unique = model_run_config['name']
        model_config_dict = model_run_config['model_config']
        
        print(f"\nProcessing Model: {model_name_unique}...")
        
        for dataset_entry in evaluation_datasets:
            # Explicitly resolve parameters for the current dataset, prioritizing dataset_entry over dataset_defaults
            dataset_name = dataset_entry.get('name')
            if not dataset_name:
                print(f"Skipping a dataset entry due to missing 'name': {dataset_entry}")
                continue
            
            dataset_root_dir = dataset_entry.get('root_dir')
            if not dataset_root_dir:
                print(f"Skipping dataset '{dataset_name}' due to missing 'root_dir'.")
                continue

            # Resolve eval_split: specific from dataset_entry, then from dataset_defaults, then fallback to 'val'
            resolved_eval_split = dataset_entry.get('eval_split', dataset_defaults.get('eval_split', 'val'))

            # Resolve class_to_idx: specific from dataset_entry, then from dataset_defaults, then fallback to {'nature': 0, 'ai': 1}
            default_class_map = {'nature': LABEL_REAL, 'ai': LABEL_AI} # Use defined labels
            resolved_class_to_idx = dataset_entry.get('class_to_idx', dataset_defaults.get('class_to_idx', default_class_map))
            
            # Resolve num_samples_eval: specific, then default, then None (meaning use all samples)
            resolved_num_samples_eval = dataset_entry.get('num_samples_eval', dataset_defaults.get('num_samples_eval', None))

            # Construct the specific configuration for the current dataset evaluation
            current_dataset_resolved_config = {
                'name': dataset_name,
                'root_dir': dataset_root_dir,
                'eval_split': resolved_eval_split,
                'class_to_idx': resolved_class_to_idx,
                'num_samples_eval': resolved_num_samples_eval
                # Add any other dataset-specific parameters here if needed in the future
            }
            
            print(f"  Preparing to evaluate on dataset: {current_dataset_resolved_config['name']} with config: {current_dataset_resolved_config}")
            
            # Pass the original config_path for copying, and the resolved dataset configuration
            run_evaluation_for_model_on_dataset(
                model_name_unique=model_name_unique,
                model_config_dict=model_config_dict,
                prompt_strategy_config_dict=prompt_strategy_config,
                current_dataset_config=current_dataset_resolved_config, # Use the explicitly resolved config
                global_cfg_dict=cfg, 
                config_path_for_copy=config_path
            )

    print("\nAll evaluations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Zero-Shot VLM Evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()
    
    run_zero_shot_evaluation(args.config)

# Example Usage (from terminal):
# python src/experiments/zero_shot_vlm_eval.py --config configs/vlm_zero_shot_custom.yaml 