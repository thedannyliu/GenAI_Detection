import argparse
import os
import random
import json
import glob # For get_image_files
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset # Dataset may not be strictly needed if processing file by file
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoConfig
import shutil # Ensure shutil is imported

# Append project root to sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config_utils import load_yaml_config
# Assuming LinearClassifier is defined in the training script or a shared models file
# If it's in the training script, we might need to duplicate or move it.
# For now, let's assume it can be imported or defined locally if simple enough.
# from src.experiments.clip_linear_probe_train import LinearClassifier # This creates a dependency

# Define labels (can be overridden by dataset specific config if needed)
LABEL_NATURE = 0
LABEL_AI = 1

# Re-define or import LinearClassifier if not accessible otherwise
# For simplicity in this script, let's redefine it if it's straightforward.
class LinearClassifier(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dims: List[int], num_classes: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)

def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # For consistency with training if set
    print(f"Random seed set to: {seed_value}")

def get_image_files(folder_path: str) -> List[str]:
    image_extensions = ('*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.webp')
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return all_files

def load_models_and_processor(
    clip_model_id: str, 
    classifier_path: str, 
    classifier_config: Dict,
    embedding_dim: int, # Expected embedding dim, usually from CLIP vision_config.hidden_size
    device: torch.device
) -> Tuple[CLIPModel, CLIPProcessor, LinearClassifier]:
    print(f"Loading CLIP model: {clip_model_id}")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model_for_eval = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_model_for_eval.eval() # Freeze all CLIP model parameters
    for param in clip_model_for_eval.parameters():
        param.requires_grad = False
    print("CLIP model loaded and frozen.")

    # Determine embedding dimension correctly from projection_dim
    actual_embedding_dim = clip_model_for_eval.config.projection_dim

    classifier_path = classifier_path
    classifier_arch_cfg = classifier_config

    print(f"Loading Linear Classifier from: {classifier_path}")
    # Ensure this local definition matches the one used in training if parameters are to be compatible
    linear_classifier = LinearClassifier(
        embedding_dim=actual_embedding_dim, # Use the correct dimension
        hidden_dims=classifier_arch_cfg.get('hidden_dims', []),
        num_classes=classifier_arch_cfg['num_classes'],
        dropout_rate=classifier_arch_cfg.get('dropout', 0.0)
    ).to(device)
    linear_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    linear_classifier.eval()
    print("Linear Classifier loaded.")
    return clip_model_for_eval, clip_processor, linear_classifier

def infer_on_dataset(
    dataset_config: Dict,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    linear_classifier: LinearClassifier,
    device: torch.device,
    num_samples_per_folder: int = None, # from eval_cfg
    batch_size_embed: int = 32,    # from eval_cfg
    batch_size_classify: int = 128, # from eval_cfg. Not directly used if embeddings processed then classified
    global_seed: int = 42
) -> Dict:
    dataset_name = dataset_config["name"]
    ai_path = dataset_config["ai_path"]
    nature_path = dataset_config["nature_path"]
    ai_label = dataset_config["ai_label"]
    nature_label = dataset_config["nature_label"]

    print(f"\n--- Evaluating Dataset: {dataset_name} ---")

    # Get image paths
    ai_files_all = get_image_files(ai_path)
    nature_files_all = get_image_files(nature_path)

    if not ai_files_all:
        print(f"Warning: No AI images found in {ai_path} for dataset {dataset_name}. Skipping AI folder.")
    if not nature_files_all:
        print(f"Warning: No Nature images found in {nature_path} for dataset {dataset_name}. Skipping Nature folder.")

    # Sample images if num_samples_per_folder is specified
    # Seed for consistent sampling for this dataset run
    # Note: get_image_files doesn't use seed. random.sample below does.
    random.seed(global_seed) # Use global_seed + an offset or dataset-specific seed if needed for perfect run-to-run sample identity *per dataset*
                            # For now, one global seed means same sample selection if datasets are identical.

    sampled_ai_files = random.sample(ai_files_all, min(num_samples_per_folder, len(ai_files_all))) if num_samples_per_folder and ai_files_all else ai_files_all
    sampled_nature_files = random.sample(nature_files_all, min(num_samples_per_folder, len(nature_files_all))) if num_samples_per_folder and nature_files_all else nature_files_all
    
    image_paths = sampled_ai_files + sampled_nature_files
    true_labels = [ai_label] * len(sampled_ai_files) + [nature_label] * len(sampled_nature_files)

    if not image_paths:
        print(f"No images to evaluate for dataset {dataset_name} after sampling. Skipping.")
        return {
            "name": dataset_name, 
            "error": "No images found or sampled",
            "accuracy": 0, "report": {}, "num_samples": 0
        }

    print(f"Processing {len(image_paths)} images for {dataset_name} ({len(sampled_ai_files)} AI, {len(sampled_nature_files)} Nature)... ")

    all_embeddings_list = []
    # Extract embeddings in batches
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size_embed), desc=f"Embedding {dataset_name}"):
            batch_paths = image_paths[i:i + batch_size_embed]
            if not batch_paths: continue
            try:
                images_pil = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = clip_processor(text=None, images=images_pil, return_tensors="pt", padding=True).to(device)
                image_features = clip_model.get_image_features(inputs.pixel_values)
                all_embeddings_list.append(image_features.cpu())
            except Exception as e:
                print(f"Error extracting embeddings for a batch in {dataset_name} (starts with {batch_paths[0]}): {e}. Skipping batch.")
    
    if not all_embeddings_list:
        print(f"No embeddings extracted for {dataset_name}. Cannot evaluate.")
        return {"name": dataset_name, "error": "Embedding extraction failed", "accuracy": 0, "report": {}, "num_samples": 0}

    all_embeddings = torch.cat(all_embeddings_list, dim=0)
    
    # Classify using the linear classifier
    all_predictions = []
    linear_classifier.eval()
    with torch.no_grad():
        # The batch_size_classify from config is for this part
        for i in tqdm(range(0, len(all_embeddings), batch_size_classify), desc=f"Classifying {dataset_name}"):
            batch_embeddings = all_embeddings[i:i + batch_size_classify].to(device)
            outputs = linear_classifier(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            
    accuracy = accuracy_score(true_labels, all_predictions)
    # Ensure class_names are derived correctly based on how labels (ai_label, nature_label) are defined
    # This assumes they are 0 and 1, and ai_label=1, nature_label=0 or vice-versa.
    # The report needs target_names in order of index [0, 1, ...]
    # If nature_label is 0 and ai_label is 1, target_names=["nature", "ai"]
    # If nature_label is 1 and ai_label is 0, target_names=["ai", "nature"]
    
    # Determine target_names based on actual label values and their names
    # This is a bit simplistic, assumes only two classes with fixed names "nature" and "ai" for now
    # A more robust way would be to pass class_to_idx from the main config if it's globally defined
    # or use a fixed convention for this script. Let's use a convention: 0=Nature, 1=AI.
    unique_labels_in_dataset = sorted(list(set([nature_label, ai_label])))
    target_names_for_report = []
    if LABEL_NATURE in unique_labels_in_dataset:
        target_names_for_report.append(f"nature (label {LABEL_NATURE})") 
    if LABEL_AI in unique_labels_in_dataset:
        target_names_for_report.append(f"ai (label {LABEL_AI})")
    # If labels are something else, this needs adjustment. For now, expecting 0 and 1.
    if not all(l in [0,1] for l in unique_labels_in_dataset):
         print(f"Warning: Unusual labels {unique_labels_in_dataset} for dataset {dataset_name}. Report target names might be affected.")
         # Defaulting if complex
         target_names_for_report = [str(l) for l in unique_labels_in_dataset]
    elif len(unique_labels_in_dataset) == 2 : # common case 0 and 1
        # Ensure order based on numeric value for scikit-learn report
        if unique_labels_in_dataset[0] == LABEL_NATURE : # [nature (0), ai (1)]
            target_names_for_report = ["nature", "ai"]
        else: # [ai (0), nature (1)] - less common but possible if labels flipped
            target_names_for_report = ["ai", "nature"]
    else: # Single class dataset (or error)
        target_names_for_report = [target_names_for_report[0].split(' ')[0]] if target_names_for_report else ["unknown"]

    report = classification_report(true_labels, all_predictions, target_names=target_names_for_report, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, all_predictions, labels=unique_labels_in_dataset) # Use actual unique labels for CM
    report["confusion_matrix"] = cm.tolist()

    print(f"Dataset: {dataset_name} - Samples: {len(true_labels)}, Accuracy: {accuracy:.4f}")
    return {"name": dataset_name, "accuracy": accuracy, "report": report, "num_samples": len(true_labels)}

# --- Main script --- 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CLIP Linear Probe on various datasets.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML evaluation configuration file.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    print(f"Evaluation configuration loaded from {args.config}")

    general_cfg = cfg.get('general', {})
    model_cfg = cfg['model']
    eval_cfg = cfg['evaluation']

    seed = general_cfg.get('seed', 42)
    gpu_id = general_cfg.get('gpu_id', 0)
    output_base_dir = general_cfg.get('output_base_dir', 'results/clip_linear_probe_evaluations')

    set_seed(seed)
    device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Create output directory for this evaluation run
    config_filename_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = Path(output_base_dir) / f"{config_filename_stem}_{timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, current_run_output_dir / Path(args.config).name) # Save config
    print(f"Evaluation results will be saved to: {current_run_output_dir}")

    # Determine embedding dimension from CLIP model
    temp_clip_config = AutoConfig.from_pretrained(model_cfg['clip_model_id'])
    embedding_dim = temp_clip_config.vision_config.hidden_size

    clip_model, clip_processor, linear_classifier = load_models_and_processor(
        model_cfg['clip_model_id'],
        model_cfg['linear_classifier_path'],
        model_cfg.get('classifier_config', {}), # Pass classifier_config from model section
        embedding_dim,
        device
    )

    datasets_to_eval = eval_cfg.get('datasets', [])
    if not datasets_to_eval:
        print("No datasets specified for evaluation. Exiting.")
        sys.exit(0)
    
    # More to come: evaluation loop over datasets
    print("\nSetup complete. Starting evaluation loop...") 

    overall_evaluation_summary = {
        "eval_config_file": args.config,
        "trained_classifier_path": model_cfg['linear_classifier_path'],
        "clip_model_id": model_cfg['clip_model_id'],
        "seed": seed,
        "evaluation_timestamp": timestamp,
        "datasets_evaluated": []
    }

    num_s_per_folder = eval_cfg.get('num_samples_per_folder', None) # Can be None
    batch_embed = eval_cfg.get('batch_size_embed_extraction', 32)
    batch_classify = eval_cfg.get('batch_size_classifier_inference', 128)

    for dataset_conf in datasets_to_eval:
        dataset_results = infer_on_dataset(
            dataset_conf,
            clip_model,
            clip_processor,
            linear_classifier,
            device,
            num_samples_per_folder=num_s_per_folder,
            batch_size_embed=batch_embed,
            batch_size_classify=batch_classify,
            global_seed=seed # Pass seed for reproducible sampling per dataset if needed
        )
        overall_evaluation_summary["datasets_evaluated"].append(dataset_results)

    # Save overall summary
    summary_json_path = current_run_output_dir / "cross_dataset_evaluation_summary.json"
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(overall_evaluation_summary, f, indent=4)
        print(f"\nOverall evaluation summary saved to: {summary_json_path}")
    except Exception as e:
        print(f"Error saving overall evaluation summary: {e}")

    print("\nCLIP Linear Probe evaluation script finished.") 