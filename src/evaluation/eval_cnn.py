import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import argparse
import yaml # Added for YAML config loading
import json # Added for JSON output
from datetime import datetime # Added for timestamped output directory

# Add project root to sys.path to allow imports from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.baseline_classifiers import ResNet50Classifier

# Define the same transformations as used during training
def get_default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Using deterministic algorithms can have a performance impact
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False 
        # For inference, benchmark = True might be faster if input sizes don't change
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def load_model(model_path: str, num_classes: int = 2, device: torch.device = torch.device("cpu")) -> ResNet50Classifier:
    """Loads the ResNet50Classifier model with state_dict from model_path."""
    model = ResNet50Classifier(pretrained=False, num_classes=num_classes) # pretrained=False as we are loading our own weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path} and moved to {device}")
    return model

def infer_on_folder(
    model: nn.Module,
    folder_path: str,
    target_label: int,
    transform: transforms.Compose,
    device: torch.device,
    num_samples: int = 500,
    seed: int = 42
) -> tuple[int, int, float]:
    """Performs inference on a random sample of images from a folder."""
    random.seed(seed) # Seed for sampling consistency for this folder
    
    image_extensions = ('*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.webp') # Common image extensions
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        # Check for images in direct subdirectories (e.g. if folder_path is 'val' and images are in 'val/cls_name/img.jpg')
        # However, the prompt implies images are directly under the specified ai/nature folders.
        # If a more nested structure is common, this glob pattern could be extended, e.g., os.path.join(folder_path, '*', ext)

    if not all_image_paths:
        print(f"No images found in {folder_path}")
        return 0, 0, 0.0

    if len(all_image_paths) < num_samples:
        print(f"Warning: Requested {num_samples} samples from {folder_path}, but only found {len(all_image_paths)}. Using all available.")
        num_samples_to_take = len(all_image_paths)
        sampled_image_paths = all_image_paths
    else:
        num_samples_to_take = num_samples
        sampled_image_paths = random.sample(all_image_paths, num_samples_to_take)

    correct_predictions = 0
    
    with torch.no_grad():
        for img_path in tqdm(sampled_image_paths, desc=f"Inferring on {os.path.basename(folder_path)}", leave=False):
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0).to(device)
                outputs = model(image_tensor)
                _, predicted_class = torch.max(outputs, 1)
                if predicted_class.item() == target_label:
                    correct_predictions += 1
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                num_samples_to_take -=1 # Adjust total if an image fails

    accuracy = (correct_predictions / num_samples_to_take) * 100 if num_samples_to_take > 0 else 0.0
    print(f"Folder: {os.path.basename(folder_path)} - Correct: {correct_predictions}/{num_samples_to_take}, Accuracy: {accuracy:.2f}%")
    return correct_predictions, num_samples_to_take, accuracy

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained CNN model using a YAML config file.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file (e.g., configs/eval_cnn_config.yaml).")
    # Removed old argparse arguments for model_path, num_samples_per_folder, seed, gpu_id
    
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing configuration file {args.config}: {e}")
        sys.exit(1)

    general_config = config.get('general', {})
    model_config = config.get('model', {})
    eval_config = config.get('evaluation', {})

    seed = general_config.get('seed', 42)
    gpu_id = general_config.get('gpu_id', 0)
    model_path = model_config.get('path', "results/cnn_output_base/resnet50_run1/checkpoints/best_model.pth")
    num_classes = model_config.get('num_classes', 2)
    num_samples_per_folder = eval_config.get('num_samples_per_folder', 500)
    datasets_to_evaluate_config = eval_config.get('datasets', []) # Renamed to avoid conflict
    output_base_dir = eval_config.get('output_base_dir', 'results/cnn_evaluations/')

    if not datasets_to_evaluate_config:
        print("Error: No datasets specified in the configuration file under evaluation.datasets")
        sys.exit(1)

    set_seed(seed)
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(model_path=model_path, num_classes=num_classes, device=device)
    transform = get_default_transform()

    # --- Prepare output directory ---
    config_filename_stem = Path(args.config).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_output_dir = Path(output_base_dir) / f"{config_filename_stem}_{timestamp}"
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {current_run_output_dir}")
    # --- End output directory preparation ---

    overall_total_correct = 0
    overall_total_samples = 0
    
    evaluation_summary = {
        "config_file": args.config,
        "model_path": model_path,
        "seed": seed,
        "num_samples_per_folder": num_samples_per_folder,
        "execution_timestamp": timestamp,
        "datasets": []
    }

    print(f"\nStarting inference on {len(datasets_to_evaluate_config)} datasets, sampling {num_samples_per_folder} images per subfolder...")
    print(f"Loaded configuration from: {args.config}")
    print("Class mapping: AI/Fake -> configured 'ai_label', Nature/Real -> configured 'nature_label'\n")

    for dataset_config in datasets_to_evaluate_config: 
        dataset_name = dataset_config.get("name", "Unnamed Dataset")
        ai_path = dataset_config.get("ai_path")
        nature_path = dataset_config.get("nature_path")
        ai_label = dataset_config.get("ai_label")
        nature_label = dataset_config.get("nature_label")

        if not all([ai_path, nature_path, isinstance(ai_label, int), isinstance(nature_label, int)]):
            print(f"Skipping dataset '{dataset_name}' due to missing path or invalid label configuration.")
            continue
            
        print(f"--- Evaluating Dataset: {dataset_name} ---")
        
        dataset_total_correct = 0
        dataset_total_samples = 0

        dataset_results_summary = {
            "name": dataset_name,
            "ai_folder": {"path": ai_path, "label": ai_label, "correct": 0, "total_samples": 0, "accuracy": 0.0},
            "nature_folder": {"path": nature_path, "label": nature_label, "correct": 0, "total_samples": 0, "accuracy": 0.0},
            "overall_correct": 0,
            "overall_total_samples": 0,
            "overall_accuracy": 0.0
        }

        # Infer on AI/Fake folder
        print(f"Processing AI/Fake folder for {dataset_name}...")
        ai_correct, ai_samples, ai_accuracy = infer_on_folder(
            model, ai_path, ai_label, transform, device, 
            num_samples=num_samples_per_folder, seed=seed
        )
        dataset_total_correct += ai_correct
        dataset_total_samples += ai_samples
        dataset_results_summary["ai_folder"].update({"correct": ai_correct, "total_samples": ai_samples, "accuracy": ai_accuracy})

        # Infer on Nature/Real folder
        print(f"Processing Nature/Real folder for {dataset_name}...")
        nature_correct, nature_samples, nature_accuracy = infer_on_folder(
            model, nature_path, nature_label, transform, device,
            num_samples=num_samples_per_folder, seed=seed + 1 
        )
        dataset_total_correct += nature_correct
        dataset_total_samples += nature_samples
        dataset_results_summary["nature_folder"].update({"correct": nature_correct, "total_samples": nature_samples, "accuracy": nature_accuracy})
        
        if dataset_total_samples > 0:
            dataset_accuracy = (dataset_total_correct / dataset_total_samples) * 100
            print(f"--- Dataset: {dataset_name} - Overall Correct: {dataset_total_correct}/{dataset_total_samples}, Accuracy: {dataset_accuracy:.2f}% ---\n")
            dataset_results_summary.update({"overall_correct": dataset_total_correct, "overall_total_samples": dataset_total_samples, "overall_accuracy": dataset_accuracy})
        else:
            print(f"--- Dataset: {dataset_name} - No samples processed. --- \n")
            
        evaluation_summary["datasets"].append(dataset_results_summary)
        overall_total_correct += dataset_total_correct
        overall_total_samples += dataset_total_samples

    if overall_total_samples > 0:
        overall_accuracy = (overall_total_correct / overall_total_samples) * 100
        print(f"=== Overall Inference Summary ===")
        print(f"Total Correct Predictions: {overall_total_correct}/{overall_total_samples}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        evaluation_summary["overall_summary"] = {
            "total_correct": overall_total_correct,
            "total_samples": overall_total_samples,
            "accuracy": overall_accuracy
        }
    else:
        print("=== No samples processed overall. ===")
        evaluation_summary["overall_summary"] = {
            "total_correct": 0,
            "total_samples": 0,
            "accuracy": 0.0
        }

    # Save evaluation summary to JSON
    summary_json_path = current_run_output_dir / "evaluation_summary.json"
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(evaluation_summary, f, indent=4)
        print(f"Evaluation summary saved to: {summary_json_path}")
    except Exception as e:
        print(f"Error saving evaluation summary to {summary_json_path}: {e}")

if __name__ == "__main__":
    main() 