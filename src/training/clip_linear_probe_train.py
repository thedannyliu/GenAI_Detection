import argparse
import os
import random
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # Though CLIP processor handles this
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, AutoConfig

# Append project root to sys.path to allow imports from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Adjust if necessary

from src.utils.config_utils import load_yaml_config # Assuming this utility exists

# Define labels based on common practice, can be overridden by config
LABEL_NATURE = 0
LABEL_AI = 1

def set_seed(seed_value: int):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed_value}")

def setup_output_dirs(output_dir_base: str, experiment_name: str, config_path: str) -> Path:
    """Creates output directories for the experiment and saves the config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = Path(output_dir_base) / f"{experiment_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save the config file used for this run
    shutil.copy(config_path, run_output_dir / Path(config_path).name)
    print(f"Output directory: {run_output_dir}")
    return run_output_dir

def get_image_files(folder_path: str) -> List[str]:
    """Gets all image file paths from a folder."""
    image_extensions = ('*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.webp')
    all_files = []
    for ext in image_extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return all_files

# --- Dataset for embeddings ---
class ImageEmbeddingDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# --- Data preparation functions ---
def prepare_image_paths(
    base_path: str, 
    ai_folder: str, 
    nature_folder: str, 
    num_samples_per_class: int, 
    dataset_type: str,
    seed: int,
    existing_paths: List[str] = None # Used for test set to ensure no overlap with val set
) -> Tuple[List[str], List[int]]:
    """Samples image paths and assigns labels."""
    ai_image_folder = Path(base_path) / ai_folder
    nature_image_folder = Path(base_path) / nature_folder

    ai_files_all = get_image_files(str(ai_image_folder))
    nature_files_all = get_image_files(str(nature_image_folder))

    if not ai_files_all:
        raise FileNotFoundError(f"No AI images found in {ai_image_folder} for {dataset_type} set.")
    if not nature_files_all:
        raise FileNotFoundError(f"No Nature images found in {nature_image_folder} for {dataset_type} set.")

    # For test set, filter out paths already used by validation set
    if existing_paths:
        ai_files_all = [p for p in ai_files_all if p not in existing_paths]
        nature_files_all = [p for p in nature_files_all if p not in existing_paths]

    if len(ai_files_all) < num_samples_per_class:
        print(f"Warning: Requested {num_samples_per_class} AI samples for {dataset_type}, but only {len(ai_files_all)} available. Using all.")
        num_ai_to_sample = len(ai_files_all)
    else:
        num_ai_to_sample = num_samples_per_class

    if len(nature_files_all) < num_samples_per_class:
        print(f"Warning: Requested {num_samples_per_class} Nature samples for {dataset_type}, but only {len(nature_files_all)} available. Using all.")
        num_nature_to_sample = len(nature_files_all)
    else:
        num_nature_to_sample = num_samples_per_class
    
    # Set random seed for consistent sampling for this specific call if needed, 
    # but global seed should handle broader reproducibility.
    # random.seed(seed) # Re-seeding here might be redundant if global seed is well-managed

    sampled_ai_files = random.sample(ai_files_all, num_ai_to_sample)
    sampled_nature_files = random.sample(nature_files_all, num_nature_to_sample)

    image_paths = sampled_ai_files + sampled_nature_files
    # Labels from config or use defaults
    # Assuming class_to_idx: {'nature': 0, 'ai': 1}
    labels = [LABEL_AI] * len(sampled_ai_files) + [LABEL_NATURE] * len(sampled_nature_files)
    
    print(f"Prepared {dataset_type} set: {len(sampled_ai_files)} AI images, {len(sampled_nature_files)} Nature images.")
    return image_paths, labels

def extract_clip_embeddings(
    image_paths: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int = 32 # Batch size for CLIP processing
) -> torch.Tensor:
    """Extracts CLIP image embeddings for a list of image paths."""
    model.eval() # Ensure model is in eval mode
    all_embeddings = []
    num_batches = (len(image_paths) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting CLIP Embeddings"):
            batch_paths = image_paths[i*batch_size : (i+1)*batch_size]
            if not batch_paths: continue

            images_pil = [Image.open(p).convert("RGB") for p in batch_paths]
            
            try:
                inputs = processor(text=None, images=images_pil, return_tensors="pt", padding=True).to(device)
                image_features = model.get_image_features(inputs.pixel_values)
                all_embeddings.append(image_features.cpu())
            except Exception as e:
                print(f"Error processing batch starting with {batch_paths[0]}: {e}")
                # Add placeholder embeddings or skip if critical, here skipping by not appending
                # For robustness, could add zero tensors of correct shape, but this might skew training
                # For now, if a batch fails, it's skipped. Consider error handling strategy.
                pass # Or raise an error / add dummy embeddings

    if not all_embeddings:
        # This case should be rare if image_paths is not empty and images are valid
        # Determine expected embedding dimension from model config
        clip_config = AutoConfig.from_pretrained(model.config.name_or_path)
        embedding_dim = clip_config.vision_config.hidden_size 
        return torch.empty(0, embedding_dim) # Return empty tensor with correct second dimension
        
    return torch.cat(all_embeddings, dim=0)

# --- 3. Define Linear Classifier ---
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

# --- 4. Training and Evaluation Functions ---
def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device,
    use_amp: bool = False # Automatic Mixed Precision
):
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for embeddings, labels in tqdm(loader, desc="Training Epoch", leave=False):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device,
    use_amp: bool = False,
    is_test_set: bool = False, # To control tqdm description
    # Make cfg accessible for class names in report
    cfg_for_report: Dict = None 
) -> Tuple[float, float, Dict[str, Any], List[int], List[int]]: # accuracy, loss, report_dict, all_preds, all_true
    model.eval()
    total_loss = 0
    all_predictions = []
    all_true_labels = []

    desc = "Test Set Evaluation" if is_test_set else "Validation Set Evaluation"
    with torch.no_grad():
        for embeddings, labels in tqdm(loader, desc=desc, leave=False):
            embeddings, labels = embeddings.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    
    report_dict = {}
    if is_test_set and cfg_for_report: 
        current_class_to_idx = cfg_for_report.get('dataset', {}).get('class_to_idx', {'nature': LABEL_NATURE, 'ai': LABEL_AI})
        sorted_class_names = sorted(current_class_to_idx.keys(), key=lambda k: current_class_to_idx[k])
        
        report_dict = classification_report(all_true_labels, all_predictions, target_names=sorted_class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_true_labels, all_predictions, labels=[current_class_to_idx[k] for k in sorted_class_names])
        report_dict["confusion_matrix"] = cm.tolist()

    return accuracy, avg_loss, report_dict, all_predictions, all_true_labels

# --- Main script logic will follow ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Linear Probing Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration
    # cfg is now loaded globally in __main__
    cfg = load_yaml_config(args.config) 
    print(f"Configuration loaded from {args.config}")

    # General settings
    general_cfg = cfg.get('general', {})
    output_dir_base = general_cfg.get('output_dir_base', 'results/clip_linear_probe')
    experiment_name = general_cfg.get('experiment_name', 'clip_linear_probe_run')
    seed_val = general_cfg.get('seed', 42) # Renamed to avoid conflict with random.seed
    # Try 'eval_gpu_id' first, then 'gpu_id' as a fallback, then default to 0
    eval_gpu_id = general_cfg.get('eval_gpu_id', general_cfg.get('gpu_id', 2))

    # Setup
    set_seed(seed_val)
    run_output_dir = setup_output_dirs(output_dir_base, experiment_name, args.config)
    
    device_str = f"cuda:{eval_gpu_id}" if torch.cuda.is_available() and eval_gpu_id is not None else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Dataset settings
    dataset_cfg = cfg['dataset']
    clip_model_id = cfg['clip_model_id']
    dataloader_cfg = cfg['dataloader']

    # --- 1. Load CLIP Model and Processor ---
    print(f"Loading CLIP model: {clip_model_id}")
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_model.eval() # Freeze all CLIP model parameters
    for param in clip_model.parameters():
        param.requires_grad = False
    print("CLIP model loaded and frozen.")

    # --- 2. Prepare Data & Extract Embeddings ---
    print("Preparing datasets and extracting embeddings...")
    base_data_path = dataset_cfg['base_path']
    # class_to_idx is used from the global cfg in evaluate_model

    train_ai_path = dataset_cfg['train_ai_path']
    train_nature_path = dataset_cfg['train_nature_path']
    val_ai_path = dataset_cfg['val_ai_path']
    val_nature_path = dataset_cfg['val_nature_path']

    num_train_per_class = dataset_cfg['num_train_samples_per_class']
    num_val_per_class = dataset_cfg['num_val_samples_per_class']
    num_test_per_class = dataset_cfg['num_test_samples_per_class']

    # Prepare Train data
    train_image_paths, train_labels_list = prepare_image_paths(
        base_data_path, train_ai_path, train_nature_path, 
        num_train_per_class, "train", seed_val
    )
    train_embeddings = extract_clip_embeddings(train_image_paths, clip_model, clip_processor, device, batch_size=dataloader_cfg.get('batch_size', 32))
    train_labels = torch.tensor(train_labels_list, dtype=torch.long)
    train_dataset = ImageEmbeddingDataset(train_embeddings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=dataloader_cfg['batch_size'], shuffle=True, num_workers=dataloader_cfg['num_workers'])

    # Prepare Validation data
    val_image_paths, val_labels_list = prepare_image_paths(
        base_data_path, val_ai_path, val_nature_path, 
        num_val_per_class, "validation", seed_val + 1 
    )
    val_embeddings = extract_clip_embeddings(val_image_paths, clip_model, clip_processor, device, batch_size=dataloader_cfg.get('batch_size', 32))
    val_labels = torch.tensor(val_labels_list, dtype=torch.long)
    val_dataset = ImageEmbeddingDataset(val_embeddings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=cfg['evaluation'].get('batch_size', 128), shuffle=False, num_workers=dataloader_cfg['num_workers'])

    # Prepare Test data (ensure no overlap with validation)
    test_image_paths, test_labels_list = prepare_image_paths(
        base_data_path, val_ai_path, val_nature_path, 
        num_test_per_class, "test", seed_val + 2, 
        existing_paths=val_image_paths 
    )
    test_embeddings = extract_clip_embeddings(test_image_paths, clip_model, clip_processor, device, batch_size=cfg['evaluation'].get('batch_size', 128))
    test_labels = torch.tensor(test_labels_list, dtype=torch.long)
    test_dataset = ImageEmbeddingDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=cfg['evaluation'].get('batch_size', 128), shuffle=False, num_workers=dataloader_cfg['num_workers'])

    if train_embeddings.nelement() == 0 or val_embeddings.nelement() == 0 or test_embeddings.nelement() == 0:
        print("Error: One of the datasets resulted in zero embeddings. Check image paths and sampling.")
        # sys.exit(1) 

    print("Datasets and embeddings prepared.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print("Data preparation complete. Next: Classifier and Training Loop.") 

    # --- 3. Initialize Classifier, Loss, Optimizer ---
    classifier_cfg = cfg['classifier']
    training_cfg = cfg['training']

    if train_embeddings.nelement() > 0:
        embedding_dim = train_embeddings.shape[1]
    else: 
        clip_vision_config = AutoConfig.from_pretrained(clip_model_id).vision_config
        embedding_dim = clip_vision_config.hidden_size
        print(f"Warning: Train embeddings were empty. Using embedding_dim {embedding_dim} from CLIP config.")

    linear_classifier = LinearClassifier(
        embedding_dim=embedding_dim,
        hidden_dims=classifier_cfg.get('hidden_dims', []),
        num_classes=classifier_cfg['num_classes'],
        dropout_rate=classifier_cfg.get('dropout', 0.0)
    ).to(device)
    print(f"Linear Classifier initialized: {linear_classifier}")

    criterion = nn.CrossEntropyLoss()
    if training_cfg['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(linear_classifier.parameters(), lr=float(training_cfg['learning_rate']), weight_decay=float(training_cfg.get('weight_decay', 0.01)))
    else: 
        optimizer = optim.Adam(linear_classifier.parameters(), lr=float(training_cfg['learning_rate']))
    
    print(f"Optimizer: {training_cfg['optimizer']}, LR: {training_cfg['learning_rate']}")

    # --- 4. Training Loop ---
    print("Starting training...")
    best_val_metric = 0 if training_cfg['early_stopping']['mode'] == 'max' else float('inf')
    epochs_no_improve = 0
    best_model_path = run_output_dir / "best_linear_classifier.pth"
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    use_amp_training = training_cfg.get('mixed_precision', False)
    if use_amp_training and not torch.cuda.is_available():
        print("Warning: Mixed precision training requested but CUDA is not available. Disabling AMP.")
        use_amp_training = False

    for epoch in range(training_cfg['num_epochs']):
        train_loss = train_one_epoch(linear_classifier, train_loader, criterion, optimizer, device, use_amp=use_amp_training)
        # Pass global cfg to evaluate_model for report generation
        val_accuracy, val_loss, _, _, _ = evaluate_model(linear_classifier, val_loader, criterion, device, use_amp=use_amp_training, cfg_for_report=cfg) 
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{training_cfg['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        current_metric_val = val_accuracy 
        if training_cfg['early_stopping']['metric'] == 'val_loss':
            current_metric_val = val_loss
        
        improved = False
        if training_cfg['early_stopping']['mode'] == 'max':
            if current_metric_val > best_val_metric:
                best_val_metric = current_metric_val
                epochs_no_improve = 0
                improved = True
            else:
                epochs_no_improve += 1
        else: 
            if current_metric_val < best_val_metric:
                best_val_metric = current_metric_val
                epochs_no_improve = 0
                improved = True
            else:
                epochs_no_improve += 1
        
        if improved and training_cfg.get('save_best_model', True):
            torch.save(linear_classifier.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} (Epoch {epoch+1}, Val Metric: {best_val_metric:.4f})")

        if training_cfg['early_stopping']['enabled'] and epochs_no_improve >= training_cfg['early_stopping']['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    history_path = run_output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    # --- 5. Final Evaluation on Test Set ---
    print("Loading best model for final evaluation on test set...")
    if training_cfg.get('save_best_model', True) and best_model_path.exists():
        linear_classifier.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("Warning: No best model saved or found. Evaluating with the model from the last epoch.")

    # Pass global cfg to evaluate_model for report generation
    test_accuracy, test_loss, test_report, test_preds, test_true = evaluate_model(
        linear_classifier, test_loader, criterion, device, use_amp=use_amp_training, is_test_set=True, cfg_for_report=cfg
    )
    print(f"Test Set Performance: Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
    if test_report:
        print("Test Set Classification Report:")
        for class_name, metrics_dict in test_report.items(): # Changed 'metrics' to 'metrics_dict'
            if class_name not in ["accuracy", "macro avg", "weighted avg", "confusion_matrix"]:
                print(f"  Class: {class_name}")
                print(f"    Precision: {metrics_dict['precision']:.4f}, Recall: {metrics_dict['recall']:.4f}, F1-Score: {metrics_dict['f1-score']:.4f}")
        print(f"  Overall Accuracy (from report): {test_report['accuracy']:.4f}")
        if "confusion_matrix" in test_report:
            print(f"  Confusion Matrix: {np.array(test_report['confusion_matrix'])}")

    test_results_summary = {
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "classification_report": test_report,
        "config_file_content": cfg 
    }
    summary_path = run_output_dir / "test_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(test_results_summary, f, indent=4)
    print(f"Test evaluation summary saved to {summary_path}")

    print("CLIP Linear Probing script finished.") 