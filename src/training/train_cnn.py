import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import numpy as np
import os
import random
from pathlib import Path
import argparse
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch.nn.functional as F # Needed for Grad-CAM
from tqdm import tqdm # Import tqdm
from torch.utils.tensorboard import SummaryWriter

# 假設的路徑，後續會從 src.data_processing 和 src.models 匯入
# from src.data_processing.custom_dataset import GenImageDataset
# from src.models.baseline_classifiers import ResNet50Classifier

# 為了讓此腳本可以獨立執行（或在 IDE 中方便測試），暫時使用相對路徑技巧
# 在實際整合到專案時，需要確保 Python PATH 或使用正確的模組匯入
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # Add project root to path

from src.data_processing.custom_dataset import GenImageDataset
from src.models.baseline_classifiers import ResNet50Classifier

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_optimizer(model, config):
    opt_config = config['training']['optimizer']
    lr = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    params_to_update = filter(lambda p: p.requires_grad, model.parameters())

    if opt_config.lower() == 'adam':
        return optim.Adam(params_to_update, lr=lr, weight_decay=weight_decay)
    elif opt_config.lower() == 'adamw':
        return optim.AdamW(params_to_update, lr=lr, weight_decay=weight_decay)
    elif opt_config.lower() == 'sgd':
        momentum = float(config['training'].get('momentum', 0.9)) # .get for optional param
        return optim.SGD(params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_config}")

def get_scheduler(optimizer, config):
    scheduler_config = config['training'].get('scheduler') # .get as scheduler is optional
    if not scheduler_config:
        return None

    if scheduler_config['type'].lower() == 'step':
        step_size = int(scheduler_config['step_size'])
        gamma = float(scheduler_config['gamma'])
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # Add other schedulers here if needed (e.g., ReduceLROnPlateau)
    else:
        print(f"Scheduler type {scheduler_config['type']} not implemented or scheduler not specified. No scheduler will be used.")
        return None

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config, writer, run_output_dir):
    num_epochs = config['training']['num_epochs']
    checkpoint_config = config['training'].get('checkpoint_dir')
    if checkpoint_config:
        checkpoint_dir = Path(checkpoint_config)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = run_output_dir / checkpoint_dir
    else:
        checkpoint_dir = run_output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = checkpoint_dir / 'best_model.pth'
    last_model_path = checkpoint_dir / 'last_model.pth'
    history_path = run_output_dir / 'training_history.json'
    plots_dir = run_output_dir / config['evaluation'].get('plot_dir', 'plots')
    plots_dir.mkdir(parents=True, exist_ok=True)

    early_stopping_config = config['training']['early_stopping']
    use_early_stopping = early_stopping_config['enabled']
    patience = early_stopping_config['patience']
    early_stopping_metric = early_stopping_config['metric']
    min_delta = early_stopping_config.get('min_delta', 0.0)

    best_metric_val = float('-inf') if early_stopping_metric == 'val_acc' else float('inf')
    epochs_no_improve = 0

    training_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    scheduler = get_scheduler(optimizer, config)
    console_log_freq = config['training'].get('console_log_frequency', 100) # For batch logging to console

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        
        # Training loop with tqdm
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Update tqdm postfix with current batch loss and running accuracy
            current_batch_loss = loss.item()
            current_running_acc = correct_train / total_train if total_train > 0 else 0
            train_pbar.set_postfix(loss=f'{current_batch_loss:.4f}', acc=f'{current_running_acc:.4f}')
            
            if writer and batch_idx % config['training'].get('log_frequency', 20) == 0:
                writer.add_scalar('Batch/TrainLoss', current_batch_loss, epoch * len(train_loader) + batch_idx)
            
            # Optional: More frequent console logging for batch metrics
            if console_log_freq > 0 and (batch_idx + 1) % console_log_freq == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}: Batch Loss: {current_batch_loss:.4f}, Running Train Acc: {current_running_acc:.4f}")

        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = correct_train / total_train if total_train > 0 else 0
        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        training_history["lr"].append(current_lr)
        if writer:
            writer.add_scalar('Epoch/TrainLoss', epoch_train_loss, epoch)
            writer.add_scalar('Epoch/TrainAcc', epoch_train_acc, epoch)
            writer.add_scalar('Epoch/LearningRate', current_lr, epoch)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        # Validation loop with tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_val_item = criterion(outputs, labels)
                val_loss += loss_val_item.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                current_val_running_acc = correct_val / total_val if total_val > 0 else 0
                val_pbar.set_postfix(acc=f'{current_val_running_acc:.4f}')

        epoch_val_loss = val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = correct_val / total_val if total_val > 0 else 0
        training_history["val_loss"].append(epoch_val_loss)
        training_history["val_acc"].append(epoch_val_acc)
        if writer:
            writer.add_scalar('Epoch/ValLoss', epoch_val_loss, epoch)
            writer.add_scalar('Epoch/ValAcc', epoch_val_acc, epoch)

        # Print epoch summary (tqdm handles progress, so this is a summary)
        print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f} | LR: {current_lr:.2e}")

        current_metric_val = epoch_val_acc if early_stopping_metric == 'val_acc' else -epoch_val_loss
        if current_metric_val > best_metric_val:
            best_metric_val = current_metric_val
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with {early_stopping_metric}: {abs(current_metric_val):.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in {early_stopping_metric} for {epochs_no_improve} epoch(s). Best: {abs(best_metric_val):.4f}")

        if use_early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {patience} epochs.")
            break
            
        if scheduler:
            scheduler.step() # For StepLR, CosineAnnealingLR etc.
            # For ReduceLROnPlateau, use: scheduler.step(epoch_val_loss_or_acc)

    # Save the last model
    torch.save(model.state_dict(), last_model_path)
    print(f"Last model saved to {last_model_path}")
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f)
    print(f"Training history saved to {history_path}")
    return str(best_model_path), str(last_model_path)

def evaluate_model(model, data_loader, criterion, device, config, eval_name="Test Set", writer=None, global_step=0, grad_cam_instance=None, output_dir_for_eval=None):
    # Access output_dir from the 'general' sub-config
    output_dir = Path(config['general']['output_dir'])
    plots_dir = output_dir / config['evaluation'].get('plot_dir', 'plots') / eval_name.lower().replace(" ", "_")
    plots_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(data_loader.dataset.class_to_idx.keys())

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return accuracy, avg_loss

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    general_config = config.get('general', {})
    model_config = config['model']
    data_config = config['data']
    training_config = config['training']
    eval_config = config['evaluation']
    ext_eval_config = config.get('external_evaluation', {'enabled': False})

    # Construct the main output directory: base_output_dir / model_name
    base_output_dir_from_config = Path(general_config.get('output_dir', 'results/cnn_runs'))
    model_name_from_config = model_config.get('name', 'default_model')
    # The actual output_dir for this run, to be passed around or stored in config for consistent access
    current_run_output_dir = base_output_dir_from_config / model_name_from_config
    current_run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"All outputs will be saved to: {current_run_output_dir}")
    
    writer = None
    if training_config.get('use_tensorboard', False) and SummaryWriter:
        # Tensorboard logs go into the model-specific directory
        log_dir = current_run_output_dir / 'tensorboard_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs will be saved to: {log_dir}")

    set_seed(general_config.get('seed', 42))
    device = torch.device(f"cuda:{general_config.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_data_path = data_config['base_data_dir']

    # Transformations - TODO: make this configurable via YAML if needed
    # For now, using the default from GenImageDataset
    img_transforms = GenImageDataset._get_default_transform()
    # if data_config.get('augmentation') == 'heavy': # Example for future extension
    #    img_transforms = ... custom heavy transforms ...

    # Datasets (as per user request: 10k train, 1k val, 1k test)
    train_dataset_params = {
        'root_dir': base_data_path, 'split': "train", 'transform': img_transforms,
        'num_samples_per_class': data_config.get('train_samples_per_class', 5000),
        'seed': general_config.get('seed', 42)
    }
    train_dataset = GenImageDataset(**train_dataset_params)
    print(f"Train dataset size: {len(train_dataset)}")

    val_dataset_params = {
        'root_dir': base_data_path, 'split': "val", 'transform': img_transforms,
        'num_samples_per_class': data_config.get('val_samples_per_class', 500),
        'seed': general_config.get('seed', 42)
    }
    val_dataset = GenImageDataset(**val_dataset_params)
    print(f"Validation dataset size: {len(val_dataset)}")

    val_image_paths = set(val_dataset.get_image_paths())
    test_dataset_params = {
        'root_dir': base_data_path, 'split': "test", 'transform': img_transforms,
        'num_samples_per_class': data_config.get('test_samples_per_class', 500),
        'exclude_files': val_image_paths, 'seed': general_config.get('seed', 42) + 1
    }
    test_dataset = GenImageDataset(**test_dataset_params)
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_image_paths = set(test_dataset.get_image_paths())
    overlap = val_image_paths.intersection(test_image_paths)
    if overlap:
        print(f"Warning: Overlap detected between validation and test sets: {len(overlap)} images.")
    else:
        print("No overlap between validation and test sets.")

    train_loader = DataLoader(train_dataset, batch_size=data_config['batch_size'], shuffle=True, 
                              num_workers=data_config['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=data_config['batch_size'], shuffle=False, 
                            num_workers=data_config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=data_config['batch_size'], shuffle=False, 
                             num_workers=data_config['num_workers'], pin_memory=True)

    # Model selection - simplistic for now, assuming ResNet from baseline_classifiers
    # TODO: Make model selection fully config-driven (e.g. model_config['type'] and model_config['name'])
    if model_config['type'].lower() == 'resnet':
        # Potentially map model_config['name'] to specific ResNet variants if more are added
        model = ResNet50Classifier(pretrained=model_config['pretrained'], 
                                   num_classes=model_config['num_classes'], 
                                   freeze_backbone=model_config['freeze_backbone'])
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)
    
    # Initialize grad_cam_instance to None *before* any conditional assignment
    grad_cam_instance = None 
    if eval_config.get('generate_gradcam', False):
        try:
            target_layer_name = model_config.get('gradcam_target_layer', eval_config.get('gradcam_layer_name', 'backbone.layer4'))
            grad_cam_instance = GradCAM(model, target_layer_name)
            print(f"GradCAM initialized for layer: {target_layer_name}")
        except AttributeError as e: # More specific exception for layer not found
            print(f"Could not initialize GradCAM (AttributeError, likely layer name issue '{target_layer_name}'): {e}. GradCAM will be disabled.")
        except Exception as e: # Catch other potential errors during GradCAM init
            print(f"Could not initialize GradCAM (Exception: '{type(e).__name__}'): {e}. GradCAM will be disabled.")

    print("\nStarting training...")
    best_model_path, last_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, device, config, writer, current_run_output_dir)
    print(f"Training finished. Best model: {best_model_path}, Last model: {last_model_path}")

    print("\nLoading best model for evaluation on Test set...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, test_loader, criterion, device, config, eval_name="Test Set (Best Model)", writer=writer, global_step=training_config['num_epochs'], grad_cam_instance=grad_cam_instance, output_dir_for_eval=current_run_output_dir)
    
    print("\nLoading last model for evaluation on Test set...")
    model.load_state_dict(torch.load(last_model_path))
    evaluate_model(model, test_loader, criterion, device, config, eval_name="Test Set (Last Model)", writer=writer, global_step=training_config['num_epochs']+1, grad_cam_instance=grad_cam_instance, output_dir_for_eval=current_run_output_dir)

    # External Evaluation
    if ext_eval_config.get('enabled', False):
        try:
            print(f"\nEvaluating BEST model on {ext_eval_config.get('name')}...")
            model.load_state_dict(torch.load(best_model_path))
            # For external eval, plots should go into a subfolder of the main run's output_dir
            ext_eval_plot_output_dir = current_run_output_dir / "external_evaluations" / ext_eval_config.get('name', 'external_set')
            evaluate_model(model, ext_loader, criterion, device, config, 
                           eval_name=f"{ext_eval_config.get('name')} (Best Model)", grad_cam_instance=grad_cam_instance, output_dir_for_eval=ext_eval_plot_output_dir)
        except Exception as e:
            print(f"Error in external evaluation: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN-based image classifier using a YAML config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file (e.g., configs/cnn_baseline.yaml)')
    args = parser.parse_args()
    main(args.config) 