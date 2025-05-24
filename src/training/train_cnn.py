import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os
import random
from pathlib import Path
import argparse
import json
import yaml

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

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    num_epochs = config['training']['num_epochs']
    output_dir = Path(config['output_dir'])
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', output_dir / 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = checkpoint_dir / 'best_model.pth'
    last_model_path = checkpoint_dir / 'last_model.pth'
    history_path = output_dir / 'training_history.json'

    best_val_metric = float('-inf') # Assuming higher is better for early stopping metric (e.g. accuracy)
    # Or float('inf') if monitoring loss and lower is better. For now, use accuracy.
    epochs_no_improve = 0
    early_stopping_config = config['training'].get('early_stopping', {})
    use_early_stopping = early_stopping_config.get('enabled', True) # Default to true if key exists
    patience = early_stopping_config.get('patience', 5)
    early_stopping_metric = early_stopping_config.get('metric', 'val_acc') # e.g. val_acc or val_loss

    training_history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }
    
    scheduler = get_scheduler(optimizer, config)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
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

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        training_history["train_loss"].append(epoch_train_loss)
        training_history["train_acc"].append(epoch_train_acc)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss_val_item = criterion(outputs, labels)
                val_loss += loss_val_item.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        training_history["val_loss"].append(epoch_val_loss)
        training_history["val_acc"].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

        current_metric_val = epoch_val_acc # Default to accuracy for now
        if early_stopping_metric == 'val_loss':
            current_metric_val = -epoch_val_loss # Negative because lower loss is better

        if current_metric_val > best_val_metric:
            best_val_metric = current_metric_val
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} with {early_stopping_metric}: {abs(current_metric_val):.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in {early_stopping_metric} for {epochs_no_improve} epoch(s). Best: {abs(best_val_metric):.4f}")

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

def evaluate_model(model, data_loader, criterion, device, class_names=['nature', 'ai']):
    """Evaluate the model on given data loader."""
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

    set_seed(general_config.get('seed', 42))
    device = torch.device(f"cuda:{general_config.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_data_path = data_config['base_data_dir']
    output_dir = Path(general_config.get('output_dir', 'results/cnn_run'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Transformations - TODO: make this configurable via YAML if needed
    # For now, using the default from GenImageDataset
    img_transforms = GenImageDataset._get_default_transform()
    # if data_config.get('augmentation') == 'heavy': # Example for future extension
    #    img_transforms = ... custom heavy transforms ...

    # Datasets (as per user request: 10k train, 1k val, 1k test)
    train_dataset = GenImageDataset(
        root_dir=base_data_path,
        split="train",
        transform=img_transforms,
        num_samples_per_class=data_config.get('train_samples_per_class', 5000),
        seed=general_config.get('seed', 42)
    )
    print(f"Train dataset size: {len(train_dataset)}")

    val_dataset = GenImageDataset(
        root_dir=base_data_path,
        split="val",
        transform=img_transforms,
        num_samples_per_class=data_config.get('val_samples_per_class', 500),
        seed=general_config.get('seed', 42)
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    val_image_paths = set(val_dataset.get_image_paths())
    test_dataset = GenImageDataset(
        root_dir=base_data_path,
        split="test", 
        transform=img_transforms,
        num_samples_per_class=data_config.get('test_samples_per_class', 500),
        exclude_files=val_image_paths,
        seed=general_config.get('seed', 42) + 1
    )
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
    
    print("\nStarting training...")
    best_model_path, last_model_path = train_model(model, train_loader, val_loader, criterion, optimizer, device, config)
    print(f"Training finished. Best model: {best_model_path}, Last model: {last_model_path}")

    print("\nLoading best model for final evaluation on Test set...")
    model.load_state_dict(torch.load(best_model_path))
    evaluate_model(model, test_loader, criterion, device)
    
    print("\nEvaluating last model on Test set...")
    model.load_state_dict(torch.load(last_model_path))
    evaluate_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN-based image classifier using a YAML config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file (e.g., configs/cnn_baseline.yaml)')
    args = parser.parse_args()
    main(args.config) 