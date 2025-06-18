import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
import time
from tqdm.auto import tqdm

from src.data_processing.custom_dataset import GenImageDataset
from src.models.gxma.gxma_fusion_detector import GXMAFusionDetector
from src.models.gxma.ablation_detectors import FrequencyOnlyDetector, SemanticOnlyDetector
from src.experiments.zero_shot_vlm_eval import vlm_collate_fn
import yaml


def get_cosine_schedule_with_warmup(
    optimizer: optim.Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function after
    a warmup period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: GXMAFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    max_steps: int,
    current_step: int,
    epoch: int,
) -> Tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)):
        if current_step >= max_steps:
            break

        # Assuming `images` is a list of Tensors, stack them and move to device
        if isinstance(images, list):
            images = torch.stack(images).to(device)
        else:
            images = images.to(device)

        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        current_step += 1

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc, current_step


def evaluate(
    model: GXMAFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Eval {epoch}", leave=False):
            # Assuming `images` is a list of Tensors, stack them and move to device
            if isinstance(images, list):
                images = torch.stack(images).to(device)
            else:
                images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


def main(config_path: str, mode: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    general = cfg.get("general", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    seed = general.get("seed", 42)
    set_seed(seed)

    device = torch.device(
        f"cuda:{general.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(general.get("output_dir", "results/gxma_runs")) / general.get(
        "experiment_name", "gxma_fusion_poc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define a basic transform to convert images to tensors
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # Potentially add normalization if needed by the model, but ToTensor already scales to [0, 1]
    ])

    dataset_root = data_cfg["base_data_dir"]

    # 允許在 YAML 中自訂資料集 split 子資料夾名稱，預設為原本的 train/val
    train_split_name = data_cfg.get("train_split_name", "train")
    val_split_name = data_cfg.get("val_split_name", "val")
    test_split_name = data_cfg.get("test_split_name", val_split_name)  # 預設與 val 相同

    class_map = data_cfg.get("class_to_idx", {"nature": 0, "ai": 1})
    num_train = data_cfg.get("train_samples_per_class")
    num_val = data_cfg.get("val_samples_per_class")
    num_test = data_cfg.get("test_samples_per_class")

    train_dataset = GenImageDataset(
        dataset_root,
        split=train_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_train,
        seed=seed,
    )
    val_dataset = GenImageDataset(
        dataset_root,
        split=val_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_val,
        seed=seed + 1,
    )
    test_dataset = GenImageDataset(
        dataset_root,
        split=test_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_test,
        seed=seed + 2,
    )

    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)

    # Dataset statistics
    print("Loaded datasets:")
    print(f"  Train set: {len(train_dataset)} images | batches: {len(train_dataset)//batch_size}")
    print(f"  Val   set: {len(val_dataset)} images | batches: {len(val_dataset)//batch_size}")
    print(f"  Test  set: {len(test_dataset)} images | batches: {len(test_dataset)//batch_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg.get("batch_size", batch_size),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vlm_collate_fn,
    )

    if mode == "fusion":
        model = GXMAFusionDetector(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_classes=model_cfg.get("num_classes", 2),
            fusion_strategy=model_cfg.get("fusion_strategy", "concat"),
        ).to(device)
    elif mode == "frequency":
        model = FrequencyOnlyDetector(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_classes=model_cfg.get("num_classes", 2),
        ).to(device)
    elif mode == "semantic":
        model = SemanticOnlyDetector(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_classes=model_cfg.get("num_classes", 2),
        ).to(device)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    num_epochs = train_cfg.get("num_epochs", 10)
    num_training_steps = num_epochs * len(train_loader)
    scheduler_cfg = train_cfg.get("scheduler", {})
    
    if scheduler_cfg.get("name") == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_cfg.get("warmup_steps", 0),
            num_training_steps=num_training_steps,
        )
    else:
        # Default to a scheduler that does nothing
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1.0)

    early_stopping_cfg = train_cfg.get("early_stopping", {})
    es_patience = early_stopping_cfg.get("patience", 10)
    es_threshold = early_stopping_cfg.get("threshold", 0.0)
    es_monitor = early_stopping_cfg.get("monitor", "val_acc")

    best_metric = 0.0 if es_monitor == "val_acc" else float('inf')
    epochs_no_improve = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    current_step = 0
    training_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")
        epoch_start_time = time.time()
        print("[Stage] Training loop -> Epoch", epoch)
        train_loss, train_acc, current_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            num_training_steps,
            current_step,
            epoch,
        )
        print("[Stage] Validation loop")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Timing information
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - training_start_time
        remaining_epochs = max(num_epochs - epoch, 0)
        eta_seconds = epoch_time * remaining_epochs
        def _format_time(sec: float) -> str:
            mins, sec = divmod(int(sec), 60)
            hrs, mins = divmod(mins, 60)
            return f"{hrs:02d}:{mins:02d}:{sec:02d}"

        print(
            f"Epoch time: {_format_time(epoch_time)} | "
            f"Elapsed: {_format_time(elapsed_time)} | "
            f"ETA: {_format_time(eta_seconds)}"
        )

        metric_to_check = val_acc if es_monitor == "val_acc" else val_loss
        
        improvement = metric_to_check - best_metric if es_monitor == "val_acc" else best_metric - metric_to_check

        if improvement > es_threshold:
            best_metric = metric_to_check
            epochs_no_improve = 0
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation metric improved to {best_metric:.4f}. Saving model.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= es_patience:
            print(f"Early stopping triggered after {es_patience} epochs with no improvement.")
            break
        
        if current_step >= num_training_steps:
            print("Completed all training steps.")
            break

    # Final evaluation on the test set using the best model
    print("[Stage] Final Test evaluation")
    print("Loading best model for final evaluation...")
    checkpoint_path = output_dir / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Test")
        print(f"Test Accuracy on best model: {test_acc:.4f}")
    else:
        print("No best model found. Evaluating model from last epoch.")
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, "Test")
        print(f"Test Accuracy on last model: {test_acc:.4f}")


    results = {
        "history": history,
        "best_val_metric": best_metric,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "config": cfg,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GXMA Fusion Detector")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "frequency", "semantic"],
        help="Which model variant to train (fusion | frequency | semantic)",
    )
    args = parser.parse_args()
    main(args.config, args.mode)
