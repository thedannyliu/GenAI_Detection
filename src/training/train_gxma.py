import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.data_processing.custom_dataset import GenImageDataset
from src.models.gxma.gxma_fusion_detector import GXMAFusionDetector
from src.experiments.zero_shot_vlm_eval import vlm_collate_fn
import yaml


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
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(
    model: GXMAFusionDetector,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for images, labels in loader:
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def main(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    general = cfg.get("general", {})
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

    dataset_root = data_cfg["base_data_dir"]
    class_map = data_cfg.get("class_to_idx", {"nature": 0, "ai": 1})
    num_train = data_cfg.get("train_samples_per_class")
    num_val = data_cfg.get("val_samples_per_class")
    num_test = data_cfg.get("test_samples_per_class")

    train_dataset = GenImageDataset(
        dataset_root,
        split="train",
        transform=None,
        class_to_idx=class_map,
        num_samples_per_class=num_train,
        seed=seed,
    )
    val_dataset = GenImageDataset(
        dataset_root,
        split="val",
        transform=None,
        class_to_idx=class_map,
        num_samples_per_class=num_val,
        seed=seed + 1,
    )
    test_dataset = GenImageDataset(
        dataset_root,
        split="val",
        transform=None,
        class_to_idx=class_map,
        num_samples_per_class=num_test,
        seed=seed + 2,
    )

    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 4)

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

    model = GXMAFusionDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    num_epochs = train_cfg.get("num_epochs", 10)
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path)

    # Final evaluation on the test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    results = {
        "history": history,
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
    args = parser.parse_args()
    main(args.config)
