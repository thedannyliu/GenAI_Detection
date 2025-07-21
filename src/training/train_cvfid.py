#!/usr/bin/env python
"""Training script for C-VFiD (RvfidModel) on AIGC Detection Benchmark.

Example usage:
    python -m src.training.train_cvfid \
        --train_dir AIGCDetectionBenchMark/progan_train \
        --val_dir   AIGCDetectionBenchMark/progan_val \
        --output_dir results/cvfid_run1 \
        --batch_size 16 --epochs 5 --gating_mode sigmoid
"""
from __future__ import annotations

import argparse, os, json, random, time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.r_vfid.model import RvfidModel
from src.data_processing.benchmark_dataset import BenchmarkImageDataset, build_default_transform
from src.utils.metrics import compute_all_metrics


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("C-VFiD training")
    p.add_argument("--train_dir", required=True, help="Path to training image folder")
    p.add_argument("--val_dir", required=True, help="Path to validation image folder")
    p.add_argument("--output_dir", default="results/cvfid", help="Directory to save checkpoints & logs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gating_mode", choices=["softmax", "sigmoid"], default="sigmoid")
    p.add_argument("--num_experts", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset & DataLoader
    transform = build_default_transform()
    train_ds = BenchmarkImageDataset(args.train_dir, transform)
    val_ds = BenchmarkImageDataset(args.val_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Model
    model = RvfidModel(num_experts=args.num_experts, gating_mode=args.gating_mode).to(device)

    # 3. Optimiser & criterion
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    history: Dict[str, Any] = {"train": [], "val": []}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        y_true, y_pred, y_score = [], [], []
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy())

        train_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        train_metrics["loss"] = epoch_loss / len(train_loader)
        history["train"].append(train_metrics)

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            y_true, y_pred, y_score = [], [], []
            val_loss = 0.0
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_score.extend(logits.softmax(dim=1)[:, 1].cpu().numpy())
        val_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        val_metrics["loss"] = val_loss / len(val_loader)
        history["val"].append(val_metrics)

        print(f"Epoch {epoch}: Train AUC={train_metrics['auc']:.4f}  Val AUC={val_metrics['auc']:.4f}")

        # Save best checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_path = Path(args.output_dir) / "best_cvfid.pt"
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "auc": best_auc}, ckpt_path)

    # Save history
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main() 