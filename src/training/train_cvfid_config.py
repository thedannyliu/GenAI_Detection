#!/usr/bin/env python
"""Advanced training script for C-VFiD on AIGCDetectionBenchmark.

• Loads hyper-params from a YAML config file (see configs/train_cvfid_example.yaml).
• Supports single-/multi-GPU training via torch.nn.DataParallel.
• Can resume from checkpoint.
• Logs richer metrics (acc, auc, ap, precision, recall, f1) and writes
  TensorBoard summaries + JSON history.
"""

from __future__ import annotations

import argparse, os, json, time, yaml, random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.models.r_vfid.model import RvfidModel
from src.data_processing.benchmark_dataset import BenchmarkImageDataset, build_default_transform
from src.utils.metrics import (
    compute_all_metrics,
    compute_pr_metrics,
    compute_roc_metrics,
    compute_calibration_metrics,
)


# -----------------------------------------------------------------------------
#   Helpers
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# -----------------------------------------------------------------------------
#   Training & Evaluation
# -----------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = logits.softmax(dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.cpu().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_score_np = np.array(y_score)

    metrics = compute_all_metrics(y_true_np, y_pred_np, y_score_np)
    # Extra curves
    metrics["roc"] = compute_roc_metrics(y_true, y_score)
    metrics["pr"] = compute_pr_metrics(y_true, y_score)
    metrics["calibration"] = compute_calibration_metrics(y_true, y_score)
    return metrics


# -----------------------------------------------------------------------------

def train(cfg: Dict[str, Any]):
    seed_everything(cfg.get("seed", 42))

    device_ids = [int(x) for x in cfg.get("gpus", "0").split(",") if x != ""]
    primary_device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    # 1. Datasets
    transform = build_default_transform()
    allowed = cfg.get("allowed_classes")
    train_ds = BenchmarkImageDataset(cfg["train_dir"], transform, allowed_classes=allowed)
    val_ds = BenchmarkImageDataset(cfg["val_dir"], transform, allowed_classes=allowed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model
    model = RvfidModel(
        num_experts=cfg["num_experts"],
        gating_mode=cfg.get("gating_mode", "sigmoid"),
    )

    if len(device_ids) > 1 and torch.cuda.device_count() >= len(device_ids):
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(primary_device)

    # 3. Optimiser
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    start_epoch = 0
    best_auc = 0.0
    history: Dict[str, Any] = {"train": [], "val": []}

    # Resume if requested
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"[Resume] Loaded checkpoint from {cfg['resume']} (epoch={start_epoch})")

    # 4. Training loop
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        y_true, y_pred, y_score = [], [], []
        grad_accum = cfg.get("gradient_accumulation_steps", 1)
        for step, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(primary_device), labels.to(primary_device)
            if step % grad_accum == 0:
                optimizer.zero_grad()
            logits = model(imgs)
            loss = model.module.compute_loss(logits, labels) if isinstance(model, nn.DataParallel) else model.compute_loss(logits, labels)
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            probs = logits.softmax(dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.detach().cpu().tolist())

        train_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        train_metrics["loss"] = epoch_loss / len(train_loader.dataset)
        history["train"].append(train_metrics)

        # TensorBoard log
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("AUC/train", train_metrics["auc"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)

        # -------------- Validation --------------
        val_metrics = evaluate(model, val_loader, primary_device)
        history["val"].append(val_metrics)
        writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.flush()

        print(f"Epoch {epoch:02d}: Train AUC={train_metrics['auc']:.4f}  Val AUC={val_metrics['auc']:.4f}")

        # Checkpoint
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            ckpt_path = out_dir / "best_cvfid.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                },
                ckpt_path,
            )

    # Write history JSON
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("[Done] Training completed. Best Val AUC = %.4f" % best_auc)


# -----------------------------------------------------------------------------
#   CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("C-VFiD training w/ YAML config")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg) 