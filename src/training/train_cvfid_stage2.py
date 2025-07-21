#!/usr/bin/env python
"""Stage-2 fusion training for C-VFiD after individual experts are pretrained.

Assumes you have three checkpoints trained by ``train_cvfid_expert.py`` located
under Directory structure:
    ckpts/
      ├─ npr/best_expert.pt
      ├─ dncnn/best_expert.pt
      └─ noiseprint/best_expert.pt

The script will:
1. Build full RvfidModel(num_experts=3).
2. Load *each* expert's LoRA weights & FrequencyExpert projection from the
   corresponding checkpoint (matching by parameter names).
3. Freeze ViT base (visual) parameters.
4. Train Router, expert LoRA deltas, fusion and classifier heads end-to-end.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.r_vfid.model import RvfidModel
from src.data_processing.benchmark_dataset import BenchmarkImageDataset, build_default_transform
from src.utils.metrics import compute_all_metrics

# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------


def load_expert_weights(model: RvfidModel, ckpt_paths: list[Path]):
    """Load pretrained expert branch weights from Stage-1 checkpoints."""
    for idx, ckpt_path in enumerate(ckpt_paths):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        # Filter keys belonging to expert idx (MultiLoRA branch & FrequencyExpert)
        prefix_vit = f"clip_model.visual."  # visual MultiLoRA layers already aligned by index
        prefix_freq = f"experts.{idx}."  # projection etc.
        subset = {k: v for k, v in sd.items() if k.startswith(prefix_vit) or k.startswith(prefix_freq)}
        model.load_state_dict(subset, strict=False)
        print(f"[Stage2] Loaded expert-{idx} weights from {ckpt_path}")


# -----------------------------------------------------------------------------

def freeze_vit_base(model: RvfidModel):
    for n, p in model.clip_model.visual.named_parameters():
        if "experts" not in n:  # LoRA delta weights live inside MultiLoRA submodules
            p.requires_grad = False


# -----------------------------------------------------------------------------

def train_fusion(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb"))

    # Dataset
    transform = build_default_transform()
    allowed = ["car", "cat", "chair", "horse"]
    train_ds = BenchmarkImageDataset(args.train_dir, transform, allowed_classes=allowed)
    val_ds = BenchmarkImageDataset(args.val_dir, transform, allowed_classes=allowed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = RvfidModel(num_experts=3, gating_mode=args.gating_mode)

    load_expert_weights(model, [Path(args.ckpt_npr), Path(args.ckpt_dncnn), Path(args.ckpt_noiseprint)])
    freeze_vit_base(model)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))

    best_auc = 0.0
    best_acc = 0.0
    epochs_no_improve = 0
    history: Dict[str, Any] = {"train": [], "val": []}

    for epoch in range(args.epochs):
        model.train()
        y_true, y_pred, y_score = [], [], []
        epoch_loss = 0.0
        import time
        start_epoch = time.time()
        for step,(imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = model.compute_loss(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            probs = logits.softmax(dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.detach().cpu().tolist())
            if (step+1) % args.log_interval == 0:
                print(f"Epoch {epoch} Step {step+1}/{len(train_loader)}  Loss={loss.item():.4f}")

        train_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        train_metrics["loss"] = epoch_loss / len(train_loader.dataset)
        history["train"].append(train_metrics)

        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("AUC/train", train_metrics["auc"], epoch)

        # Validation
        model.eval()
        y_true, y_pred, y_score = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += model.compute_loss(logits, labels).item() * imgs.size(0)
                probs = logits.softmax(dim=1)[:, 1]
                preds = (probs >= 0.5).long()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_score.extend(probs.cpu().tolist())
        val_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        val_metrics["loss"] = val_loss / len(val_loader.dataset)
        history["val"].append(val_metrics)

        writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
        writer.flush()

        if val_metrics["accuracy"] > best_acc + 1e-4:
            best_acc = val_metrics["accuracy"]
            best_auc = val_metrics["auc"]
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "acc": best_acc, "auc": best_auc}, out_dir / "best_stage2.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"[EarlyStop] No val accuracy improvement for {args.patience} epochs. Stopping.")
            break
        dur=time.time()-start_epoch
        print(f"Epoch {epoch}: Train AUC={train_metrics['auc']:.4f}  Val AUC={val_metrics['auc']:.4f}  Time={dur:.1f}s")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[Done] Stage-2 fusion training complete. Best Val Acc = %.4f  AUC = %.4f" % (best_acc, best_auc))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("C-VFiD Stage-2 fusion training")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ckpt_npr", required=True)
    p.add_argument("--ckpt_dncnn", required=True)
    p.add_argument("--ckpt_noiseprint", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gating_mode", choices=["softmax", "sigmoid"], default="sigmoid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience based on val accuracy")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_fusion(args) 