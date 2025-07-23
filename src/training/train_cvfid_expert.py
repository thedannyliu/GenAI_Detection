#!/usr/bin/env python
"""Stage-1 training for a single FrequencyExpert branch.

This script trains *only one* expert (RGB, NPR, DnCNN, or NoisePrint) as a
binary classifier on ProGAN training data.  The ViT backbone weights remain
frozen; only the corresponding LoRA branch + classification head are updated.

Run example:
    python -m src.training.train_cvfid_expert \
        --train_dir AIGCDetectionBenchMark/progan_train \
        --val_dir   AIGCDetectionBenchMark/progan_val \
        --output_dir results/expert_npr \
        --expert_mode npr
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
from typing import Any, Dict

import numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from src.utils.ddp_utils import init_ddp, is_main_process

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

def freeze_except_expert(model: nn.Module, expert_idx: int):
    """Freeze all parameters except the specified expert LoRA + classifier.

    Accepts either ``RvfidModel`` or ``nn.DataParallel`` wrapper.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_ref = model.module  # unwrap
    else:
        model_ref = model

    for _, p in model_ref.named_parameters():
        p.requires_grad = False  # freeze all

    from src.models.r_vfid.multi_lora import MultiLoRALinear

    # Unfreeze selected LoRA branch
    for m in model_ref.clip_model.visual.modules():
        if isinstance(m, MultiLoRALinear):
            for i, exp in enumerate(m.experts):
                for p in exp.parameters():
                    p.requires_grad = (i == expert_idx)

    # Unfreeze FrequencyExpert projection of this branch
    for p in model_ref.experts[expert_idx].parameters():
        p.requires_grad = True

    # Unfreeze classifier head
    for p in model_ref.classifier.parameters():
        p.requires_grad = True


# -----------------------------------------------------------------------------

def train_one_expert(args):
    seed_everything(args.seed)
    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if is_main_process() else None

    # Dataset
    transform = build_default_transform()
    allowed = args.allowed_classes.split(",") if args.allowed_classes else None
    train_ds = BenchmarkImageDataset(args.train_dir, transform, allowed_classes=allowed)
    val_ds = BenchmarkImageDataset(args.val_dir, transform, allowed_classes=allowed)

    if is_main_process():
        print(f"[Info] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # Model w/ single expert
    model = RvfidModel(num_experts=1, gating_mode="softmax").to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    # Set expert mode inside underlying model
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.experts[0].mode = args.expert_mode
    else:
        model.experts[0].mode = args.expert_mode

    freeze_except_expert(model, 0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

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
        grad_accum = args.gradient_accumulation_steps
        from tqdm import tqdm
        train_sampler.set_epoch(epoch)
        scaler=torch.cuda.amp.GradScaler()
        for step,(imgs, labels) in enumerate(tqdm(train_loader, desc=f"Rank{local_rank} Epoch {epoch}", disable=not is_main_process())):
            imgs, labels = imgs.to(device), labels.to(device)
            if step % grad_accum == 0:
                optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = nn.functional.binary_cross_entropy_with_logits(logits[:, 1], labels.float())
            scaler.scale(loss/grad_accum).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer); scaler.update()

            epoch_loss += loss.item() * imgs.size(0)
            probs = logits.softmax(dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_score.extend(probs.detach().cpu().tolist())
            if (step+1) % args.log_interval == 0 and is_main_process():
                print(f"Epoch {epoch} Step {step+1}/{len(train_loader)}  Loss={loss.item():.4f}")

        train_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        train_metrics["loss"] = epoch_loss / len(train_loader.dataset)
        history["train"].append(train_metrics)

        if writer:
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("AUC/train", train_metrics["auc"], epoch)

        # ---- validation ----
        model.eval()
        y_true, y_pred, y_score = [], [], []
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += nn.functional.binary_cross_entropy_with_logits(logits[:, 1], labels.float()).item() * imgs.size(0)
                probs = logits.softmax(dim=1)[:, 1]
                preds = (probs >= 0.5).long()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                y_score.extend(probs.cpu().tolist())

        val_metrics = compute_all_metrics(np.array(y_true), np.array(y_pred), np.array(y_score))
        val_metrics["loss"] = val_loss / len(val_loader.dataset)
        history["val"].append(val_metrics)

        if writer:
            writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
            writer.flush()

        if val_metrics["accuracy"] > best_acc + 1e-4 and is_main_process():
            best_acc = val_metrics["accuracy"]
            best_auc = val_metrics["auc"]
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "acc": best_acc, "auc": best_auc}, out_dir / "best_expert.pt")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if is_main_process() and epochs_no_improve >= args.patience:
            print(f"[EarlyStop] No val accuracy improvement for {args.patience} epochs. Stopping.")
            break
        if is_main_process():
            dur = time.time()-start_epoch
            print(f"Epoch {epoch}: Train AUC={train_metrics['auc']:.4f}  Val AUC={val_metrics['auc']:.4f}  Time={dur:.1f}s")

    if is_main_process():
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print("[Done] Expert training complete. Best Val Acc = %.4f  AUC = %.4f" % (best_acc, best_auc))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("C-VFiD Stage-1 single expert training")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--expert_mode", choices=["npr", "dncnn", "noiseprint"], required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--allowed_classes", default="car,cat,chair,horse")
    p.add_argument("--patience", type=int, default=3, help="Early stopping patience based on val accuracy")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # local_rank comes from torchrun; do not set default manually
    p.add_argument("--local_rank", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_one_expert(args) 