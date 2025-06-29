import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
import time
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import shutil

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

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)):
        if len(batch) == 3:
            images, freq_feats, labels = batch
        else:
            images, labels = batch  # type: ignore
            freq_feats = None
        if current_step >= max_steps:
            break

        # Move to device
        if isinstance(images, list):
            images = torch.stack(images).to(device)
        else:
            images = images.to(device)
        if freq_feats is not None:
            freq_feats = freq_feats.to(device)

        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, freq_feats) if freq_feats is not None else model(images)
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
) -> Tuple[float, float, float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []  # For AUC

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Eval {epoch}", leave=False):
            if len(batch) == 3:
                images, freq_feats, labels = batch
            else:
                images, labels = batch  # type: ignore
                freq_feats = None
            # Assuming `images` is a list of Tensors, stack them and move to device
            if isinstance(images, list):
                images = torch.stack(images).to(device)
            else:
                images = images.to(device)
            if freq_feats is not None:
                freq_feats = freq_feats.to(device)
            labels = labels.to(device)
            outputs = model(images, freq_feats) if freq_feats is not None else model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    # Calculate AUC & F1 safely
    try:
        auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    except Exception:
        auc = 0.0
    f1 = f1_score(all_labels, all_preds, average="binary") if all_labels else 0.0
    return avg_loss, acc, auc, f1, torch.tensor(all_preds), torch.tensor(all_labels)


# === Utility: checkpoint I/O ===

def _save_checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    best_metric: float,
    current_step: int,
    epochs_no_improve: int,
    filename: Path,
) -> None:
    """Persist full training state (model + optimizer + scheduler)."""
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
        "current_step": current_step,
        "epochs_no_improve": epochs_no_improve,
    }
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, filename)


def main(config_path: Optional[str], mode: str, resume_path: Optional[str] = None) -> None:
    # ---------------------------------------------------------------
    # 0. Resolve which config YAML to load
    #    • If --resume is passed and <run_dir>/config_used.yaml exists, use it
    #    • Otherwise fall back to --config provided via CLI
    # ---------------------------------------------------------------
    cfg_path_final: Optional[Path] = Path(config_path) if config_path else None
    if resume_path:
        resume_file = Path(resume_path)
        candidate_cfg = resume_file.parent / "config_used.yaml"
        if candidate_cfg.exists():
            print(f"[Config] Using config file from previous run: {candidate_cfg}")
            cfg_path_final = candidate_cfg
        elif cfg_path_final is None:
            raise FileNotFoundError(
                "--resume specified but config_used.yaml not found, and --config not provided."
            )

    if cfg_path_final is None or not cfg_path_final.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path_final}")

    with open(cfg_path_final, "r") as f:
        cfg = yaml.safe_load(f)

    general = cfg.get("general", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})

    freq_methods = model_cfg.get("freq_methods", ["radial", "dct", "wavelet"])

    seed = general.get("seed", 42)
    set_seed(seed)

    device = torch.device(
        f"cuda:{general.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(general.get("output_dir", "results/gxma_runs")) / general.get(
        "experiment_name", "gxma_fusion_poc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config YAML for reproducibility
    try:
        shutil.copy(cfg_path_final, output_dir / "config_used.yaml")
    except Exception as e_copy:
        print(f"Warning: Failed to copy config file to output dir: {e_copy}")

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

    # Prepare datasets with pre-computed frequency features
    train_dataset = GenImageDataset(
        dataset_root,
        split=train_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_train,
        seed=seed,
        include_freq_features=True,
        freq_methods=freq_methods,
    )
    val_dataset = GenImageDataset(
        dataset_root,
        split=val_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_val,
        seed=seed + 1,
        include_freq_features=True,
        freq_methods=freq_methods,
    )
    test_dataset = GenImageDataset(
        dataset_root,
        split=test_split_name,
        transform=transform,
        class_to_idx=class_map,
        num_samples_per_class=num_test,
        seed=seed + 2,
        include_freq_features=True,
        freq_methods=freq_methods,
    )

    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 8)  # Increased workers

    def collate_with_freq(batch):
        # Each item: (image, freq_feat, label)
        images, freqs, labels = zip(*batch)
        images = torch.stack(images)
        freqs = torch.stack(freqs)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, freqs, labels

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
        collate_fn=collate_with_freq,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_freq,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg.get("batch_size", batch_size),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_freq,
        pin_memory=True,
        persistent_workers=True,
    )

    if mode == "fusion":
        model = GXMAFusionDetector(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_heads=model_cfg.get("num_heads", 4),
            num_classes=model_cfg.get("num_classes", 2),
            freq_methods=freq_methods,
            model_cfg=model_cfg,
        ).to(device)
    elif mode == "frequency":
        model = FrequencyOnlyDetector(
            hidden_dim=model_cfg.get("hidden_dim", 256),
            num_classes=model_cfg.get("num_classes", 2),
            freq_methods=freq_methods,
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
    if es_monitor in ["val_acc", "val_auc", "val_f1"]:
        best_metric = 0.0
    else:
        best_metric = float('inf')
    epochs_no_improve = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_auc": [],
        "val_f1": [],
        "lr": [],
    }

    # If resuming, try to load previous history from training_results.json
    json_prev = None
    if resume_path:
        json_candidate = (Path(resume_path).parent / "training_results.json")
        if json_candidate.exists():
            try:
                with open(json_candidate, "r") as f_prev:
                    prev_data = json.load(f_prev)
                    prev_hist = prev_data.get("history", {})
                    # Merge metric lists if lengths match epoch numbers before resume
                    for k in history.keys():
                        if k in prev_hist and isinstance(prev_hist[k], list):
                            history[k] = prev_hist[k].copy()
                print(f"[Resume] Loaded previous history with {len(history['train_loss'])} epochs.")
            except Exception as e_hist:
                print(f"Warning: could not load previous history: {e_hist}")

    current_step = 0
    training_start_time = time.time()
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    # ---------------- Resume from checkpoint (if provided) ----------------
    start_epoch: int = 1  # default begin at epoch 1
    epochs_no_improve = 0  # will be updated if resume
    current_step = 0  # for schedulers using total steps

    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.exists():
            print(f"[Resume] Loading checkpoint from {resume_file} ...")
            ckpt = torch.load(resume_file, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e_sched:
                print(f"Warning: failed to load scheduler state: {e_sched}")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", best_metric))
            current_step = int(ckpt.get("current_step", 0))
            epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
            print(f"[Resume] Resuming from epoch {start_epoch} with best_metric={best_metric:.4f}")
        else:
            print(f"[Resume] Checkpoint {resume_file} not found. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs + 1):
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
        val_loss, val_acc, val_auc, val_f1, val_preds_tensor, val_labels_tensor = evaluate(model, val_loader, criterion, device, epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}, "
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
        if es_monitor == "val_auc":
            metric_to_check = val_auc
        elif es_monitor == "val_f1":
            metric_to_check = val_f1
        
        if es_monitor in ["val_acc", "val_auc", "val_f1"]:
            improvement = metric_to_check - best_metric
        else:
            improvement = best_metric - metric_to_check

        if improvement > es_threshold:
            best_metric = metric_to_check
            epochs_no_improve = 0
            checkpoint_path_best = output_dir / "best_model.pth"
            torch.save(model.state_dict(), checkpoint_path_best)
            print(f"Validation metric improved to {best_metric:.4f}. Saving BEST model.")
        else:
            epochs_no_improve += 1

        # ---- Save last checkpoint every epoch ----
        _save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_metric=best_metric,
            current_step=current_step,
            epochs_no_improve=epochs_no_improve,
            filename=output_dir / "last.pth",
        )

        if epochs_no_improve >= es_patience:
            print(f"Early stopping triggered after {es_patience} epochs with no improvement.")
            break
        
        if current_step >= num_training_steps:
            print("Completed all training steps.")
            break

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # Final evaluation on the test set using the best model
    print("[Stage] Final Test evaluation")
    print("Loading best model for final evaluation...")
    checkpoint_path = output_dir / "best_model.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        test_loss, test_acc, test_auc, test_f1, test_preds_tensor, test_labels_tensor = evaluate(model, test_loader, criterion, device, 0)
        print(f"Test Accuracy on best model: {test_acc:.4f}")
        print(f"Test AUC on best model: {test_auc:.4f}, Test F1: {test_f1:.4f}")
    else:
        print("No best model found. Evaluating model from last epoch.")
        test_loss, test_acc, test_auc, test_f1, test_preds_tensor, test_labels_tensor = evaluate(model, test_loader, criterion, device, 0)
        print(f"Test Accuracy on last model: {test_acc:.4f}")
        print(f"Test AUC on last model: {test_auc:.4f}, Test F1: {test_f1:.4f}")

    results = {
        "history": history,
        "best_val_metric": best_metric,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_auc": test_auc,
        "test_f1": test_f1,
        "confusion_matrix_test": confusion_matrix(test_labels_tensor.cpu().numpy(), test_preds_tensor.cpu().numpy()).tolist(),
        "config": cfg,
    }
    # ---- Extra test datasets evaluation ----
    extra_test_results = {}
    extra_tests_cfg = eval_cfg.get("extra_tests", [])
    for extra_cfg in extra_tests_cfg:
        ds_name = extra_cfg.get("name", "extra")
        ds_root = extra_cfg["base_data_dir"]
        split_name = extra_cfg.get("split_name", "")
        class_map_extra = extra_cfg.get("class_to_idx", class_map)
        num_samples_extra = extra_cfg.get("num_samples_per_class")

        extra_dataset = GenImageDataset(
            ds_root,
            split=split_name,
            transform=transform,
            class_to_idx=class_map_extra,
            num_samples_per_class=num_samples_extra,
            seed=seed + 3,
        )
        extra_loader = DataLoader(
            extra_dataset,
            batch_size=eval_cfg.get("batch_size", batch_size),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_with_freq,
        )
        print(f"[Stage] Extra Test evaluation on {ds_name} | images: {len(extra_dataset)}")
        extra_loss, extra_acc, extra_auc, extra_f1, extra_preds_tensor, extra_labels_tensor = evaluate(
            model, extra_loader, criterion, device, 0
        )
        print(
            f"Extra Test ({ds_name}) - Loss: {extra_loss:.4f}, Acc: {extra_acc:.4f}, AUC: {extra_auc:.4f}, F1: {extra_f1:.4f}"
        )
        extra_test_results[ds_name] = {
            "loss": extra_loss,
            "acc": extra_acc,
            "auc": extra_auc,
            "f1": extra_f1,
            "confusion_matrix": confusion_matrix(extra_labels_tensor.cpu().numpy(), extra_preds_tensor.cpu().numpy()).tolist(),
        }

    results["extra_tests"] = extra_test_results

    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_dir}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GXMA Fusion Detector")
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config (optional if --resume is used)")
    parser.add_argument(
        "--mode",
        type=str,
        default="fusion",
        choices=["fusion", "frequency", "semantic"],
        help="Which model variant to train (fusion | frequency | semantic)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth) to resume training from",
    )
    args = parser.parse_args()
    main(args.config, args.mode, args.resume)
