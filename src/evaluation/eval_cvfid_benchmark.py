#!/usr/bin/env python
"""Batch evaluation of C-VFiD model on AIGCDetectionBenchmark test splits.

Usage example:
    python -m src.evaluation.eval_cvfid_benchmark \
        --ckpt results/cvfid_replication/best_cvfid.pt \
        --test_root AIGCDetectionBenchMark/AIGCDetectionBenchMark/test \
        --output_dir results/cvfid_eval

The script will iterate over the *generator* sub-folders under ``test_root``
(e.g. ``stable_diffusion_v_1_5``), evaluate on each, compute metrics
(accuracy, AUC, AP, precision, recall, F1) and produce:

    • <output_dir>/metrics.csv          – per-generator + average table
    • <output_dir>/<gen>_metrics.json   – raw metrics per generator
    • <output_dir>/<gen>_roc.png        – ROC curve
    • <output_dir>/<gen>_pr.png         – Precision-Recall curve
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.models.r_vfid.model import RvfidModel
from src.data_processing.benchmark_dataset import BenchmarkImageDataset, build_default_transform
from src.utils.metrics import (
    compute_all_metrics,
    compute_pr_metrics,
    compute_roc_metrics,
)


# -----------------------------------------------------------------------------
#   Evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
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
    metrics["roc"] = compute_roc_metrics(y_true, y_score)
    metrics["pr"] = compute_pr_metrics(y_true, y_score)
    return metrics


# -----------------------------------------------------------------------------
#   Plotting utilities
# -----------------------------------------------------------------------------

def _plot_roc(roc: Dict[str, Any], title: str, save_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.plot(roc["fpr"], roc["tpr"], label=f"AUC={roc['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _plot_pr(pr: Dict[str, Any], title: str, save_path: Path) -> None:
    plt.figure(figsize=(4, 4))
    plt.plot(pr["recall"], pr["precision"], label=f"AP={pr['auc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# -----------------------------------------------------------------------------
#   Main
# -----------------------------------------------------------------------------

def main(args):
    device_ids = [int(x) for x in args.gpus.split(",") if x]
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Model ----------------
    # Need num_experts & gating info; we allow CLI overrides
    model = RvfidModel(num_experts=args.num_experts, gating_mode=args.gating_mode)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state", ckpt)  # tolerate raw state_dict
    model.load_state_dict(state, strict=False)

    if len(device_ids) > 1 and torch.cuda.device_count() >= len(device_ids):
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)
    model.eval()

    # ---------------- Generators list ----------------
    test_root = Path(args.test_root)
    generators = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
    if args.generators:  # allow manual subset
        generators = [g for g in generators if g in args.generators.split(",")]
    print(f"[Info] Evaluating {len(generators)} generators: {generators}")

    transform = build_default_transform()
    batch_size = args.batch_size

    table_rows = []

    for gen in tqdm(generators, desc="Generators"):
        gen_dir = test_root / gen
        dataset = BenchmarkImageDataset(gen_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        metrics = evaluate_loader(model, loader, device)

        # Save curves & metrics
        with open(out_dir / f"{gen}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        _plot_roc(metrics["roc"], f"{gen} ROC", out_dir / f"{gen}_roc.png")
        _plot_pr(metrics["pr"], f"{gen} PR", out_dir / f"{gen}_pr.png")

        table_rows.append({
            "generator": gen,
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "ap": metrics["pr"]["auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        })

    # ---------------- Aggregate ----------------
    df = pd.DataFrame(table_rows)
    avg_row = {k: (df[k].mean() if k != "generator" else "Average") for k in df.columns}
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Done] Evaluation complete. Results saved to {csv_path}")


# -----------------------------------------------------------------------------
#   CLI parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Evaluate C-VFiD on AIGCDetectionBenchmark generators")
    p.add_argument("--ckpt", required=True, help="Path to trained model checkpoint (.pt)")
    p.add_argument("--test_root", required=True, help="Root of benchmark test directory")
    p.add_argument("--output_dir", default="results/cvfid_eval", help="Directory to save metrics & plots")
    p.add_argument("--num_experts", type=int, default=3, help="Number of experts in the model (must match training)")
    p.add_argument("--gating_mode", choices=["softmax", "sigmoid"], default="sigmoid")
    p.add_argument("--gpus", default="0", help="GPU ids, e.g. '0' or '0,1,2,3'")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--generators", default="", help="Comma-separated subset of generators to evaluate (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args) 