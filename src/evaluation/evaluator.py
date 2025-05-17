from typing import Dict, Any, List, Optional, Tuple
import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger
from src.utils.metrics import compute_metrics, compute_all_metrics


class Evaluator:
    """
    Evaluator class for model evaluation and visualization.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        output_dir: str = "results/evaluation",
        logger_name: str = "evaluator"
    ):
        """
        Initialize evaluator.
        Args:
            model (torch.nn.Module): Model to evaluate
            test_loader (DataLoader): Test data loader
            device (torch.device): Device to use
            output_dir (str): Directory to save evaluation results
            logger_name (str): Name for logger
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        self.logger = get_logger(logger_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                scores = outputs.softmax(dim=1)[:, 1]
                
                # Collect predictions
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        # Compute metrics
        metrics = compute_all_metrics(all_labels, all_preds, all_scores)
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value:.4f}")
        
        # Save metrics
        self._save_metrics(metrics)
        
        # Generate visualizations
        self._plot_roc_curve(all_labels, all_scores)
        self._plot_confusion_matrix(all_labels, all_preds)
        
        return metrics

    def _save_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Save metrics to file.
        Args:
            metrics (Dict[str, float]): Evaluation metrics
        """
        save_path = os.path.join(self.output_dir, "metrics.txt")
        with open(save_path, "w") as f:
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")

    def _plot_roc_curve(
        self,
        labels: np.ndarray,
        scores: np.ndarray
    ) -> None:
        """
        Plot ROC curve.
        Args:
            labels (np.ndarray): Ground truth labels
            scores (np.ndarray): Prediction scores
        """
        fpr, tpr, _ = roc_curve(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)
        
        save_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()

    def _plot_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> None:
        """
        Plot confusion matrix.
        Args:
            labels (np.ndarray): Ground truth labels
            preds (np.ndarray): Predictions
        """
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Real', 'AI-Generated'],
            yticklabels=['Real', 'AI-Generated']
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()

    def generate_gradcam(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        layer_name: str = "layer4"
    ) -> None:
        """
        Generate Grad-CAM visualizations.
        Args:
            images (torch.Tensor): Input images
            labels (torch.Tensor): Ground truth labels
            layer_name (str): Name of the layer to use for Grad-CAM
        """
        # TODO: Implement Grad-CAM visualization
        pass

    def evaluate_by_domain(
        self,
        domain_key: str = "generator"
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance by domain (e.g., generator type).
        
        Args:
            domain_key (str): Key in metadata to use for domain grouping
            
        Returns:
            Dict[str, Dict[str, float]]: Metrics per domain
        """
        self.logger.info(f"Evaluating by {domain_key}...")
        
        # Group samples by domain
        domain_samples = {}
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating by domain"):
                # Get batch data
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                metadata = batch[2]
                
                # Get domain for each sample
                domains = metadata[domain_key]
                
                # Forward pass
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                scores = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                # Group by domain
                for domain, label, pred, score in zip(
                    domains, labels.cpu().numpy(), preds, scores
                ):
                    if domain not in domain_samples:
                        domain_samples[domain] = {
                            "labels": [],
                            "preds": [],
                            "scores": []
                        }
                    
                    domain_samples[domain]["labels"].append(label)
                    domain_samples[domain]["preds"].append(pred)
                    domain_samples[domain]["scores"].append(score)
        
        # Compute metrics per domain
        domain_metrics = {}
        for domain, samples in domain_samples.items():
            metrics = compute_metrics(
                samples["labels"],
                samples["preds"]
            )
            
            # Add ROC AUC
            fpr, tpr, _ = roc_curve(
                samples["labels"],
                samples["scores"]
            )
            metrics["roc_auc"] = auc(fpr, tpr)
            
            domain_metrics[domain] = metrics
        
        # Save domain results if configured
        if self.config["evaluation"]["save_predictions"]:
            output_dir = self.config["evaluation"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            with open(
                os.path.join(output_dir, f"domain_metrics_{domain_key}.json"),
                "w"
            ) as f:
                json.dump(domain_metrics, f, indent=2)
        
        return domain_metrics 