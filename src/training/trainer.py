from typing import Optional, Dict, Any, List
import os
import time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.logger import get_logger
from src.utils.metrics import compute_all_metrics


class Trainer:
    """Trainer class for model training and evaluation.
    
    This class handles the training loop, validation, and model checkpointing.
    It supports both fine-tuning and zero-shot evaluation modes.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for training
        config (Dict): Training configuration
        logger_name (str): Name for logger
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        logger_name: str = "trainer"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.logger = get_logger(logger_name)
        
        # Training state
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.best_model_path = os.path.join(
            config.get("output_dir", "results/saved_models"),
            "best_model.pth"
        )
        
        # Create output directory
        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
        
        # Set up optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
    
    def _setup_optimizer(self):
        """Set up the optimizer with the specified configuration."""
        train_config = self.config["training"]
        
        # Get parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            params,
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"]
        )
    
    def _setup_scheduler(self):
        """Set up the learning rate scheduler."""
        train_config = self.config["training"]
        
        if train_config["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config["num_epochs"],
                eta_min=0
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dict[str, float]: Training metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Compute metrics
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            outputs.detach().cpu().numpy()
        )
        metrics["loss"] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dict[str, float]: Validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(outputs.softmax(dim=1)[:, 1].cpu().numpy())
        
        # Compute metrics
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_scores)
        )
        metrics["loss"] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model for specified number of epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
        
        Returns:
            Dict[str, List[float]]: Training history
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
                f"Train AUC: {train_metrics['auc']:.4f}"
            )
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )
            
            # Update history
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_auc"].append(train_metrics["auc"])
            history["val_auc"].append(val_metrics["auc"])
            
            # Save best model
            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.save_checkpoint()
                self.logger.info(f"Saved best model with AUC: {self.best_val_auc:.4f}")
        
        return history
    
    def save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
            "config": self.config
        }
        torch.save(checkpoint, self.best_model_path)
    
    def load_checkpoint(self, path: Optional[str] = None) -> None:
        """Load model checkpoint.
        
        Args:
            path (Optional[str]): Path to checkpoint file
        """
        if path is None:
            path = self.best_model_path
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_auc = checkpoint["best_val_auc"]
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step() 