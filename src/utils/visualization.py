import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from PIL import Image
import cv2


def plot_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> None:
    """Plot training metrics.
    
    Args:
        metrics (Dict[str, List[float]]): Dictionary of metrics to plot
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: Dict[str, int],
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix.
    
    Args:
        cm (Dict[str, int]): Confusion matrix dictionary
        save_path (Optional[str]): Path to save the plot
    """
    # Convert dictionary to numpy array
    cm_array = np.array([
        [cm["tn"], cm["fp"]],
        [cm["fn"], cm["tp"]]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auc: float,
    save_path: Optional[str] = None
) -> None:
    """Plot ROC curve.
    
    Args:
        fpr (List[float]): False positive rates
        tpr (List[float]): True positive rates
        auc (float): Area under curve
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_pr_curve(
    precision: List[float],
    recall: List[float],
    auc: float,
    save_path: Optional[str] = None
) -> None:
    """Plot precision-recall curve.
    
    Args:
        precision (List[float]): Precision values
        recall (List[float]): Recall values
        auc (float): Area under curve
        save_path (Optional[str]): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR (AUC = {auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    calibration_data: List[Dict[str, Any]],
    ece: float,
    save_path: Optional[str] = None
) -> None:
    """Plot calibration curve.
    
    Args:
        calibration_data (List[Dict[str, Any]]): Calibration data per bin
        ece (float): Expected calibration error
        save_path (Optional[str]): Path to save the plot
    """
    mean_scores = [d["mean_score"] for d in calibration_data]
    mean_labels = [d["mean_label"] for d in calibration_data]
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_scores, mean_labels, "o-", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Mean predicted score")
    plt.ylabel("Mean true label")
    plt.title(f"Calibration Curve (ECE = {ece:.3f})")
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_attention(
    image: torch.Tensor,
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """Visualize attention weights on image.
    
    Args:
        image (torch.Tensor): Input image tensor
        attention_weights (torch.Tensor): Attention weights
        save_path (Optional[str]): Path to save the visualization
    """
    # Convert image to numpy array
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Resize attention weights to match image size
    attention_map = cv2.resize(
        attention_weights.cpu().numpy(),
        (img.shape[1], img.shape[0])
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap="jet")
    plt.title("Attention Map")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(attention_map, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_gradcam(
    image: torch.Tensor,
    gradcam_map: torch.Tensor,
    save_path: Optional[str] = None
) -> None:
    """Visualize Grad-CAM on image.
    
    Args:
        image (torch.Tensor): Input image tensor
        gradcam_map (torch.Tensor): Grad-CAM map
        save_path (Optional[str]): Path to save the visualization
    """
    # Convert image to numpy array
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min())
    
    # Resize Grad-CAM map to match image size
    gradcam = cv2.resize(
        gradcam_map.cpu().numpy(),
        (img.shape[1], img.shape[0])
    )
    
    # Create heatmap
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(gradcam, cmap="jet")
    plt.title("Grad-CAM Map")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(gradcam, cmap="jet", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 