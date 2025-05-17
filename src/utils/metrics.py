from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)


def compute_metrics(
    labels: List[int],
    preds: List[int]
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        labels (List[int]): True labels
        preds (List[int]): Predicted labels
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }
    
    # Add confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
    
    return metrics


def compute_roc_metrics(
    labels: List[int],
    scores: List[float]
) -> Dict[str, Any]:
    """Compute ROC curve metrics.
    
    Args:
        labels (List[int]): True labels
        scores (List[float]): Prediction scores
        
    Returns:
        Dict[str, Any]: Dictionary containing ROC curve data
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold)
    }


def compute_pr_metrics(
    labels: List[int],
    scores: List[float]
) -> Dict[str, Any]:
    """Compute precision-recall curve metrics.
    
    Args:
        labels (List[int]): True labels
        scores (List[float]): Prediction scores
        
    Returns:
        Dict[str, Any]: Dictionary containing PR curve data
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = average_precision_score(labels, scores)
    
    # Find optimal threshold (F1 score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(pr_auc),
        "optimal_threshold": float(optimal_threshold)
    }


def compute_calibration_metrics(
    labels: List[int],
    scores: List[float],
    n_bins: int = 10
) -> Dict[str, Any]:
    """Compute calibration metrics.
    
    Args:
        labels (List[int]): True labels
        scores (List[float]): Prediction scores
        n_bins (int): Number of bins for calibration
        
    Returns:
        Dict[str, Any]: Dictionary containing calibration metrics
    """
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(scores, bin_edges) - 1
    
    # Compute calibration metrics per bin
    calibration_data = []
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_scores = np.array(scores)[bin_mask]
            bin_labels = np.array(labels)[bin_mask]
            
            mean_score = np.mean(bin_scores)
            mean_label = np.mean(bin_labels)
            count = len(bin_scores)
            
            calibration_data.append({
                "bin": i,
                "mean_score": float(mean_score),
                "mean_label": float(mean_label),
                "count": int(count)
            })
    
    # Compute ECE (Expected Calibration Error)
    ece = np.mean([
        abs(d["mean_score"] - d["mean_label"]) * d["count"]
        for d in calibration_data
    ]) / len(scores)
    
    return {
        "calibration_data": calibration_data,
        "ece": float(ece)
    }


def compute_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute accuracy score.
    Args:
        labels (np.ndarray): Ground truth labels
        predictions (np.ndarray): Model predictions
    Returns:
        float: Accuracy score
    """
    return accuracy_score(labels, predictions)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve (AUC).
    Args:
        labels (np.ndarray): Ground truth labels
        scores (np.ndarray): Model prediction scores
    Returns:
        float: AUC score
    """
    return roc_auc_score(labels, scores)


def compute_precision_recall_f1(labels: np.ndarray, predictions: np.ndarray) -> tuple:
    """
    Compute precision, recall, and F1 score.
    Args:
        labels (np.ndarray): Ground truth labels
        predictions (np.ndarray): Model predictions
    Returns:
        tuple: (precision, recall, f1) scores
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    return precision, recall, f1


def compute_all_metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    """
    Compute all metrics and return as a dictionary.
    Args:
        labels (np.ndarray): Ground truth labels
        predictions (np.ndarray): Model predictions
        scores (np.ndarray): Model prediction scores
    Returns:
        dict: Dictionary containing all metrics
    """
    accuracy = compute_accuracy(labels, predictions)
    auc = compute_auc(labels, scores)
    precision, recall, f1 = compute_precision_recall_f1(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 