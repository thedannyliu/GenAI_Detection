import random
import numpy as np
import torch
import yaml
from typing import Dict, Any, Optional

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    Args:
        path (str): Path to YAML file
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """
    Get available device (GPU or CPU).
    Returns:
        torch.device: Device to use
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file.
    Args:
        config (Dict[str, Any]): Configuration dictionary
        path (str): Path to save YAML file
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    Args:
        model (torch.nn.Module): PyTorch model
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 