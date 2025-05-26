import os
from typing import Tuple, Optional, Dict, List, Set
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

class GenImageDataset(Dataset):
    """
    Dataset class for loading GenImage dataset.
    Supports random sampling of a fixed number of images per class.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        num_samples_per_class: Optional[int] = None, # Added for sampling
        exclude_files: Optional[Set[str]] = None,    # Added for exclusion (used by test set)
        seed: int = 42 # Added for reproducibility of sampling
    ):
        """
        Initialize dataset.
        Args:
            root_dir (str): Root directory of dataset
            split (str): 'train' or 'val' or 'test'. Note: 'test' split uses 'val' directory.
            transform (Optional[transforms.Compose]): Image transformations
            class_to_idx (Optional[Dict[str, int]]): Mapping from class names to indices
            num_samples_per_class (Optional[int]): Number of images to sample per class. If None, all images are used.
            exclude_files (Optional[Set[str]]): A set of absolute image paths to exclude.
            seed (int): Random seed for sampling.
        """
        self.root_dir = root_dir
        self.data_split_dir = split # Directly use the provided split name
        self.transform = transform or self._get_default_transform()
        self.class_to_idx = class_to_idx
        self.num_samples_per_class = num_samples_per_class
        self.exclude_files = exclude_files or set()
        self.seed = seed
        
        # Setup paths
        self.nature_dir = os.path.join(root_dir, self.data_split_dir, "nature")
        self.ai_dir = os.path.join(root_dir, self.data_split_dir, "ai")
        
        # Get all image paths and labels
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        random.seed(self.seed) # Set seed before listing and sampling

        # Dynamically load images based on class_to_idx
        if not self.class_to_idx:
            # Fallback or error if class_to_idx is not provided, 
            # though the VLMGenImageDataset in zero_shot_vlm_eval.py seems to always get it from config.
            # For robustness, we could default to the old behavior or raise an error.
            # Here, we'll print a warning and attempt the old hardcoded paths if class_to_idx is empty.
            print("Warning: class_to_idx is empty. Attempting to load with default 'nature' and 'ai' subdirectories.")
            default_class_map = {"nature": 0, "ai": 1}
            for class_name, class_idx in default_class_map.items(): # class_idx is not directly used here, but good to have
                class_dir = os.path.join(root_dir, self.data_split_dir, class_name)
                self._load_images_from_class_dir(class_dir, class_name) # Pass class_name for label mapping
        else:
            for class_name, class_idx in self.class_to_idx.items(): # class_idx is not directly used here
                # class_name here will be "0_real", "1_fake" for Chameleon
                # or "nature", "ai" for others, as defined in YAML.
                class_dir = os.path.join(root_dir, self.data_split_dir, class_name)
                self._load_images_from_class_dir(class_dir, class_name) # Pass class_name for label mapping

    def _load_images_from_class_dir(self, class_root_dir: str, class_name: str):
        """Helper function to load images for a specific class, with sampling."""
        candidate_images = []
        if os.path.exists(class_root_dir) and os.path.isdir(class_root_dir):
            # Check if images are directly under class_root_dir or in subdirectories
            # This simplified logic assumes images are directly under nature/ai folders,
            # or one level down if those contain subfolders (as per original logic).
            # For GenImage, it seems images are directly under 'nature' and 'ai'.
            
            potential_image_files = []
            # First, try to list files directly in class_root_dir
            for item_name in os.listdir(class_root_dir):
                item_path = os.path.join(class_root_dir, item_name)
                if os.path.isfile(item_path) and item_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if item_path not in self.exclude_files:
                        potential_image_files.append(item_path)
                elif os.path.isdir(item_path): # If it's a directory, look inside it
                    for img_name in os.listdir(item_path):
                        img_path_nested = os.path.join(item_path, img_name)
                        if os.path.isfile(img_path_nested) and img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                             if img_path_nested not in self.exclude_files:
                                potential_image_files.append(img_path_nested)

            if self.num_samples_per_class and len(potential_image_files) > self.num_samples_per_class:
                candidate_images.extend(random.sample(potential_image_files, self.num_samples_per_class))
            else:
                candidate_images.extend(potential_image_files)
                if self.num_samples_per_class and len(potential_image_files) < self.num_samples_per_class:
                     print(f"Warning: For class '{class_name}' in '{class_root_dir}', wanted {self.num_samples_per_class} samples, but only found {len(potential_image_files)} (after exclusions). Using all available.")
            
            self.image_paths.extend(candidate_images)
            self.labels.extend([self.class_to_idx[class_name]] * len(candidate_images))
        else:
            print(f"Warning: Directory not found or is not a directory: {class_root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single data sample.
        Args:
            idx (int): Index of the sample
        Returns:
            Tuple[torch.Tensor, int]: (image tensor, label)
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_image_paths(self) -> List[str]:
        """Returns the list of image paths used by this dataset instance."""
        return self.image_paths

    @staticmethod
    def _get_default_transform() -> transforms.Compose:
        """
        Get default image transformations.
        Returns:
            transforms.Compose: Default transforms
        """
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) 