import os
from typing import Tuple, Optional, Dict, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class GenImageDataset(Dataset):
    """
    Dataset class for loading GenImage dataset.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        """
        Initialize dataset.
        Args:
            root_dir (str): Root directory of dataset
            split (str): 'train' or 'val'
            transform (Optional[transforms.Compose]): Image transformations
            class_to_idx (Optional[Dict[str, int]]): Mapping from class names to indices
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._get_default_transform()
        self.class_to_idx = class_to_idx or {"nature": 0, "ai": 1}
        
        # Setup paths
        self.nature_dir = os.path.join(root_dir, split, "nature")
        self.ai_dir = os.path.join(root_dir, split, "ai")
        
        # Get all image paths and labels
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        # Load nature (real) images
        if os.path.exists(self.nature_dir) and os.path.isdir(self.nature_dir):
            # Check if the first level items are directories (class folders) or files
            first_item_in_nature = next(os.scandir(self.nature_dir), None)
            if first_item_in_nature and first_item_in_nature.is_dir():
                # Original logic: Iterate through class folders
                for class_name_dir in os.listdir(self.nature_dir):
                    class_dir_path = os.path.join(self.nature_dir, class_name_dir)
                    if os.path.isdir(class_dir_path):
                        for img_name in os.listdir(class_dir_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                self.image_paths.append(os.path.join(class_dir_path, img_name))
                                self.labels.append(self.class_to_idx["nature"])
            elif first_item_in_nature and first_item_in_nature.is_file():
                # Modified logic: Images are directly under nature_dir
                for img_name in os.listdir(self.nature_dir):
                    img_path = os.path.join(self.nature_dir, img_name)
                    if os.path.isfile(img_path) and img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx["nature"])
            # else: the directory is empty or contains non-image/non-dir items only at first level.
        else:
            print(f"Warning: Nature directory not found or is not a directory: {self.nature_dir}")

        # Load AI-generated images
        if os.path.exists(self.ai_dir) and os.path.isdir(self.ai_dir):
            first_item_in_ai = next(os.scandir(self.ai_dir), None)
            if first_item_in_ai and first_item_in_ai.is_dir():
                # Original logic: Iterate through class folders
                for class_name_dir in os.listdir(self.ai_dir):
                    class_dir_path = os.path.join(self.ai_dir, class_name_dir)
                    if os.path.isdir(class_dir_path):
                        for img_name in os.listdir(class_dir_path):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                self.image_paths.append(os.path.join(class_dir_path, img_name))
                                self.labels.append(self.class_to_idx["ai"])
            elif first_item_in_ai and first_item_in_ai.is_file():
                # Modified logic: Images are directly under ai_dir
                for img_name in os.listdir(self.ai_dir):
                    img_path = os.path.join(self.ai_dir, img_name)
                    if os.path.isfile(img_path) and img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx["ai"])
            # else: the directory is empty or contains non-image/non-dir items only at first level.
        else:
            print(f"Warning: AI directory not found or is not a directory: {self.ai_dir}")

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