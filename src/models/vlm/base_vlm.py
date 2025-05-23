from abc import ABC, abstractmethod
from PIL import Image
from typing import List, Any

class BaseVLM(ABC):
    """
    Abstract base class for Vision-Language Models.
    """
    def __init__(self, model_name: str, config: Any = None):
        self.model_name = model_name
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """
        Loads the pre-trained model and processor from Hugging Face or other sources.
        This method should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, image: Image.Image, text_prompts: List[str]) -> Any:
        """
        Performs zero-shot prediction (e.g., classification, image-text matching)
        based on the input image and text prompts.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of text prompts.

        Returns:
            Any: The prediction results, format depends on the specific VLM and task.
                 For classification, it might be a dictionary of prompts to scores.
        """
        pass

    def get_model_name(self) -> str:
        return self.model_name 