from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptStrategy(ABC):
    """
    Abstract base class for defining prompt generation strategies for VLMs.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config if config is not None else {}

    @abstractmethod
    def get_prompts(self, class_names: List[str] = None, image_info: Dict[str, Any] = None) -> List[str]:
        """
        Generates a list of text prompts based on the strategy.

        Args:
            class_names (List[str], optional): A list of class names that might be used 
                                               by some strategies (e.g., for classification).
                                               Defaults to None.
            image_info (Dict[str, Any], optional): Additional information about the image 
                                                   that might be used by some strategies.
                                                   Defaults to None.

        Returns:
            List[str]: A list of text prompts to be used with a VLM.
        """
        pass

    def get_vlm_question(self) -> str:
        """
        Returns a specific question to be used for generative VLMs (like InstructBLIP, LLaVA)
        if the strategy requires a particular question format.
        Otherwise, the VLM's default question (from its own config) might be used.

        Returns:
            str: A question string, or None if no specific question is defined by this strategy.
        """
        return self.config.get("vlm_question", None)

    def get_keywords_for_response_check(self) -> List[str]:
        """
        For generative VLMs, these are the keywords/phrases to check for in the model's response.
        This might be different from the `get_prompts` output, which could be more general purpose
        or used by discriminative models like CLIP.

        Returns:
            List[str]: A list of keywords, or None.
        """
        return self.config.get("keywords_for_response_check", None) 