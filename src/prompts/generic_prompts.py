from typing import List, Dict, Any
from .base_prompt_strategy import BasePromptStrategy

class GenImageDetectPrompts(BasePromptStrategy):
    """
    A prompt strategy tailored for detecting AI-generated images versus real images.
    Provides prompts suitable for CLIP/BLIP-ITM and question/keywords for generative VLMs.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Default prompts for discriminative models (e.g., CLIP)
        self.default_discriminative_prompts = [
            "a real photograph",
            "an authentic image",
            "a natural image",
            "an AI-generated image",
            "a computer-generated artwork",
            "a synthetic image"
        ]
        # Default question for generative models (e.g., LLaVA, InstructBLIP)
        self.default_generative_question = "Is this image a real photograph or an AI-generated image? Provide a concise answer focusing on the nature of the image."
        # Default keywords to check in the response of generative models
        self.default_keywords = ["real photograph", "ai-generated image", "authentic image", "synthetic image", "natural image", "computer-generated"] 

    def get_prompts(self, class_names: List[str] = None, image_info: Dict[str, Any] = None) -> List[str]:
        """
        Returns a list of prompts. 
        If class_names are provided (e.g. ["real", "generated"]), it can adapt.
        Otherwise, returns a more general set of prompts.

        These prompts are primarily intended for models like CLIP or BLIP-ITM that perform
        similarity matching between the image and each prompt.
        """
        if class_names:
            # Example: if class_names = ["real", "fake"], create more targeted prompts
            # This is a simple version; can be made more sophisticated
            prompts = []
            if "real" in class_names or "authentic" in class_names or "natural" in class_names:
                prompts.extend(["a real photograph", "an authentic image", "a natural image from a camera"])
            if "generated" in class_names or "fake" in class_names or "synthetic" in class_names or "ai" in class_names:
                prompts.extend(["an AI-generated image", "a computer-generated artwork", "a synthetic picture"])
            if not prompts: # Fallback if class_names are not recognized
                return self.config.get("discriminative_prompts", self.default_discriminative_prompts)
            return prompts
        
        return self.config.get("discriminative_prompts", self.default_discriminative_prompts)

    def get_vlm_question(self) -> str:
        """
        Returns a question suitable for generative VLMs to determine image authenticity.
        """
        return self.config.get("generative_question", self.default_generative_question)
    
    def get_keywords_for_response_check(self) -> List[str]:
        """
        Returns keywords to look for in the response of a generative VLM.
        These keywords help classify the image based on the VLM's textual output.
        """
        return self.config.get("generative_response_keywords", self.default_keywords)


class SimpleBinaryPrompts(BasePromptStrategy):
    """
    A very simple binary prompt strategy, expecting two class names like "real" and "generated".
    """
    def get_prompts(self, class_names: List[str] = None, image_info: Dict[str, Any] = None) -> List[str]:
        if not class_names or len(class_names) < 2:
            # Default for binary classification like GenImage (real/AI)
            return ["a real image", "an AI-generated image"]
        
        # Assumes class_names[0] is positive (e.g., real) and class_names[1] is negative (e.g., AI)
        # This can be made more robust with explicit mapping in config.
        return [
            f"a photograph of a {class_names[0]}", 
            f"an image of a {class_names[1]}"
        ]
    
    def get_vlm_question(self) -> str:
        # A generic question that can be paired with specific answer keywords
        return self.config.get("vlm_question", "What is depicted in this image and what is its nature?")

    def get_keywords_for_response_check(self) -> List[str]:
        # These would typically be the class names themselves or related terms
        return self.config.get("keywords_for_response_check", ["real", "generated", "photograph", "artwork"]) 