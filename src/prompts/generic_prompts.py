from PIL import Image
from typing import List, Dict, Any, Optional
from .base_prompt_strategy import BasePromptStrategy

class GenImageDetectPrompts(BasePromptStrategy):
    """
    A prompt strategy tailored for detecting AI-generated images versus real images.
    Provides prompts suitable for CLIP/BLIP-ITM and question/keywords for generative VLMs.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Normalize keys in prompt_to_class_map and keyword_to_class_map upon initialization
        self.prompt_to_class_map: Dict[str, int] = {
            k.lower().strip(): v 
            for k, v in self.config.get("prompt_to_class_map", {}).items()
        }
        self.keyword_to_class_map: Dict[str, int] = {
            k.lower().strip(): v 
            for k, v in self.config.get("keyword_to_class_map", {}).items()
        }
        self.default_label_if_no_match: int = self.config.get("default_label_if_no_match", 1) # Default to AI if not specified
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

    def get_class_for_prompt(self, prompt_text: str) -> Optional[int]:
        """Returns the class index for a given prompt text, or None if not found."""
        # Normalize the prompt text for lookup (e.g., lowercasing)
        normalized_prompt = prompt_text.lower().strip()
        
        # Direct match in prompt_to_class_map
        if normalized_prompt in self.prompt_to_class_map:
            return self.prompt_to_class_map[normalized_prompt]
        
        # --- Begin Debug Block for Missing Prompt ---
        if prompt_text in self.prompt_to_class_map: # Check original casing too, just in case
             # This case should ideally not be hit if normalization is consistent
             print(f"DEBUG (GenImageDetectPrompts): Prompt '{prompt_text}' found with original casing but not normalized '{normalized_prompt}'. This is unusual.")
             return self.prompt_to_class_map[prompt_text]

        # If not found, print debug information before returning None
        # This helps diagnose why a specific prompt might not be mapping correctly.
        print(f"DEBUG (GenImageDetectPrompts): Prompt '{prompt_text}' (normalized: '{normalized_prompt}') not found in prompt_to_class_map.")
        print(f"DEBUG (GenImageDetectPrompts): Current prompt_to_class_map (normalized keys used for lookup): {self.prompt_to_class_map}")
        # You might also want to log the original config's map if it differs from the processed one stored in self.prompt_to_class_map
        # print(f"DEBUG (GenImageDetectPrompts): Original config map was: {self.config.get('prompt_to_class_map', {})}")
        # --- End Debug Block ---

        return None # Default if no direct match

    def get_class_for_keywords(self, text: str) -> Optional[int]:
        # Implementation of get_class_for_keywords method
        pass

    def get_prompts_for_image(self, image: Optional[Image.Image] = None, class_label: Optional[int] = None) -> List[str]:
        """Returns a list of prompts, typically for discriminative VLMs.
        Ignores image and class_label for this generic strategy, returning general prompts.
        """
        return self.get_prompts(class_names=None, image_info=None)

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