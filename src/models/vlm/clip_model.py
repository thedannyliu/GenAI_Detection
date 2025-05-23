from PIL import Image
from typing import List, Any, Dict
import torch
from transformers import CLIPProcessor, CLIPModel
from .base_vlm import BaseVLM

class CLIPModelWrapper(BaseVLM):
    """
    Wrapper for the CLIP model from Hugging Face.
    """
    def _load_model(self):
        """
        Loads the pre-trained CLIP model and processor.
        Uses "openai/clip-vit-large-patch14" as the default model.
        """
        model_id = self.config.get("model_id", "openai/clip-vit-large-patch14")
        try:
            self.model = CLIPModel.from_pretrained(model_id)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval() # Set to evaluation mode
        except Exception as e:
            print(f"Error loading CLIP model {model_id}: {e}")
            raise

    def predict(self, image: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """
        Performs zero-shot classification using CLIP.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of text prompts for classification.

        Returns:
            Dict[str, float]: A dictionary mapping each prompt to its similarity score.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get probabilities

            results = {}
            for i, prompt in enumerate(text_prompts):
                results[prompt] = probs[0][i].item()
            return results
        except Exception as e:
            print(f"Error during CLIP prediction: {e}")
            # You might want to return a specific error indicator or re-raise
            return {prompt: 0.0 for prompt in text_prompts} # Example error return 