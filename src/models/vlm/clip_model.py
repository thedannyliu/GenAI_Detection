from PIL import Image
from typing import List, Any, Dict
import torch
from transformers import CLIPProcessor, CLIPModel
from .base_vlm import BaseVLM

class CLIPModelWrapper(BaseVLM):
    """
    Wrapper for the CLIP model from Hugging Face.
    """
    def __init__(self, model_name: str, config: Any = None):
        super().__init__(model_name, config)
        self.prompt_strategy = None # Initialize prompt_strategy

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

            best_prompt_idx = probs.argmax().item()
            best_prompt_text = text_prompts[best_prompt_idx]
            
            predicted_label = self.prompt_strategy.get_class_for_prompt(best_prompt_text)
            
            if predicted_label is None:
                # --- Begin Debug Block ---
                print(f"DEBUG (CLIPModelWrapper): For VLM {self.model_name}, best_prompt_text='{best_prompt_text}' (type: {type(best_prompt_text)})")
                current_map_keys = list(self.prompt_strategy.prompt_to_class_map.keys())
                print(f"DEBUG (CLIPModelWrapper): Keys in prompt_strategy.prompt_to_class_map ({len(current_map_keys)} total): {current_map_keys}")
                
                found_match_direct_check = False
                for key_in_map in self.prompt_strategy.prompt_to_class_map.keys():
                    if key_in_map == best_prompt_text:
                        found_match_direct_check = True
                        break
                print(f"DEBUG (CLIPModelWrapper): Direct equality check for '{best_prompt_text}' in keys: {found_match_direct_check}")
                print(f"DEBUG (CLIPModelWrapper): Full prompt_to_class_map from strategy: {self.prompt_strategy.prompt_to_class_map}")
                # --- End Debug Block ---
                print(f"Warning for {self.model_name}: Best prompt '{best_prompt_text}' not in prompt_to_class_map. Using default/tie-breaking.")
                predicted_label = self.prompt_strategy.default_label_if_no_match # Fallback

            results = {}
            for i, prompt in enumerate(text_prompts):
                results[prompt] = probs[0][i].item()
            return results
        except Exception as e:
            print(f"Error during CLIP prediction: {e}")
            # You might want to return a specific error indicator or re-raise
            return {prompt: 0.0 for prompt in text_prompts} # Example error return

    def predict_batch(self, images: List[Image.Image], text_prompts: List[str]) -> List[Dict[str, float]]:
        """
        Performs zero-shot classification for a batch of images.

        Args:
            images (List[Image.Image]): A list of input images.
            text_prompts (List[str]): A list of text prompts for classification.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, where each dictionary 
                                     maps each prompt to its similarity score for an image.
        """
        batch_results = []
        for image in images:
            # Assuming the existing 'predict' method handles individual images and the same text_prompts
            # and returns a Dict[str, float] as expected by the evaluation script for each item in the list.
            image_scores = self.predict(image, text_prompts)
            batch_results.append(image_scores)
        return batch_results 