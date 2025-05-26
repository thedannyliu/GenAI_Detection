from PIL import Image
from typing import List, Any, Dict
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval
from .base_vlm import BaseVLM

class BlipModelWrapper(BaseVLM):
    """
    Wrapper for the BLIP model from Hugging Face.
    This wrapper can handle different BLIP variations based on config.
    Default is "Salesforce/blip-image-captioning-large" for captioning, 
    or "Salesforce/blip-itm-large-eval" for image-text matching.
    """
    def __init__(self, model_name: str, config: Any = None):
        super().__init__(model_name, config)
        self.prompt_strategy = None # Initialize prompt_strategy

    def _load_model(self):
        model_id = self.config.get("model_id", "Salesforce/blip-itm-large-eval") # Default to ITM for zero-shot classification
        self.task = self.config.get("task", "image-text-matching") # or "image-captioning"

        try:
            self.processor = BlipProcessor.from_pretrained(model_id)
            if self.task == "image-text-matching":
                self.model = BlipForImageTextRetrieval.from_pretrained(model_id)
            elif self.task == "image-captioning": # Though not directly for zero-shot classification, can be adapted
                self.model = BlipForConditionalGeneration.from_pretrained(model_id)
            else:
                raise ValueError(f"Unsupported BLIP task: {self.task}. Choose 'image-text-matching' or 'image-captioning'.")

            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
        except Exception as e:
            print(f"Error loading BLIP model {model_id} for task {self.task}: {e}")
            raise

    def predict(self, image: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """
        Performs prediction using BLIP for a single image and multiple text prompts.
        For 'image-text-matching', calculates similarity scores.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of text prompts.

        Returns:
            Dict[str, float]: A dictionary mapping each prompt to its similarity score.
        """
        if self.model is None or self.processor is None:
            self._load_model() # Ensure model is loaded

        if self.task == "image-text-matching":
            try:
                results = {}
                for prompt in text_prompts:
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                    # Get device from the model itself, as self.device might not be reliably set
                    model_device = self.model.device 
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # For ITM, the second column of itm_score contains the logits for the positive class (text matches the image).
                    match_logit = outputs.itm_score[0, 1] 
                    results[prompt] = torch.sigmoid(match_logit).item() # Convert to pseudo-probability
                return results
            except Exception as e:
                print(f"Error during BLIP image-text-matching prediction: {e}")
                return {prompt: 0.0 for prompt in text_prompts}
        elif self.task == "image-captioning":
            print("Warning: BLIP 'image-captioning' task not directly suited for prompt-based classification. Returning zero scores.")
            return {prompt: 0.0 for prompt in text_prompts}
        else:
            raise ValueError(f"Unsupported BLIP task for prediction: {self.task}")

    def predict_batch(self, images: List[Image.Image], text_prompts: List[str]) -> List[Dict[str, float]]:
        """
        Wrapper for predict to handle a "batch" of images by iterating.
        This is for API consistency if the main eval loop sends a list, even if bsize=1.
        BLIP ITM typically processes one image-text pair at a time due to processor.
        """
        # if not images: # Handle case of empty images list
        #     return []
            
        # For simplicity, and given that batch_size is 1 in the script,
        # this will just call predict for the first (and only) image.
        # If batch_size > 1 were truly supported for images with ITM, this would loop.
        if len(images) != 1:
            # This case should ideally not be hit with batch_size=1 in eval script
            print(f"Warning (BlipModelWrapper.predict_batch): Expected 1 image in batch, got {len(images)}. Processing first image.")
        
        # The eval loop `run_evaluation_for_model` expects `predict_batch` to take a list of images.
        # However, with `vlm_collate_fn`, `batch_images` is `List[Image.Image]`.
        # And the loop calls `model.predict_batch([current_image_pil], current_prompts)`
        # So, `images` here will be `[current_image_pil]`.
        
        return [self.predict(images[0], text_prompts)]

    def predict_batch_with_prompt_strategy(self, images: List[Image.Image], text_prompts: List[str]) -> List[Dict[str, float]]:
        """
        Performs zero-shot prediction using BLIP for a batch of images and text prompts with a prompt strategy.
        For 'image-text-matching', calculates similarity scores.
        For 'image-captioning', it's not a direct classification, so this might need adaptation 
        or a different interpretation of "zero-shot" for this task.
        Here, we implement for 'image-text-matching'.

        Args:
            images (List[Image.Image]): The input images.
            text_prompts (List[str]): A list of text prompts.

        Returns:
            List[Dict[str, float]]: A list of dictionaries mapping each prompt to its similarity score (if task is matching).
                                    Behavior for other tasks might differ.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if self.task == "image-text-matching":
            try:
                # Process each prompt separately with each image
                results_batch = []
                for image in images:
                    results = {}
                    for prompt in text_prompts:
                        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                        if torch.cuda.is_available():
                            inputs = {k: v.to("cuda") for k, v in inputs.items()}
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        
                        # For ITM, the itm_score is a logit. Higher means more related.
                        # We can use it directly or apply sigmoid for a probability-like score.
                        # outputs.itm_score is usually a tensor of shape [batch_size, 2] (match, no_match)
                        # We are interested in the 'match' score if available in this format, 
                        # or the direct output if it's a single score per pair.
                        # BlipForImageTextRetrieval output for ITM is a dict with `itm_score`
                        # The `itm_score` contains logits for matching the image with the text.
                        # It's a tensor, e.g., tensor([[-0.3691,  0.4008]], device='cuda:0')
                        # where the second value can be interpreted as the "match" score.
                        # It might be better to consult specific model docs for exact interpretation.
                        # Here, we'll assume the itm_score directly reflects similarity for simplicity or use a specific head if available.
                        # A common approach for ITM models is to get the probability that text matches image.
                        # For simplicity, we take the raw itm_score for the positive class (if applicable) or a normalized score.
                        # Let's assume itm_score directly provides a comparable value or we take the 'match' logit.
                        # Some BLIP ITM models provide `vision_model_output` and `text_model_output` then a multimedia head. 
                        # Salesforce/blip-itm-large-eval output structure:
                        # BlipOutputWithPoolingAndCrossAttention / BlipImageTextMatchingModelOutput
                        #   itm_score: `torch.FloatTensor` of shape `(batch_size, num_labels)` 
                        #   where num_labels is usually 2 (match, no_match)
                        # The first column of itm_score contains the logits for the negative class (text does not match the image) 
                        # and the second column of itm_score contains the logits for the positive class (text matches the image).
                        
                        # For a single image and single prompt, batch_size is 1.
                        # We want the score for "text matches the image", which is the second logit.
                        match_logit = outputs.itm_score[0, 1] 
                        results[prompt] = torch.sigmoid(match_logit).item() # Convert to pseudo-probability

                    # Apply prompt strategy to get the predicted label
                    probs = torch.tensor(list(results.values()))
                    best_prompt_idx = probs.argmax().item()
                    best_prompt_text = text_prompts[best_prompt_idx]
                    
                    predicted_label = self.prompt_strategy.get_class_for_prompt(best_prompt_text)

                    if predicted_label is None:
                        # --- Begin Debug Block ---
                        print(f"DEBUG (BlipModelWrapper): For VLM {self.model_name}, best_prompt_text='{best_prompt_text}' (type: {type(best_prompt_text)})")
                        current_map_keys = list(self.prompt_strategy.prompt_to_class_map.keys())
                        print(f"DEBUG (BlipModelWrapper): Keys in prompt_strategy.prompt_to_class_map ({len(current_map_keys)} total): {current_map_keys}")
                        
                        found_match_direct_check = False
                        for key_in_map in self.prompt_strategy.prompt_to_class_map.keys():
                            if key_in_map == best_prompt_text:
                                found_match_direct_check = True
                                break
                        print(f"DEBUG (BlipModelWrapper): Direct equality check for '{best_prompt_text}' in keys: {found_match_direct_check}")
                        print(f"DEBUG (BlipModelWrapper): Full prompt_to_class_map from strategy: {self.prompt_strategy.prompt_to_class_map}")
                        # --- End Debug Block ---
                        print(f"Warning for {self.model_name}: Best prompt '{best_prompt_text}' not in prompt_to_class_map. Using default/tie-breaking for ITM.")
                        # For ITM, if no prompt matches, it's ambiguous. We might rely on a default or tie-breaking in the main script.
                        # Here, we'll set it to a value that indicates no definitive class from prompts.
                        # The main script will then use `tie_breaking_label_for_generative` or `default_label_for_generative_no_match`.
                        # However, these are for generative. For ITM, it's different. The class is derived from the prompt.
                        # If the *best* prompt isn't in the map, it's a configuration issue.
                        predicted_label = self.prompt_strategy.default_label_if_no_match # Fallback based on GenImageDetectPrompts default

                    results_batch.append({
                        "predicted_label": predicted_label,
                        "prompt_scores": results
                    })

                return results_batch
            except Exception as e:
                print(f"Error during BLIP image-text-matching prediction: {e}")
                return [{prompt: 0.0 for prompt in text_prompts} for _ in images]
        elif self.task == "image-captioning":
            # This is not a direct classification. 
            # We could generate a caption and then compare it to prompts, but that's more complex.
            # For now, returning empty or raising error for this task in a classification context.
            print("Warning: BLIP 'image-captioning' task not directly suited for prompt-based classification in this method.")
            # Example: generate caption (not used for classification score here)
            # inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
            # generated_ids = self.model.generate(**inputs, max_length=50)
            # generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            # print(f"Generated caption: {generated_caption}")
            return [{prompt: 0.0 for prompt in text_prompts} for _ in images] # Placeholder
        else:
            # Should not happen if _load_model validated task
            raise ValueError(f"Unsupported BLIP task for prediction: {self.task}") 