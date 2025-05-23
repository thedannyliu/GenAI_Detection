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
        Performs zero-shot prediction using BLIP. 
        For 'image-text-matching', calculates similarity scores.
        For 'image-captioning', it's not a direct classification, so this might need adaptation 
        or a different interpretation of "zero-shot" for this task.
        Here, we implement for 'image-text-matching'.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of text prompts.

        Returns:
            Dict[str, float]: A dictionary mapping each prompt to its similarity score (if task is matching).
                              Behavior for other tasks might differ.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        if self.task == "image-text-matching":
            try:
                # Process each prompt separately with the image
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

                return results
            except Exception as e:
                print(f"Error during BLIP image-text-matching prediction: {e}")
                return {prompt: 0.0 for prompt in text_prompts}
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
            return {prompt: 0.0 for prompt in text_prompts} # Placeholder
        else:
            # Should not happen if _load_model validated task
            raise ValueError(f"Unsupported BLIP task for prediction: {self.task}") 