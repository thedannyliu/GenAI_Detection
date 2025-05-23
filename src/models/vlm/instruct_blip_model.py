from PIL import Image
from typing import List, Any, Dict
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from .base_vlm import BaseVLM

class InstructBlipModelWrapper(BaseVLM):
    """
    Wrapper for the InstructBLIP model from Hugging Face.
    """
    def _load_model(self):
        model_id = self.config.get("model_id", "Salesforce/instructblip-vicuna-7b") # Example, choose appropriate model
        try:
            self.processor = InstructBlipProcessor.from_pretrained(model_id)
            self.model = InstructBlipForConditionalGeneration.from_pretrained(model_id)
            
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
        except Exception as e:
            print(f"Error loading InstructBLIP model {model_id}: {e}")
            raise

    def predict(self, image: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """
        Performs zero-shot prediction using InstructBLIP. 
        InstructBLIP is generative, so true "zero-shot classification" typically involves 
        asking the model a question and interpreting its generated answer.
        For simplicity in this context, we might try to see if it generates one of the prompts 
        or assign scores based on perplexity if possible, but this is less straightforward 
        than models with direct classification/matching heads.

        A common approach is to phrase prompts as questions and evaluate the generated text.
        E.g., for "real image" vs "AI-generated image", prompt could be:
        "Is this a real photograph or an AI-generated image? Answer:"
        Then check if the generated text contains "real photograph" or "AI-generated image".
        
        This implementation will attempt a simplified version by checking if the 
        generated response to a generic question contains the prompt text.
        This is a heuristic and might need significant refinement for good performance.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of candidate texts (e.g., "real image", "generated image").

        Returns:
            Dict[str, float]: A dictionary mapping each prompt to a score (1.0 if present in output, 0.0 otherwise).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Generic question to elicit a descriptive response about the image content or nature
        # This question can be configured via self.config
        question = self.config.get("instructblip_question", "Describe the image.") 
        # For AI detection, a better question would be:
        # question = self.config.get("instructblip_question", "Is this image real or AI-generated?")

        results = {prompt: 0.0 for prompt in text_prompts}
        try:
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # `max_new_tokens` can be configured
                max_new_tokens = self.config.get("max_new_tokens", 50) 
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
            # print(f"InstructBLIP generated text for '{question}': {generated_text}")

            # Check for presence of each prompt in the generated text (case-insensitive)
            for prompt in text_prompts:
                if prompt.lower() in generated_text:
                    results[prompt] = 1.0
            
            # If multiple prompts match, this simple method doesn't differentiate well.
            # If no prompt matches, all scores will be 0.
            # A more sophisticated approach would be needed for nuanced classification.
            # For example, using specific prompts for positive/negative classes and analyzing the answer.
            # E.g. prompt1: "This is a real image.", prompt2: "This is an AI generated image."
            # Question: "Which of the following best describes the image: 'This is a real image.' or 'This is an AI generated image.'?"

        except Exception as e:
            print(f"Error during InstructBLIP prediction: {e}")
            # Return 0.0 for all prompts in case of error
        
        return results 