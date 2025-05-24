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

        # More direct question for AI/Nature detection
        question = self.config.get("instructblip_question", "Is this image real or AI-generated?") 

        results = {prompt: 0.0 for prompt in text_prompts}
        try:
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                max_new_tokens = self.config.get("max_new_tokens", 50) 
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            generated_text_full = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            question_lower = question.lower() # question is from self.config.get(...)
            generated_text_lower = generated_text_full.lower()

            actual_answer = ""
            if generated_text_lower.startswith(question_lower):
                actual_answer = generated_text_lower[len(question_lower):].strip()
            else:
                # If the question is not exactly at the beginning (e.g., model adds a prefix like "Answer: ")
                # we might need a more robust way to find the start of the actual answer.
                # For now, if question not prefix, assume the whole text is the answer or needs other forms of cleaning.
                # A simple fallback for now is to use the full text, but log it.
                print(f"Warning: InstructBLIP generated text does not start with the question. Full text: '{generated_text_full}'")
                actual_answer = generated_text_lower

            # print(f"InstructBLIP Question: '{question}'")
            # print(f"InstructBLIP Full Generated Text: '{generated_text_full}'")
            # print(f"InstructBLIP Extracted Answer: '{actual_answer}'")

            is_predicted_ai = False
            is_predicted_real = False

            target_real_phrase = "real photograph"
            target_ai_phrase = "ai generated"
            
            if target_ai_phrase in actual_answer:
                is_predicted_ai = True
            elif target_real_phrase in actual_answer:
                is_predicted_real = True
            
            # Assign scores to the prompts based on this direct prediction
            # This allows the main script's logic (best_prompt_text -> get_class_for_prompt) to still work.
            if is_predicted_ai:
                for prompt in text_prompts:
                    if self.prompt_strategy.get_class_for_prompt(prompt) == 1: # Assuming 1 is AI
                        results[prompt] = 1.0
                    else:
                        results[prompt] = 0.0
            elif is_predicted_real:
                for prompt in text_prompts:
                    if self.prompt_strategy.get_class_for_prompt(prompt) == 0: # Assuming 0 is Real
                        results[prompt] = 1.0
                    else:
                        results[prompt] = 0.0
            # If neither, all scores remain 0.0, and fallback in main script will apply.

        except Exception as e:
            print(f"Error during InstructBLIP prediction: {e}")
        
        return results

    def predict_batch(self, images: List[Image.Image], text_prompts: List[str]) -> List[Dict[str, float]]:
        """
        Performs prediction for a batch of images using InstructBLIP.

        Args:
            images (List[Image.Image]): A list of input images.
            text_prompts (List[str]): A list of text prompts for classification/evaluation.

        Returns:
            List[Dict[str, float]]: A list of dictionaries, where each dictionary 
                                     maps each prompt to its score for an image.
        """
        batch_results = []
        for image in images:
            image_scores = self.predict(image, text_prompts)
            batch_results.append(image_scores)
        return batch_results 