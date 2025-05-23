from PIL import Image
from typing import List, Any, Dict
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from .base_vlm import BaseVLM

class LlavaModelWrapper(BaseVLM):
    """
    Wrapper for the LLaVA model from Hugging Face.
    Uses llava-hf/llava-1.5-7b-hf as a potential default.
    """
    def _load_model(self):
        model_id = self.config.get("model_id", "llava-hf/llava-1.5-7b-hf")
        # LlavaNextProcessor and LlavaNextForConditionalGeneration are for Llama-3 based LLaVA
        # For older LLaVA 1.5, it might be LlamaProcessor and LlamaForCausalLM with custom handling
        # or specific LLaVA classes if available directly in transformers for that version.
        # The llava-hf repositories usually specify the correct processor and model classes.
        # Assuming llava-1.5-7b-hf uses LlavaNext* classes based on recent trends,
        # but this might need adjustment based on the exact model version from HF.
        try:
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
        except Exception as e:
            print(f"Error loading LLaVA model {model_id}: {e}")
            print("If using LLaVA 1.5 (non-Next/Llama-3 based), you might need different Hugging Face classes e.g. AutoProcessor and AutoModelForCausalLM with custom prompt templating for LLaVA.")
            raise

    def predict(self, image: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        """
        Performs zero-shot prediction using LLaVA.
        Similar to InstructBLIP, this involves prompting the model with the image and a question,
        then evaluating its generated response against the provided text_prompts.

        Args:
            image (Image.Image): The input image.
            text_prompts (List[str]): A list of candidate texts (e.g., "real image", "generated image").
                                      The LLaVA model will be asked a question, and its response
                                      will be checked for the presence of these prompt texts.

        Returns:
            Dict[str, float]: A dictionary mapping each prompt text to a score (1.0 if present, 0.0 otherwise).
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Default question for LLaVA, can be configured.
        # For LLaVA, the prompt needs to be in a specific format, often including <image> tokens.
        # The processor usually handles the correct formatting.
        # Example for LLaVA 1.5: "USER: <image>\nWhat is this image about? ASSISTANT:"
        # We want to ask about whether the image is real or AI-generated.
        
        # A general question, the answer to which might reveal if it's AI or real.
        # Or a more direct question:
        llava_question = self.config.get("llava_question", "Is this image a real photograph or an AI-generated image? Provide a concise answer.")
        
        # The LLaVA processor typically requires the prompt to follow a specific template
        # where an <image> token is replaced by image embeddings.
        # For llava-hf/llava-1.5-7b-hf, the prompt format is usually like:
        # "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{question} ASSISTANT:"
        # The new LlavaNextProcessor might handle this differently or expect a simpler format.
        # Let's assume the processor.preprocess or __call__ handles the multimodal aspect.
        # For LlavaNextProcessor with LlavaNextForConditionalGeneration:
        prompt_template = self.config.get("llava_prompt_template", "USER: <image>\n{} ASSISTANT:")
        full_prompt = prompt_template.format(llava_question)


        results = {prompt: 0.0 for prompt in text_prompts}
        try:
            inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                max_new_tokens = self.config.get("max_new_tokens", 70) # Increased for LLaVA
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            
            # The generated text includes the prompt, so we need to decode carefully
            # or use processor.batch_decode, which often handles this.
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip().lower()
            
            # Extract only the assistant's response part
            assistant_response_marker = "assistant:"
            response_start_index = generated_text.rfind(assistant_response_marker)
            if response_start_index != -1:
                assistant_answer = generated_text[response_start_index + len(assistant_response_marker):].strip()
            else:
                # If marker not found, use the whole generated text after prompt (less reliable)
                # This part needs to be robust based on exact model output format
                # For now, if "assistant:" isn't in the decoded output (e.g. if skip_special_tokens removed it along with prompt),
                # we assume the decoded output is primarily the answer.
                assistant_answer = generated_text 


            # print(f"LLaVA prompt: {full_prompt}")
            # print(f"LLaVA generated text (raw): {generated_text}")
            # print(f"LLaVA assistant answer: {assistant_answer}")

            for prompt_text in text_prompts:
                if prompt_text.lower() in assistant_answer:
                    results[prompt_text] = 1.0
        
        except Exception as e:
            print(f"Error during LLaVA prediction: {e}")
            # Return 0.0 for all prompts in case of error
            
        return results 