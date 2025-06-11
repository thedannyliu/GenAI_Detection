import argparse
import torch
from pathlib import Path
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import PeftModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROMPT = "Is this image AI-generated? Answer 'yes' or 'no'."


class PromptTuningInference:
    """Inference class for prompt-tuned InstructBLIP model."""
    
    def __init__(self, model_path: str, base_model_name: str = "Salesforce/instructblip-vicuna-7b"):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the saved prompt-tuned model
            base_model_name: Name of the base InstructBLIP model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load processor
        logger.info("Loading processor...")
        self.processor = InstructBlipProcessor.from_pretrained(base_model_name)
        
        # Setup tokenizer padding
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id
        
        # Load base model
        logger.info("Loading base model...")
        self.base_model = InstructBlipForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load prompt-tuned model
        logger.info(f"Loading prompt-tuned model from {model_path}...")
        self.model = PeftModel.from_pretrained(
            self.base_model.language_model,
            model_path,
            is_trainable=False
        )
        
        # Replace the language model in the base model
        self.base_model.language_model = self.model
        self.base_model.eval()
        
        logger.info("Model loaded successfully!")
    
    def predict_single_image(self, image_path: str) -> dict:
        """
        Predict whether a single image is AI-generated.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                text=PROMPT,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    min_length=1,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.1,
                    repetition_penalty=1.1
                )
            
            # Decode response
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            answer = self._extract_answer(generated_text)
            confidence = self._calculate_confidence(outputs[0])
            
            return {
                "image_path": image_path,
                "prediction": answer,
                "confidence": confidence,
                "raw_output": generated_text,
                "is_ai_generated": answer.lower() == "yes"
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                "image_path": image_path,
                "prediction": "error",
                "confidence": 0.0,
                "raw_output": str(e),
                "is_ai_generated": None
            }
    
    def predict_batch(self, image_paths: list) -> list:
        """
        Predict for multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path)
            results.append(result)
            logger.info(f"Processed {image_path}: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return results
    
    def _extract_answer(self, generated_text: str) -> str:
        """Extract yes/no answer from generated text."""
        text_lower = generated_text.lower()
        
        # Look for explicit yes/no answers
        if "yes" in text_lower and "no" not in text_lower:
            return "yes"
        elif "no" in text_lower and "yes" not in text_lower:
            return "no"
        elif "yes" in text_lower and "no" in text_lower:
            # If both appear, take the last one
            yes_pos = text_lower.rfind("yes")
            no_pos = text_lower.rfind("no")
            return "yes" if yes_pos > no_pos else "no"
        else:
            return "uncertain"
    
    def _calculate_confidence(self, output_ids: torch.Tensor) -> float:
        """Calculate a simple confidence score based on output length and tokens."""
        # This is a simplified confidence calculation
        # In practice, you might want to use the actual logits for better confidence estimation
        return min(1.0, len(output_ids) / 20.0)


def main():
    parser = argparse.ArgumentParser(description="Inference with prompt-tuned InstructBLIP model")
    parser.add_argument("--model_path", required=True, help="Path to the prompt-tuned model")
    parser.add_argument("--image_path", help="Path to a single image")
    parser.add_argument("--image_dir", help="Path to directory containing images")
    parser.add_argument("--base_model", default="Salesforce/instructblip-vicuna-7b", 
                       help="Base model name")
    parser.add_argument("--output_file", help="Path to save prediction results")
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        raise ValueError("Either --image_path or --image_dir must be provided")
    
    # Initialize inference pipeline
    inference = PromptTuningInference(args.model_path, args.base_model)
    
    # Collect image paths
    image_paths = []
    if args.image_path:
        image_paths.append(args.image_path)
    
    if args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            image_paths.extend(list(image_dir.glob(f"*{ext}")))
            image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    if not image_paths:
        raise ValueError("No images found to process")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Run inference
    results = inference.predict_batch([str(p) for p in image_paths])
    
    # Print results
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    ai_count = 0
    real_count = 0
    
    for result in results:
        status = "✓" if result["prediction"] != "error" else "✗"
        print(f"{status} {Path(result['image_path']).name:30} | "
              f"{result['prediction']:10} | "
              f"Confidence: {result['confidence']:.3f}")
        
        if result["is_ai_generated"] is True:
            ai_count += 1
        elif result["is_ai_generated"] is False:
            real_count += 1
    
    print("="*80)
    print(f"SUMMARY: {ai_count} AI-generated, {real_count} Real images")
    
    # Save results if requested
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()