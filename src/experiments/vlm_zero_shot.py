import os
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json  # For saving detailed results
import contextlib  # For torch.autocast

# VLM imports
import open_clip
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BlipProcessor,
    BlipForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    AutoTokenizer,  # For Flamingo's LM
    ViTModel,  # Example for Flamingo's Vision Encoder
    AutoModelForCausalLM,
    GenerationConfig
)

# Define the prompts structure as provided by the user
USER_PROMPTS = [
    {"id": "1.1", "question": None, "options": {"real": "This is a real photograph.", "fake": "This is an AI-generated image."}},
    {"id": "1.2", "question": None, "options": {"real": "An authentic, unaltered photograph.", "fake": "An image fabricated by an artificial intelligence."}},
    {"id": "2.1", "question": None, "options": {"real": "A photograph taken by a physical camera.", "fake": "An image created by a computer algorithm."}},
    {"id": "3.1", "question": None, "options": {"real": "A natural image with authentic textures, lighting, and imperfections.", "fake": "A synthetic image, potentially exhibiting overly smooth areas, unusual patterns, or digital artifacts."}},
    {"id": "4.1", "question": None, "options": {"real": "This image documents a real event or subject.", "fake": "This image is an artistic or illustrative creation by an AI."}},
    {"id": "5.1", "question": None, "options": {"real": "Genuine.", "fake": "Artificial."}},
    {"id": "5.2", "question": None, "options": {"real": "Real.", "fake": "Fake."}},
    {"id": "6.1", "question": "Is this image a real photograph or an AI-generated image?", "options": {"real": "Real photograph.", "fake": "AI-generated image."}}
]

class VLMZeroShotExperiment:
    def __init__(self, data_path, num_samples=25, results_dir="results/vlm_zero_shot_detailed", specific_gpu_id=None):
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.device = torch.device(f"cuda:{specific_gpu_id}") if specific_gpu_id is not None and torch.cuda.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_config = USER_PROMPTS

        # Define model configurations
        self.model_configs = {
            "CLIP": {"type": "clip", "model_name": 'ViT-L-14', "pretrained": 'openai'},
            "BLIP": {"type": "generative", "model_name": "Salesforce/blip-image-captioning-large", "dtype": torch.float16},
            "InstructBLIP": {"type": "generative", "model_name": "Salesforce/instructblip-flan-t5-xl", "dtype": torch.float16},
            "LLaVA": {"type": "generative", "model_name": "llava-hf/llava-1.5-7b-hf", "dtype": torch.float16, "low_cpu_mem_usage": True},
        }
        self.model_names_to_run = ["CLIP", "BLIP", "InstructBLIP", "LLaVA"]
        self.loaded_model_instance = None
        self.loaded_processor_instance = None

    def load_model(self, model_name_to_load):
        if model_name_to_load not in self.model_configs:
            print(f"Configuration for model {model_name_to_load} not found. Skipping.")
            return False

        config = self.model_configs[model_name_to_load]
        model_type = config["type"]

        try:
            print(f"Initializing {model_name_to_load} ({config['model_name']})...")
            if model_type == "clip":
                model, _, processor = open_clip.create_model_and_transforms(config['model_name'], pretrained=config['pretrained'])
                self.loaded_model_instance = model.to(self.device)
                self.loaded_processor_instance = processor

            elif model_type == "generative":
                # choose processor and model classes
                if model_name_to_load == "BLIP":
                    processor_class = BlipProcessor
                    model_class = BlipForConditionalGeneration
                elif model_name_to_load == "InstructBLIP":
                    processor_class = InstructBlipProcessor
                    model_class = InstructBlipForConditionalGeneration
                else:  # LLaVA
                    processor_class = AutoProcessor
                    model_class = LlavaForConditionalGeneration

                # load processor & model
                self.loaded_processor_instance = processor_class.from_pretrained(config['model_name'])
                model_args = {"torch_dtype": config.get("dtype", torch.float32)}
                if config.get("low_cpu_mem_usage", False):
                    model_args["low_cpu_mem_usage"] = True
                self.loaded_model_instance = model_class.from_pretrained(config['model_name'], **model_args).to(self.device)

                # --- NEW: explicitly set pad_token_id and eos_token_id ---
                tokenizer = getattr(self.loaded_processor_instance, 'tokenizer', None)
                if tokenizer is not None:
                    pad_id = tokenizer.pad_token_id
                    eos_id = tokenizer.eos_token_id
                else:
                    pad_id = self.loaded_model_instance.config.pad_token_id
                    eos_id = self.loaded_model_instance.config.eos_token_id
                # fallback to decoder_start_token_id if needed
                if eos_id is None and hasattr(self.loaded_model_instance.config, 'decoder_start_token_id'):
                    eos_id = self.loaded_model_instance.config.decoder_start_token_id
                # set them on model config
                if pad_id is not None:
                    self.loaded_model_instance.config.pad_token_id = pad_id
                if eos_id is not None:
                    self.loaded_model_instance.config.eos_token_id = eos_id
                # -------------------------------------------------------

            else:
                print(f"Unknown model type {model_type} for {model_name_to_load}. Skipping.")
                return False

            print(f"{model_name_to_load} initialized successfully.")
            return True

        except Exception as e:
            print(f"Error initializing {model_name_to_load} ({config.get('model_name')}): {e}")
            self.loaded_model_instance = None
            self.loaded_processor_instance = None
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return False

    def unload_model(self):
        print(f"Unloading current model...")
        del self.loaded_model_instance, self.loaded_processor_instance
        self.loaded_model_instance = None
        self.loaded_processor_instance = None
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        print("Model unloaded and CUDA cache cleared.")

    def load_random_samples(self):
        real_paths = list((self.data_path / "0_real").glob("*.*"))
        fake_paths = list((self.data_path / "1_fake").glob("*.*"))
        random.seed(42)
        real = random.sample(real_paths, min(self.num_samples, len(real_paths)))
        fake = random.sample(fake_paths, min(self.num_samples, len(fake_paths)))
        return {str(p): 0 for p in real} | {str(p): 1 for p in fake}

    def _predict_clip_with_prompt(self, image_pil, processor, model, prompt_options):
        texts = [prompt_options['real'], prompt_options['fake']]
        text_tokens = open_clip.tokenize(texts).to(self.device)
        image_input = processor(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad(), torch.autocast(device_type=self.device.type) if self.device.type == 'cuda' else contextlib.nullcontext():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # score is probability of being fake
        return probs[0][1].item(), { "real_prob": probs[0][0].item(), "fake_prob": probs[0][1].item(), "raw_output": "CLIP similarity scores" }

    def _predict_generative_with_prompt(self, image_pil, processor, model, model_name, prompt_config):
        question = prompt_config['question']
        
        effective_question = question
        if model_name == "LLaVA":
            effective_question = question if question else "Is this image real or fake?"
            text_input = f"USER: <image>\n{effective_question} ASSISTANT:"
            # For LLaVA, processor might handle dtype internally or expect specific dtypes
            inputs = processor(text=text_input, images=image_pil, return_tensors="pt").to(self.device)
        elif model_name == "InstructBLIP" or model_name == "BLIP":
            effective_question = question if question else "Describe the image. Is it real or fake?"
            # For BLIP/InstructBLIP, ensure inputs are float16 if model is float16
            model_dtype = self.model_configs[model_name].get("dtype", torch.float32)
            inputs = processor(images=image_pil, text=effective_question, return_tensors="pt")
            inputs = {k: v.to(self.device).to(model_dtype if k=='pixel_values' else v.dtype) for k,v in inputs.items()}

        else: # Fallback, should not happen with current model list
            print(f"Warning: Model {model_name} not specifically handled for input generation. Using generic approach.")
            effective_question = question if question else "Is this image real or fake?"
            inputs = processor(images=image_pil, text=effective_question, return_tensors="pt").to(self.device)


        generation_config = GenerationConfig(max_new_tokens=70, num_beams=1) # Increased max_new_tokens slightly

        pad_token_id_to_set = None
        eos_token_id_to_set = None

        if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
            pad_token_id_to_set = processor.tokenizer.pad_token_id
            eos_token_id_to_set = processor.tokenizer.eos_token_id
        
        if pad_token_id_to_set is None and hasattr(model, 'config') and model.config.pad_token_id is not None:
            pad_token_id_to_set = model.config.pad_token_id
        if eos_token_id_to_set is None and hasattr(model, 'config') and model.config.eos_token_id is not None:
            eos_token_id_to_set = model.config.eos_token_id
            
        if pad_token_id_to_set is not None:
            generation_config.pad_token_id = pad_token_id_to_set
        # It's crucial for some models that eos_token_id is set.
        # If pad_token_id is None, but eos_token_id is available, some models use eos_token_id for padding.
        if eos_token_id_to_set is not None:
             generation_config.eos_token_id = eos_token_id_to_set
             if pad_token_id_to_set is None : # Common strategy
                 generation_config.pad_token_id = eos_token_id_to_set 
        else:
            print(f"Warning: Could not reliably determine eos_token_id for {model_name}. Using model's default or potentially no eos token if not in config.")
            # If eos_token_id is still None here, generation might not stop correctly or might raise error.
            # Some models like LLaVA might have it in model.config.eos_token_id, others might not set it explicitly if it's inherited.

        # Ensure inputs are on the correct device (redundant for some parts, but safe)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        autocast_context = torch.autocast(device_type=self.device.type, dtype=self.model_configs[model_name].get("dtype", None) if self.device.type == 'cuda' else None) if self.device.type == 'cuda' else contextlib.nullcontext()

        with torch.no_grad(), autocast_context:
            generated_ids = model.generate(**inputs, generation_config=generation_config)
        
        raw_generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if model_name == "LLaVA" and "ASSISTANT:" in raw_generated_text:
            raw_generated_text = raw_generated_text.split("ASSISTANT:")[-1].strip()
        elif (model_name == "BLIP" or model_name == "InstructBLIP") and effective_question in raw_generated_text:
            raw_generated_text = raw_generated_text.replace(effective_question, "").strip()


        generated_text_lower = raw_generated_text.lower().strip()
        score = 0.5 

        found_fake = any(keyword in generated_text_lower for keyword in ["fake", "artificial", "generated", "synthetic", "ai-generated", "computer-generated", "not real", "isn't real"])
        found_real = any(keyword in generated_text_lower for keyword in ["real", "authentic", "photograph", "genuine", "natural", "actual", "not fake", "isn't fake"])
        
        # More nuanced scoring
        if "not real" in generated_text_lower or "isn't real" in generated_text_lower :
             score = 1.0 # Strong indication of fake
        elif "not fake" in generated_text_lower or "isn't fake" in generated_text_lower:
             score = 0.0 # Strong indication of real
        elif found_fake and not found_real:
            score = 1.0
        elif found_real and not found_fake:
            score = 0.0
        elif found_fake and found_real: 
            # If both terms like "real" and "fake" are present, e.g. "This looks like a real image, but it could be fake."
            # we need to be careful. Let's check for more direct statements.
            if "is a real" in generated_text_lower and "is fake" not in generated_text_lower and "is a fake" not in generated_text_lower :
                score = 0.0
            elif ("is fake" in generated_text_lower or "is a fake" in generated_text_lower or "ai-generated" in generated_text_lower) and "is real" not in generated_text_lower :
                score = 1.0
            else: # Truly ambiguous
                score = 0.75 
        elif not generated_text_lower or generated_text_lower in ["unspecified", "unknown", "i cannot determine", "i can't tell"]:
            score = 0.5 
            
        return score, { "raw_output": raw_generated_text, "parsed_score": score }

    def run_experiment(self):
        image_paths_with_labels = self.load_random_samples()
        all_detailed_results = {str(img_path): {"true_label": label, "model_outputs": {}} 
                                for img_path, label in image_paths_with_labels.items()}

        for model_name_to_run in self.model_names_to_run:
            print(f"--- Running inference for model: {model_name_to_run} ---")
            if not self.load_model(model_name_to_run):
                print(f"Skipping model {model_name_to_run} due to load failure.")
                # Record error for this model for all images
                for img_path_str in image_paths_with_labels.keys():
                    all_detailed_results[img_path_str]["model_outputs"][model_name_to_run] = {}
                    for prompt_conf in self.prompts_config:
                        all_detailed_results[img_path_str]["model_outputs"][model_name_to_run][prompt_conf['id']] = \
                            { "score": 0.5, "raw_output": "Model load error", "error": True }
                continue # Move to the next model

            # Initialize results structure for this model if not already present
            for img_path_str in image_paths_with_labels.keys():
                if model_name_to_run not in all_detailed_results[img_path_str]["model_outputs"]:
                    all_detailed_results[img_path_str]["model_outputs"][model_name_to_run] = {}

            for img_path_str, true_label in tqdm(image_paths_with_labels.items(), desc=f"Processing images with {model_name_to_run}"):
                try:
                    image_pil = Image.open(img_path_str).convert('RGB')
                except Exception as e:
                    print(f"Error opening image {img_path_str}: {e}")
                    for prompt_conf in self.prompts_config:
                         all_detailed_results[img_path_str]["model_outputs"][model_name_to_run][prompt_conf['id']] = \
                            { "score": 0.5, "raw_output": f"Image load error: {e}", "error": True }
                    continue

                for prompt_conf in self.prompts_config:
                    prompt_id = prompt_conf['id']
                    score = 0.5 # Default score
                    details = {}
                    
                    try:
                        model_type = self.model_configs[model_name_to_run]["type"]
                        if model_type == "clip":
                            if prompt_conf["question"] is not None: # CLIP uses options, not questions
                                score, details = 0.5, {"error": "CLIP does not use 'question' field."}
                            else:
                                score, details = self._predict_clip_with_prompt(
                                    image_pil, self.loaded_processor_instance, self.loaded_model_instance, prompt_conf["options"]
                                )
                        elif model_type == "generative":
                            score, details = self._predict_generative_with_prompt(
                                image_pil, self.loaded_processor_instance, self.loaded_model_instance, model_name_to_run, prompt_conf
                            )
                        # Add Flamingo or other types here
                        # elif model_type == "flamingo":
                        #     score, details = self._predict_flamingo_with_prompt(...) 

                        all_detailed_results[img_path_str]["model_outputs"][model_name_to_run][prompt_id] = {
                            "score": score, 
                            "raw_output": details.get("raw_output", ""),
                            "real_prob_clip": details.get("real_prob"), # For CLIP
                            "fake_prob_clip": details.get("fake_prob"), # For CLIP
                            "parsed_score_gen": details.get("parsed_score"), # For generative
                            "error": details.get("error", False)
                        }
                    except Exception as e:
                        print(f"Error during prediction for {img_path_str} with {model_name_to_run} (prompt {prompt_id}): {e}")
                        all_detailed_results[img_path_str]["model_outputs"][model_name_to_run][prompt_id] = \
                            { "score": 0.5, "raw_output": f"Prediction error: {e}", "error": True }
            
            self.unload_model() # Unload model after processing all images for it

        # Save all detailed results
        detailed_results_path = self.results_dir / "all_detailed_results.json"
        with open(detailed_results_path, 'w') as f:
            json.dump(all_detailed_results, f, indent=4)
        print(f"All detailed results saved to {detailed_results_path}")

        # Post-process results to generate confusion matrices and reports per model & prompt
        self.post_process_results(all_detailed_results)
        return all_detailed_results

    def post_process_results(self, all_detailed_results):
        for model_name in self.model_names_to_run:
            # Check if model had loading errors and skip if so
            # Need to check if any image has valid output for this model
            has_valid_model_output = False
            for img_data in all_detailed_results.values():
                if model_name in img_data["model_outputs"] and img_data["model_outputs"][model_name]:
                    # Check if there's any prompt data that isn't a load error placeholder
                    for prompt_id, prompt_data in img_data["model_outputs"][model_name].items():
                        if not (prompt_data.get("raw_output") == "Model load error" and prompt_data.get("error")):
                            has_valid_model_output = True
                            break
                    if has_valid_model_output:
                        break
            
            if not has_valid_model_output:
                print(f"Skipping post-processing for {model_name} as no valid outputs were found (likely model load error).")
                continue


            for prompt_conf in self.prompts_config:
                prompt_id = prompt_conf['id']
                
                # Skip CLIP for prompts that have a "question"
                if self.model_configs[model_name]["type"] == "clip" and prompt_conf["question"] is not None:
                    continue
                
                y_true = []
                y_pred_scores = []
                has_data_for_prompt = False

                for img_path_str, data in all_detailed_results.items():
                    if model_name in data["model_outputs"] and prompt_id in data["model_outputs"][model_name]:
                        prompt_output = data["model_outputs"][model_name][prompt_id]
                        if not prompt_output.get("error", False): # Only consider non-error results
                            y_true.append(data["true_label"])
                            y_pred_scores.append(prompt_output["score"])
                            has_data_for_prompt = True
                
                if not has_data_for_prompt or not y_true: # Ensure there's data to process
                    print(f"No valid data for {model_name} with prompt {prompt_id}. Skipping matrix/report.")
                    continue

                # Convert scores to binary predictions (0 or 1) based on a 0.5 threshold
                # 0 = real, 1 = fake. Score > 0.5 means fake.
                y_pred_binary = [1 if score > 0.5 else 0 for score in y_pred_scores]

                # Create a subdirectory for this model's results if it doesn't exist
                model_results_dir = self.results_dir / model_name
                model_results_dir.mkdir(parents=True, exist_ok=True)

                try:
                    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]) # Explicitly define labels
                    report = classification_report(y_true, y_pred_binary, target_names=['real (0)', 'fake (1)'], output_dict=True, zero_division=0)
                    
                    # Save classification report
                    report_path = model_results_dir / f"report_prompt_{prompt_id.replace('.', '_')}.json"
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=4)
                    print(f"Classification report for {model_name} (Prompt {prompt_id}) saved to {report_path}")

                    # Plot and save confusion matrix
                    self.plot_confusion_matrix(cm, y_true, y_pred_binary, model_name, prompt_id, model_results_dir)

                except Exception as e:
                    print(f"Error during post-processing for {model_name} (Prompt {prompt_id}): {e}")


    def plot_confusion_matrix(self, cm, y_true, y_pred_binary, model_name, prompt_id, model_results_dir):
        # Fallback if cm is None or not correctly generated (e.g. no samples)
        if cm is None or cm.shape != (2,2):
             # Create a dummy CM if y_true or y_pred_binary is empty or leads to invalid CM
            print(f"Not enough data to generate a valid confusion matrix for {model_name}, prompt {prompt_id}. Plotting placeholder.")
            cm = np.zeros((2,2), dtype=int) # Placeholder
            # Try to populate with counts if possible, otherwise it's all zeros
            if y_true and y_pred_binary:
                # This situation (invalid CM shape but data exists) should ideally not happen if labels=[0,1] is used
                # and there's mixed data, but as a safe fallback:
                for yt, yp in zip(y_true, y_pred_binary):
                    if yt in [0,1] and yp in [0,1]: # Ensure valid labels
                         cm[yt, yp] += 1


        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real (0)', 'Fake (1)'], yticklabels=['Real (0)', 'Fake (1)'])
        plt.title(f'Confusion Matrix: {model_name} - Prompt {prompt_id}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Sanitize prompt_id for filename
        safe_prompt_id = prompt_id.replace('.', '_')
        plot_path = model_results_dir / f"cm_prompt_{safe_prompt_id}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrix for {model_name} (Prompt {prompt_id}) saved to {plot_path}")

def main(gpu_id_to_use=None):
    # data_path = "/path/to/your/Chameleon/test" # Replace with the actual path
    # For testing, let's assume a structure within the project or a readily available small dataset path
    # Example: create dummy folders data/Chameleon/test/0_real and data/Chameleon/test/1_fake
    # For actual runs, this path needs to be correct.
    
    # Using the provided path directly
    data_path = "/ssd6/GAIC_Dataset/Chameleon/test/" 
    
    # num_samples = 2 # Small number for testing
    num_samples = 25 # As per original request

    # Create dummy image files for testing if the path doesn't exist or is empty
    # This is useful for ensuring the script runs end-to-end without the actual large dataset.
    # In a real scenario, you would remove this dummy data creation.
    test_setup_path = Path(data_path)
    if not test_setup_path.exists() or not any((test_setup_path / "0_real").iterdir()) :
        print(f"Warning: Dataset path {data_path} not found or '0_real' is empty. Creating dummy data for testing.")
        dummy_data_path = Path("dummy_chameleon_test")
        dummy_data_path.mkdir(parents=True, exist_ok=True)
        (dummy_data_path / "0_real").mkdir(exist_ok=True)
        (dummy_data_path / "1_fake").mkdir(exist_ok=True)
        try:
            for i in range(5): # Create 5 dummy images for each class
                Image.new('RGB', (60, 30), color = 'red').save(dummy_data_path / "0_real" / f"real_{i}.png")
                Image.new('RGB', (60, 30), color = 'blue').save(dummy_data_path / "1_fake" / f"fake_{i}.png")
            data_path = str(dummy_data_path) # Use dummy data for this test run
            print(f"Using dummy data from: {data_path}")
        except ImportError:
            print("Pillow (PIL) is not installed. Cannot create dummy images. Please install it if you want to run with dummy data.")
        except Exception as e:
            print(f"Could not create dummy images: {e}")


    results_dir = "results/vlm_zero_shot_chameleon_detailed"
    
    experiment = VLMZeroShotExperiment(data_path, num_samples=num_samples, results_dir=results_dir, specific_gpu_id=gpu_id_to_use)
    experiment.run_experiment()

if __name__ == "__main__":
    # Example: To run on GPU 1, you would call: python vlm_zero_shot.py --gpu 1
    # For now, let's make it runnable without command-line args, using auto-detection or a default.
    # You can pass a GPU ID here if needed, e.g., main(gpu_id_to_use=0)
    
    # Simple argument parsing for GPU ID
    import argparse
    parser = argparse.ArgumentParser(description="Run VLM Zero-Shot Experiment.")
    parser.add_argument("--gpu", type=int, default=None, help="Specify the GPU ID to use (e.g., 0, 1). Default is None (auto-select or CPU).")
    args = parser.parse_args()

    main(gpu_id_to_use=args.gpu) 