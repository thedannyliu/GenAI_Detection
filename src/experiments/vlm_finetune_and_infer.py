import os
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import contextlib
from torch.utils.data import Dataset

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    AutoProcessor, # For loading processor of fine-tuned model
    AutoModelForVision2Seq, # For loading fine-tuned vision2seq model
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

# Define the prompts structure (can be reused or adapted)
# For fine-tuning, we'll typically use a fixed input format.
# For inference, we can test various prompts as before.
USER_PROMPTS = [
    {
        "id": "ft_default", # Default prompt used during fine-tuning
        "question": "Is this image real or fake?",
        "options": {"real": "This is a real photograph.", "fake": "This is an AI-generated image."}
    },
    { # Example of another prompt for testing inference
        "id": "alt_1",
        "question": "Provide a verdict: authentic photograph or AI fabrication?",
        "options": {"real": "Authentic photograph.", "fake": "AI fabrication."}
    }
    # Add more prompts for inference testing if desired
]

# --- 1. Dataset Class for Fine-tuning ---
class FinetuneImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor, prompt, target_texts, max_length=128):
        """
        Args:
            image_paths (list): List of paths to images.
            labels (list): List of labels (0 for real, 1 for fake).
            processor (InstructBlipProcessor): The processor for the model.
            prompt (str): The question/prompt to be prepended to the image.
            target_texts (dict): Dict with keys 0 and 1, mapping to target sentences.
                                 e.g., {0: "This is a real photograph.", 1: "This is an AI-generated image."}
            max_length (int): Max length for tokenization.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.prompt = prompt # The fixed question for fine-tuning
        self.target_texts = target_texts
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}. Skipping.")
            # Return a dummy item or handle appropriately
            # For simplicity, we might raise an error or return None and filter in collator/loader
            # Here, let's assume valid images for now or pre-filter.
            # A more robust way would be to return a flag and handle it.
            # For now, we'll let it potentially fail image processing if image is bad.
            pass

        # Target text based on label
        target_text = self.target_texts[label]

        # Processing for InstructBLIP: text is the prompt, image is the image
        # The model will learn to generate `target_text` based on the image and prompt.
        # For training, processor needs to prepare `labels` from `target_text`.
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        
        # Tokenize the target text to create decoder_input_ids / labels for training
        labels_tokenized = self.processor.tokenizer(
            text=target_text,
            padding="max_length", # Pad to max_length
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # For vision-to-text models in Hugging Face, the 'labels' are the tokenized target sequences.
        # The model internally shifts these to create decoder_input_ids.
        inputs["labels"] = labels_tokenized.input_ids.squeeze() # Remove batch dim if present

        # Squeeze all tensor inputs to remove the batch dimension added by the processor
        # as DataLoader will add it back.
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 1 else v for k, v in inputs.items()}
        
        return inputs

# --- 2. Data Preparation Function ---
def prepare_finetuning_and_inference_data(
    data_path: Path, 
    num_total_finetune_samples: int = 1000, 
    train_val_split_ratio: float = 0.2, # Anteil für Validation-Set
    seed: int = 42
):
    """
    Prepares data for fine-tuning and inference.
    Selects `num_total_finetune_samples` for fine-tuning, splits them into train/val.
    The remaining images are candidates for the inference set.
    """
    random.seed(seed)
    np.random.seed(seed)

    real_images_paths = sorted(list((data_path / "0_real").glob("*.jpg")) + 
                               list((data_path / "0_real").glob("*.png")) + 
                               list((data_path / "0_real").glob("*.jpeg")))
    fake_images_paths = sorted(list((data_path / "1_fake").glob("*.jpg")) + 
                               list((data_path / "1_fake").glob("*.png")) + 
                               list((data_path / "1_fake").glob("*.jpeg")))

    # Ensure balanced selection for fine-tuning
    num_real_ft = num_total_finetune_samples // 2
    num_fake_ft = num_total_finetune_samples - num_real_ft

    if len(real_images_paths) < num_real_ft:
        print(f"Warning: Not enough real images for the requested {num_real_ft}. Using all {len(real_images_paths)} real images.")
        num_real_ft = len(real_images_paths)
    if len(fake_images_paths) < num_fake_ft:
        print(f"Warning: Not enough fake images for the requested {num_fake_ft}. Using all {len(fake_images_paths)} fake images.")
        num_fake_ft = len(fake_images_paths)
    
    actual_total_finetune_samples = num_real_ft + num_fake_ft
    if actual_total_finetune_samples == 0:
        raise ValueError("No images available for fine-tuning after checking availability.")
    print(f"Actual total fine-tuning samples: {actual_total_finetune_samples} ({num_real_ft} real, {num_fake_ft} fake)")

    selected_real_for_ft = random.sample(real_images_paths, num_real_ft)
    selected_fake_for_ft = random.sample(fake_images_paths, num_fake_ft)

    finetune_image_paths = selected_real_for_ft + selected_fake_for_ft
    finetune_labels = ([0] * num_real_ft) + ([1] * num_fake_ft) # 0 for real, 1 for fake

    # Shuffle them together
    temp = list(zip(finetune_image_paths, finetune_labels))
    random.shuffle(temp)
    finetune_image_paths, finetune_labels = zip(*temp)
    finetune_image_paths, finetune_labels = list(finetune_image_paths), list(finetune_labels)


    # Split fine-tuning data into training and validation sets
    # Ensure there are enough samples for splitting, especially if actual_total_finetune_samples is small
    if len(finetune_image_paths) < 2 or (len(finetune_image_paths) * train_val_split_ratio < 1) or (len(finetune_image_paths) * (1-train_val_split_ratio) < 1):
        print(f"Warning: Not enough samples ({len(finetune_image_paths)}) to perform a meaningful train/val split with ratio {train_val_split_ratio}.")
        # Fallback: use all for training, or handle as error, or adjust split
        if len(finetune_image_paths) >= 1:
            print("Using all available fine-tuning samples for training, and duplicating for validation if needed for API.")
            train_paths, val_paths, train_labels, val_labels = finetune_image_paths, finetune_image_paths, finetune_labels, finetune_labels
        else:
            # This case should be caught by actual_total_finetune_samples == 0 check earlier
            train_paths, val_paths, train_labels, val_labels = [], [], [], [] 
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            finetune_image_paths, 
            finetune_labels, 
            test_size=train_val_split_ratio, 
            random_state=seed,
            stratify=finetune_labels if len(set(finetune_labels)) > 1 else None # Stratify only if multiple classes
        )

    print(f"Total images for fine-tuning: {len(finetune_image_paths)}")
    print(f"Training samples: {len(train_paths)} (Real: {train_labels.count(0)}, Fake: {train_labels.count(1)})")
    print(f"Validation samples: {len(val_paths)} (Real: {val_labels.count(0)}, Fake: {val_labels.count(1)})")

    # Prepare inference data: images not used in fine-tuning
    finetune_paths_set = set(map(str, finetune_image_paths)) # Ensure paths are strings for set operations
    inference_candidate_real_paths = [p for p in real_images_paths if str(p) not in finetune_paths_set]
    inference_candidate_fake_paths = [p for p in fake_images_paths if str(p) not in finetune_paths_set]

    print(f"Remaining real images for inference: {len(inference_candidate_real_paths)}")
    print(f"Remaining fake images for inference: {len(inference_candidate_fake_paths)}")
    
    return {
        "train_paths": train_paths, "train_labels": train_labels,
        "val_paths": val_paths, "val_labels": val_labels,
        "inference_candidate_real": inference_candidate_real_paths,
        "inference_candidate_fake": inference_candidate_fake_paths,
        "all_finetune_paths": finetune_paths_set
    }

# --- 3. Fine-tuning Function ---
def finetune_model(
    model_name: str,
    train_dataset: FinetuneImageDataset,
    val_dataset: FinetuneImageDataset,
    output_dir: Path,
    logging_dir: Path,
    epochs: int = 3,
    batch_size: int = 4, # Adjust based on GPU memory
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f"\n--- Starting Fine-tuning for model: {model_name} ---")
    print(f"Using device: {device}")

    # Load pre-trained model and processor
    # The processor should be loaded first and passed to the dataset
    # For fine-tuning, we assume the processor used for dataset creation is the correct one.
    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=str(logging_dir),
        logging_steps=10, # Log every 10 steps
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save a checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model (based on loss) at the end of training
        metric_for_best_model="loss", # Can also be an eval metric if compute_metrics is set
        greater_is_better=False, # For loss, lower is better
        fp16=torch.cuda.is_available(), # Enable mixed precision training if CUDA is available
        #gradient_accumulation_steps=2, # Optional: if batch size is too small
        # dataloader_num_workers=2, # Optional: for faster data loading
        report_to="tensorboard", # or "wandb", "none"
        remove_unused_columns=False, # Important for custom datasets where __getitem__ returns dict
    )

    # Initialize Trainer
    # Note: compute_metrics can be added for more detailed evaluation during training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=train_dataset.processor.tokenizer # Not strictly needed if processor passed to dataset and handles tokenization for labels
    )

    # Start fine-tuning
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and processor
    print("Training complete. Saving model...")
    final_model_path = output_dir / "final_model"
    trainer.save_model(str(final_model_path))
    # The processor used for the dataset should also be saved if it's not standard for the base model_name
    # However, InstructBlipProcessor is standard. For safety, one might save it too.
    train_dataset.processor.save_pretrained(str(final_model_path))
    print(f"Fine-tuned model and processor saved to {final_model_path}")
    
    return str(final_model_path)

# --- 3.5 Fine-tuning Function with LoRA ---
def finetune_model_with_lora(
    base_model_name: str,
    train_dataset: FinetuneImageDataset,
    val_dataset: FinetuneImageDataset,
    output_dir_checkpoints: Path, # Where Trainer saves its checkpoints
    final_lora_adapter_dir: Path, # Where final LoRA adapter is saved
    logging_dir: Path,
    lora_config_params: dict, # e.g., {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q", "v"]}
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4, # Often slightly higher for LoRA
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f"\n--- Starting LoRA Fine-tuning for model: {base_model_name} ---")
    print(f"Using device: {device}")

    # Load pre-trained base model
    model = InstructBlipForConditionalGeneration.from_pretrained(base_model_name)
    # model.to(device) # PEFT model will be moved to device later if needed or handled by Trainer

    # Define LoRA Config
    # For InstructBLIP (which uses Flan-T5 as LLM), common task_type is SEQ_2_SEQ_LM
    # If the LLM part is decoder-only, it would be CAUSAL_LM
    # InstructBLIP is a VisionEncoderDecoderModel, where QFormer and LLM act as decoder part essentially.
    # The Flan-T5 part is indeed Seq2Seq. VISION_ENCODER_DECODER is also a PEFT TaskType.
    # Given that we are modifying the LLM part (Flan-T5), SEQ_2_SEQ_LM is appropriate for the LoraConfig's task_type.
    lora_config = LoraConfig(
        r=lora_config_params.get("r", 16),
        lora_alpha=lora_config_params.get("lora_alpha", 32),
        target_modules=lora_config_params.get("target_modules", ["q", "v"]),
        lora_dropout=lora_config_params.get("lora_dropout", 0.05),
        bias="none", # Usually set to "none" for LoRA, can be "all" or "lora_only"
        task_type=TaskType.SEQ_2_SEQ_LM # For Flan-T5 used in InstructBLIP
        # For some models, you might need to manually identify and list all linear layers to target.
        # Example for a more generic application if PEFT cannot infer easily:
        # modules_to_save = ["embed_tokens", "lm_head"] # if you want to train these too (not typical for LoRA)
    )

    # Wrap the base model with PEFT model for LoRA training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Shows how many parameters are actually being trained
    model.to(device) # Now move the PEFT model to device

    # Define TrainingArguments (similar to full fine-tuning but output_dir is for checkpoints)
    training_args = TrainingArguments(
        output_dir=str(output_dir_checkpoints),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=str(logging_dir),
        logging_steps=20, # Log more frequently for LoRA potentially
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, # The PEFT model
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting LoRA training...")
    trainer.train()

    print("LoRA Training complete. Saving LoRA adapter...")
    # Ensure the final LoRA adapter directory exists
    final_lora_adapter_dir.mkdir(parents=True, exist_ok=True)
    
    # For PEFT models, save_pretrained saves only the adapter weights and config.
    model.save_pretrained(str(final_lora_adapter_dir))
    
    # Save the processor used (important for inference consistency)
    train_dataset.processor.save_pretrained(str(final_lora_adapter_dir))
    print(f"LoRA adapter and processor saved to {final_lora_adapter_dir}")
    
    return str(final_lora_adapter_dir)

# --- 4. Inference Function on Fine-tuned Model (Full FT) ---
def run_inference_on_finetuned_model(
    finetuned_model_path: str, # Path to the directory containing the fine-tuned model and processor
    inference_image_paths: list,
    inference_labels: list,
    prompts_config_for_inference: list, # List of prompt configs (like USER_PROMPTS)
    results_dir: Path,
    device: torch.device,
    batch_size: int = 8, # Can be different from training batch size
    max_new_tokens_inf: int = 70 # Max new tokens for generation during inference
):
    print(f"\n--- Running Inference on Fine-tuned Model from: {finetuned_model_path} ---")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the fine-tuned model and its processor
    try:
        # It's often better to load the processor specifically saved with the model
        processor = AutoProcessor.from_pretrained(finetuned_model_path)
        model = AutoModelForVision2Seq.from_pretrained(finetuned_model_path)
        model.to(device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        print(f"Error loading fine-tuned model or processor from {finetuned_model_path}: {e}")
        return None

    all_detailed_results = {}
    num_images = len(inference_image_paths)

    for i in tqdm(range(0, num_images, batch_size), desc="Processing images for inference"):
        batch_image_paths = inference_image_paths[i:i+batch_size]
        batch_labels = inference_labels[i:i+batch_size]
        batch_pil_images = []
        valid_indices_in_batch = [] # To track images that loaded successfully

        for idx, img_path_str in enumerate(batch_image_paths):
            try:
                img_pil = Image.open(img_path_str).convert('RGB')
                batch_pil_images.append(img_pil)
                valid_indices_in_batch.append(idx)
            except Exception as e:
                print(f"Error opening image {img_path_str}: {e}. Skipping this image.")
                # Record error for this specific image path immediately
                path_key = str(img_path_str) # Ensure consistency in key format
                all_detailed_results[path_key] = {
                    "true_label": batch_labels[idx] if idx < len(batch_labels) else -1, # Original label if available
                    "model_outputs": {prompt_conf["id"]: {"score": 0.5, "raw_output": f"Image load error: {e}", "error": True}
                                      for prompt_conf in prompts_config_for_inference}
                }
        
        if not batch_pil_images: # If all images in batch failed to load
            continue

        original_batch_paths_valid = [batch_image_paths[j] for j in valid_indices_in_batch]
        original_batch_labels_valid = [batch_labels[j] for j in valid_indices_in_batch]

        for prompt_conf in prompts_config_for_inference:
            prompt_id = prompt_conf["id"]
            question = prompt_conf["question"] # The question for this specific prompt config

            # Prepare inputs for the batch using the prompt's question
            # For InstructBLIP, the text is the question/instruction.
            inputs = processor(images=batch_pil_images, text=question, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generation_config = GenerationConfig(max_new_tokens=max_new_tokens_inf, num_beams=1)
            # Try to set pad_token_id and eos_token_id from processor or model config for generation
            if processor.tokenizer.pad_token_id is not None:
                generation_config.pad_token_id = processor.tokenizer.pad_token_id
            if processor.tokenizer.eos_token_id is not None:
                generation_config.eos_token_id = processor.tokenizer.eos_token_id
            if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
                 generation_config.pad_token_id = generation_config.eos_token_id # Common fallback

            with torch.no_grad(), torch.autocast(device_type=device.type) if device.type == 'cuda' else contextlib.nullcontext():
                generated_ids = model.generate(**inputs, generation_config=generation_config)
            
            raw_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for batch_idx, raw_text in enumerate(raw_generated_texts):
                img_path_str_current = original_batch_paths_valid[batch_idx]
                true_label_current = original_batch_labels_valid[batch_idx]
                path_key = str(img_path_str_current)

                if path_key not in all_detailed_results:
                    all_detailed_results[path_key] = {"true_label": true_label_current, "model_outputs": {}}
                
                # Clean up generated text if the prompt question is echoed
                cleaned_text = raw_text
                if question and question in raw_text:
                    cleaned_text = raw_text.replace(question, "").strip()
                
                generated_text_lower = cleaned_text.lower().strip()
                score = 0.5 # Default score

                # Keyword-based scoring (similar to zero-shot, can be refined)
                # This part should ideally use the specific phrases the model was fine-tuned on for ft_default prompt
                # For other prompts, general keyword matching is a fallback.
                target_real_text_lower = prompt_conf["options"]["real"].lower()
                target_fake_text_lower = prompt_conf["options"]["fake"].lower()

                # More robust check for keywords
                found_fake_keywords = ["fake", "artificial", "generated", "synthetic", "ai-generated", "computer-generated", "not real", "isn't real"]
                found_real_keywords = ["real", "authentic", "photograph", "genuine", "natural", "actual", "not fake", "isn't fake"]

                has_fake = any(keyword in generated_text_lower for keyword in found_fake_keywords) or (target_fake_text_lower in generated_text_lower)
                has_real = any(keyword in generated_text_lower for keyword in found_real_keywords) or (target_real_text_lower in generated_text_lower)
                
                if "not real" in generated_text_lower or "isn't real" in generated_text_lower:
                    score = 1.0
                elif "not fake" in generated_text_lower or "isn't fake" in generated_text_lower:
                    score = 0.0
                elif has_fake and not has_real:
                    score = 1.0
                elif has_real and not has_fake:
                    score = 0.0
                elif has_fake and has_real:
                    if target_real_text_lower in generated_text_lower and target_fake_text_lower not in generated_text_lower:
                         score = 0.0
                    elif target_fake_text_lower in generated_text_lower and target_real_text_lower not in generated_text_lower:
                         score = 1.0
                    else:
                        score = 0.75 # Ambiguous
                elif not generated_text_lower or generated_text_lower in ["unspecified", "unknown", "i cannot determine", "i can't tell"]:
                    score = 0.5

                all_detailed_results[path_key]["model_outputs"][prompt_id] = {
                    "score": score,
                    "raw_output": raw_text, # Store the original raw output
                    "cleaned_output": cleaned_text,
                    "parsed_score": score, 
                    "error": False
                }
    
    # Save all detailed results for inference
    detailed_results_path = results_dir / "inference_detailed_results.json"
    with open(detailed_results_path, 'w') as f:
        json.dump(all_detailed_results, f, indent=4)
    print(f"Inference detailed results saved to {detailed_results_path}")

    # Post-process results (similar to zero-shot script's post_process_results)
    post_process_inference_results(all_detailed_results, prompts_config_for_inference, results_dir, model_name_for_report="fine_tuned_instructblip")
    return all_detailed_results

# --- 4.5 Inference Function on LoRA Fine-tuned Model ---
def run_inference_on_lora_model(
    base_model_name: str, # Name of the base pre-trained model
    lora_adapter_path: str, # Path to the directory containing the LoRA adapter and its processor
    inference_image_paths: list,
    inference_labels: list,
    prompts_config_for_inference: list,
    results_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens_inf: int = 70
):
    print(f"\n--- Running Inference on LoRA-adapted Model ({base_model_name} + adapter from {lora_adapter_path}) ---")
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load the base pre-trained model
        base_model = InstructBlipForConditionalGeneration.from_pretrained(base_model_name)
        
        # 2. Load the LoRA adapter and apply it to the base model
        # The PeftModel class will automatically handle loading the adapter into the base model.
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model.to(device)
        model.eval() # Set model to evaluation mode

        # 3. Load the processor saved with the LoRA adapter
        processor = AutoProcessor.from_pretrained(lora_adapter_path)

    except Exception as e:
        print(f"Error loading base model, LoRA adapter, or processor: {e}")
        print(f"Base model: {base_model_name}, Adapter: {lora_adapter_path}")
        return None

    # The rest of the inference logic is very similar to run_inference_on_finetuned_model
    all_detailed_results = {}
    num_images = len(inference_image_paths)

    for i in tqdm(range(0, num_images, batch_size), desc="Processing images for LoRA inference"):
        batch_image_paths = inference_image_paths[i:i+batch_size]
        batch_labels = inference_labels[i:i+batch_size]
        batch_pil_images = []
        valid_indices_in_batch = []

        for idx, img_path_str in enumerate(batch_image_paths):
            try:
                img_pil = Image.open(img_path_str).convert('RGB')
                batch_pil_images.append(img_pil)
                valid_indices_in_batch.append(idx)
            except Exception as e:
                print(f"Error opening image {img_path_str}: {e}. Skipping.")
                path_key = str(img_path_str)
                all_detailed_results[path_key] = {
                    "true_label": batch_labels[idx] if idx < len(batch_labels) else -1,
                    "model_outputs": {p_conf["id"]: {"score": 0.5, "raw_output": f"Image load error: {e}", "error": True}
                                      for p_conf in prompts_config_for_inference}
                }
        
        if not batch_pil_images: continue

        original_batch_paths_valid = [batch_image_paths[j] for j in valid_indices_in_batch]
        original_batch_labels_valid = [batch_labels[j] for j in valid_indices_in_batch]

        for prompt_conf in prompts_config_for_inference:
            prompt_id = prompt_conf["id"]
            question = prompt_conf["question"]

            inputs = processor(images=batch_pil_images, text=question, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            generation_config = GenerationConfig(max_new_tokens=max_new_tokens_inf, num_beams=1)
            if processor.tokenizer.pad_token_id is not None: generation_config.pad_token_id = processor.tokenizer.pad_token_id
            if processor.tokenizer.eos_token_id is not None: generation_config.eos_token_id = processor.tokenizer.eos_token_id
            if generation_config.pad_token_id is None and generation_config.eos_token_id is not None: generation_config.pad_token_id = generation_config.eos_token_id

            with torch.no_grad(), torch.autocast(device_type=device.type) if device.type == 'cuda' else contextlib.nullcontext():
                # When using PeftModel, it automatically uses the merged model for generation if needed,
                # or handles the adapter layers correctly.
                generated_ids = model.generate(**inputs, generation_config=generation_config)
            
            raw_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for batch_idx, raw_text in enumerate(raw_generated_texts):
                img_path_str_current = original_batch_paths_valid[batch_idx]
                true_label_current = original_batch_labels_valid[batch_idx]
                path_key = str(img_path_str_current)

                if path_key not in all_detailed_results:
                    all_detailed_results[path_key] = {"true_label": true_label_current, "model_outputs": {}}
                
                cleaned_text = raw_text
                if question and question in raw_text: cleaned_text = raw_text.replace(question, "").strip()
                generated_text_lower = cleaned_text.lower().strip()
                score = 0.5

                target_real_text_lower = prompt_conf["options"]["real"].lower()
                target_fake_text_lower = prompt_conf["options"]["fake"].lower()
                found_fake_keywords = ["fake", "artificial", "generated", "synthetic", "ai-generated", "computer-generated", "not real", "isn't real"]
                found_real_keywords = ["real", "authentic", "photograph", "genuine", "natural", "actual", "not fake", "isn't fake"]
                has_fake = any(keyword in generated_text_lower for keyword in found_fake_keywords) or (target_fake_text_lower in generated_text_lower)
                has_real = any(keyword in generated_text_lower for keyword in found_real_keywords) or (target_real_text_lower in generated_text_lower)
                
                if "not real" in generated_text_lower or "isn't real" in generated_text_lower: score = 1.0
                elif "not fake" in generated_text_lower or "isn't fake" in generated_text_lower: score = 0.0
                elif has_fake and not has_real: score = 1.0
                elif has_real and not has_fake: score = 0.0
                elif has_fake and has_real:
                    if target_real_text_lower in generated_text_lower and target_fake_text_lower not in generated_text_lower: score = 0.0
                    elif target_fake_text_lower in generated_text_lower and target_real_text_lower not in generated_text_lower: score = 1.0
                    else: score = 0.75
                elif not generated_text_lower or generated_text_lower in ["unspecified", "unknown", "i cannot determine", "i can't tell"]: score = 0.5

                all_detailed_results[path_key]["model_outputs"][prompt_id] = {
                    "score": score, "raw_output": raw_text, "cleaned_output": cleaned_text,
                    "parsed_score": score, "error": False
                }
    
    detailed_results_path = results_dir / "lora_inference_detailed_results.json"
    with open(detailed_results_path, 'w') as f:
        json.dump(all_detailed_results, f, indent=4)
    print(f"LoRA Inference detailed results saved to {detailed_results_path}")

    post_process_inference_results(all_detailed_results, prompts_config_for_inference, results_dir, model_name_for_report=f"lora_adapted_{base_model_name.split('/')[-1]}")
    return all_detailed_results

# --- 5. Post-processing for Inference Results ---
def post_process_inference_results(all_detailed_results, prompts_config, base_results_dir, model_name_for_report):
    """ Adapted from vlm_zero_shot.py to handle results structure and save paths """
    
    model_results_dir = base_results_dir / model_name_for_report
    model_results_dir.mkdir(parents=True, exist_ok=True)

    for prompt_conf in prompts_config:
        prompt_id = prompt_conf['id']
        y_true = []
        y_pred_scores = []
        has_data_for_prompt = False

        for img_path_str, data in all_detailed_results.items():
            if "model_outputs" in data and prompt_id in data["model_outputs"]:
                prompt_output = data["model_outputs"][prompt_id]
                if not prompt_output.get("error", False): 
                    y_true.append(data["true_label"])
                    y_pred_scores.append(prompt_output["score"])
                    has_data_for_prompt = True
        
        if not has_data_for_prompt or not y_true:
            print(f"No valid data for prompt {prompt_id} with {model_name_for_report}. Skipping matrix/report.")
            continue

        y_pred_binary = [1 if score > 0.5 else 0 for score in y_pred_scores]

        try:
            cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
            report = classification_report(y_true, y_pred_binary, target_names=['real (0)', 'fake (1)'], output_dict=True, zero_division=0)
            
            report_path = model_results_dir / f"report_prompt_{prompt_id.replace('.', '_')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Classification report for {model_name_for_report} (Prompt {prompt_id}) saved to {report_path}")

            plot_cm_path = model_results_dir / f"cm_prompt_{prompt_id.replace('.', '_')}.png"
            plot_confusion_matrix_graphic(cm, y_true, y_pred_binary, model_name_for_report, prompt_id, plot_cm_path)

        except Exception as e:
            print(f"Error during post-processing for {model_name_for_report} (Prompt {prompt_id}): {e}")

def plot_confusion_matrix_graphic(cm, y_true, y_pred_binary, model_name, prompt_id, plot_path):
    """ Adapted from vlm_zero_shot.py """
    if cm is None or cm.shape != (2,2):
        print(f"Not enough data for CM: {model_name}, prompt {prompt_id}. Plotting placeholder.")
        cm = np.zeros((2,2), dtype=int)
        if y_true and y_pred_binary: # Attempt to populate if possible
            for yt, yp in zip(y_true, y_pred_binary):
                if yt in [0,1] and yp in [0,1]: cm[yt, yp] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real (0)', 'Fake (1)'], yticklabels=['Real (0)', 'Fake (1)'])
    plt.title(f'CM: {model_name} - Prompt {prompt_id}')
    plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix for {model_name} (Prompt {prompt_id}) saved to {plot_path}")

# --- Main Script Execution ---
if __name__ == '__main__':
    # --- Configuration ---
    DATA_PATH = Path("/ssd6/GAIC_Dataset/Chameleon/test/")
    NUM_FINETUNE_SAMPLES = 1000 
    VAL_SPLIT_RATIO = 0.2 
    SEED = 42
    NUM_INFERENCE_SAMPLES_PER_CLASS = 25

    FINETUNE_METHOD = "lora" # Options: "full", "lora"

    # --- LoRA Specific Config (if FINETUNE_METHOD is "lora") ---
    LORA_R = 16 # Rank of the LoRA matrices
    LORA_ALPHA = 32 # Alpha scaling factor for LoRA
    LORA_DROPOUT = 0.05
    # For InstructBLIP (Flan-T5), common target modules are in the T5Block
    # e.g., ["q", "v"] or ["q", "k", "v", "o"] or more specific like ["q_proj", "v_proj"]
    # Let's try a general approach first, PEFT can often infer for known TaskTypes
    # Or specify more directly: e.g. for T5: ["q", "k", "v", "o", "wi_0", "wi_1", "wo"]
    # For now, using a common subset for T5-like models within InstructBLIP
    LORA_TARGET_MODULES = ["q", "v"] # Query and Value matrices in attention layers of the LLM

    # --- Directory and Model Naming --- 
    MODEL_NAME_PRETRAINED = "Salesforce/instructblip-flan-t5-xl"
    RUN_ID_PREFIX = f"{FINETUNE_METHOD}_ft_samples{NUM_FINETUNE_SAMPLES}_val{VAL_SPLIT_RATIO}"
    if FINETUNE_METHOD == "lora":
        RUN_ID_PREFIX += f"_lora_r{LORA_R}_alpha{LORA_ALPHA}"
    RUN_ID = f"{RUN_ID_PREFIX}_seed{SEED}"
    
    BASE_RESULTS_DIR = Path(f"results/instructblip_chameleon/{RUN_ID}")

    # Path for Trainer to save checkpoints (common for both full and LoRA)
    TRAINER_CHECKPOINTS_DIR = BASE_RESULTS_DIR / "trainer_checkpoints" 
    # Path for the final model artifacts (differs for full vs. LoRA)
    # For full FT, it's the whole model. For LoRA, it's just the adapter.
    FINAL_MODEL_ARTIFACTS_DIR = BASE_RESULTS_DIR / "final_model_artifacts"
    
    INFERENCE_RESULTS_DIR = BASE_RESULTS_DIR / "inference_results"
    LOGGING_DIR = BASE_RESULTS_DIR / "training_logs"
    
    # --- Prompt and Tokenizer Config ---
    FINETUNE_PROMPT_CONFIG = USER_PROMPTS[0] 
    FINETUNE_QUESTION = FINETUNE_PROMPT_CONFIG["question"]
    FINETUNE_TARGET_TEXTS = {
        0: FINETUNE_PROMPT_CONFIG["options"]["real"], 
        1: FINETUNE_PROMPT_CONFIG["options"]["fake"]  
    }
    MAX_TOKEN_LENGTH_TARGET = 64 

    # --- Training Hyperparameters (common unless specified otherwise) ---
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 4 
    INFERENCE_BATCH_SIZE = 8
    LEARNING_RATE = 5e-5 # May need adjustment for LoRA (sometimes smaller LR for LoRA)
    if FINETUNE_METHOD == "lora":
        LEARNING_RATE = 1e-4 # LoRA can sometimes benefit from a slightly higher LR
    WARMUP_STEPS = 50
    WEIGHT_DECAY = 0.01

    # Create directories
    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINER_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected Fine-tuning Method: {FINETUNE_METHOD}")

    # --- 1. Prepare Data ---
    print("\n--- Preparing Data ---")
    data_splits = prepare_finetuning_and_inference_data(
        DATA_PATH, 
        num_total_finetune_samples=NUM_FINETUNE_SAMPLES,
        train_val_split_ratio=VAL_SPLIT_RATIO,
        seed=SEED
    )
    if not data_splits['train_paths'] or not data_splits['val_paths']:
        print("Error: No data for training or validation. Exiting."); exit()

    # --- 2. Initialize Processor and Create Datasets for Fine-tuning ---
    print("\n--- Initializing Processor and Creating Datasets for Fine-tuning ---")
    try:
        processor = InstructBlipProcessor.from_pretrained(MODEL_NAME_PRETRAINED)
        train_dataset = FinetuneImageDataset(image_paths=data_splits['train_paths'], labels=data_splits['train_labels'], processor=processor, prompt=FINETUNE_QUESTION, target_texts=FINETUNE_TARGET_TEXTS, max_length=MAX_TOKEN_LENGTH_TARGET)
        val_dataset = FinetuneImageDataset(image_paths=data_splits['val_paths'], labels=data_splits['val_labels'], processor=processor, prompt=FINETUNE_QUESTION, target_texts=FINETUNE_TARGET_TEXTS, max_length=MAX_TOKEN_LENGTH_TARGET)
        print(f"Train dataset: {len(train_dataset)} samples. Val dataset: {len(val_dataset)} samples.")
    except Exception as e:
        print(f"Error initializing processor or datasets: {e}. Exiting."); exit()

    # --- 3. Fine-tune the Model (Full or LoRA) ---
    SHOULD_TRAIN = True 
    # Path to the final artifacts (either full model dir or LoRA adapter dir)
    path_to_trained_artifacts_for_inference = FINAL_MODEL_ARTIFACTS_DIR

    if SHOULD_TRAIN or not (path_to_trained_artifacts_for_inference / ("pytorch_model.bin" if FINETUNE_METHOD == "full" else "adapter_model.bin")).exists():
        print(f"\n--- Starting Fine-tuning (Method: {FINETUNE_METHOD}) --- ")
        if FINETUNE_METHOD == "full":
            # finetune_model now saves its final model to TRAINER_CHECKPOINTS_DIR / "final_model"
            # And processor is also saved there.
            full_model_path = finetune_model(
                model_name=MODEL_NAME_PRETRAINED,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=TRAINER_CHECKPOINTS_DIR, # Trainer saves checkpoints here
                logging_dir=LOGGING_DIR,
                epochs=EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                weight_decay=WEIGHT_DECAY,
                device=device
            )
            # For full fine-tuning, the artifacts directory is where the model and processor are saved by finetune_model.
            # Ensure path_to_trained_artifacts_for_inference points to this for consistency.
            # finetune_model returns the path to the directory containing the final model & processor.
            path_to_trained_artifacts_for_inference = Path(full_model_path) 
            print(f"Full Fine-tuning complete. Model artifacts for inference are at: {path_to_trained_artifacts_for_inference}")
        
        elif FINETUNE_METHOD == "lora":
            lora_params = {
                "r": LORA_R, 
                "lora_alpha": LORA_ALPHA, 
                "lora_dropout": LORA_DROPOUT, 
                "target_modules": LORA_TARGET_MODULES
            }
            # finetune_model_with_lora saves the adapter to FINAL_MODEL_ARTIFACTS_DIR directly.
            adapter_path = finetune_model_with_lora(
                base_model_name=MODEL_NAME_PRETRAINED,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir_checkpoints=TRAINER_CHECKPOINTS_DIR, # Trainer saves its checkpoints here
                final_lora_adapter_dir=FINAL_MODEL_ARTIFACTS_DIR, # Final adapter saved here
                logging_dir=LOGGING_DIR,
                lora_config_params=lora_params,
                epochs=EPOCHS,
                batch_size=TRAIN_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS,
                weight_decay=WEIGHT_DECAY,
                device=device
            )
            path_to_trained_artifacts_for_inference = Path(adapter_path)
            print(f"LoRA Fine-tuning complete. Adapter for inference is at: {path_to_trained_artifacts_for_inference}")
    else:
        print(f"Skipping training. Using existing artifacts from: {path_to_trained_artifacts_for_inference}")

    # --- Sanity check for trained artifacts before inference ---
    if FINETUNE_METHOD == "full" and not (path_to_trained_artifacts_for_inference / "pytorch_model.bin").exists():
        print(f"Error: Full fine-tuned model not found at {path_to_trained_artifacts_for_inference}. Exiting."); exit()
    elif FINETUNE_METHOD == "lora" and not (path_to_trained_artifacts_for_inference / "adapter_model.bin").exists():
        print(f"Error: LoRA adapter not found at {path_to_trained_artifacts_for_inference}. Exiting."); exit()
    if not (path_to_trained_artifacts_for_inference / "preprocessor_config.json").exists(): # Processor should also be there
        print(f"Error: Processor config not found at {path_to_trained_artifacts_for_inference}. Exiting."); exit()

    # --- 4. Run Inference on Unseen Data ---
    print("\n--- Preparing for Inference on Unseen Data ---")
    available_inf_real = data_splits['inference_candidate_real']
    available_inf_fake = data_splits['inference_candidate_fake']
    num_inf_real_to_sample = min(NUM_INFERENCE_SAMPLES_PER_CLASS, len(available_inf_real))
    num_inf_fake_to_sample = min(NUM_INFERENCE_SAMPLES_PER_CLASS, len(available_inf_fake))
    inference_real_paths = random.sample(available_inf_real, num_inf_real_to_sample) if num_inf_real_to_sample > 0 else []
    inference_fake_paths = random.sample(available_inf_fake, num_inf_fake_to_sample) if num_inf_fake_to_sample > 0 else []
    if not inference_real_paths and not inference_fake_paths:
        print("Warning: No real or fake images available for inference sampling based on current settings.")
    
    inference_paths = inference_real_paths + inference_fake_paths
    inference_labels = ([0] * len(inference_real_paths)) + ([1] * len(inference_fake_paths))
    
    if not inference_paths:
        print("No unseen images available or sampled for inference. Exiting."); exit()
    
    print(f"Running inference on {len(inference_paths)} unseen samples ({len(inference_real_paths)} real, {len(inference_fake_paths)} fake)." )
    
    if FINETUNE_METHOD == "full":
        run_inference_on_finetuned_model(
            finetuned_model_path=str(path_to_trained_artifacts_for_inference), 
            inference_image_paths=inference_paths,
            inference_labels=inference_labels,
            prompts_config_for_inference=USER_PROMPTS,
            results_dir=INFERENCE_RESULTS_DIR,
            device=device,
            batch_size=INFERENCE_BATCH_SIZE,
            max_new_tokens_inf=70
        )
    elif FINETUNE_METHOD == "lora":
        run_inference_on_lora_model(
            base_model_name=MODEL_NAME_PRETRAINED,
            lora_adapter_path=str(path_to_trained_artifacts_for_inference),
            inference_image_paths=inference_paths,
            inference_labels=inference_labels,
            prompts_config_for_inference=USER_PROMPTS, 
            results_dir=INFERENCE_RESULTS_DIR,
            device=device,
            batch_size=INFERENCE_BATCH_SIZE,
            max_new_tokens_inf=70
        )

    print("\n--- Fine-tuning and Inference Script Finished ---") 