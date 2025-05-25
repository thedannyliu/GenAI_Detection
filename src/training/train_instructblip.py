#!/usr/bin/env python3
"""
InstructBLIP LoRA Fine-tuning for AI-Generated Image Detection
===========================================================

This script fine-tunes the Salesforce/instructblip-vicuna-7b model using LoRA
for binary classification of real vs AI-generated images.

Usage:
    python train_instructblip_lora.py --config configs/train_instructblip_config.yaml

Requirements:
    pip install transformers torch torchvision peft datasets accelerate pillow pyyaml
"""

import argparse
import os
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIImageDataset(Dataset):
    """Dataset for AI-generated image detection."""
    
    def __init__(
        self,
        real_images: List[Path],
        ai_images: List[Path],
        processor: InstructBlipProcessor,
        max_target_length: int = 64
    ):
        self.real_images = real_images
        self.ai_images = ai_images
        self.processor = processor
        self.max_target_length = max_target_length
        
        # Combine all images with labels
        self.images = real_images + ai_images
        self.labels = [0] * len(real_images) + [1] * len(ai_images)  # 0: real, 1: AI
        
        # Define prompts
        self.question = "Is this image real or AI-generated? Answer with one word."
        self.label_to_text = {0: "real", 1: "fake"}
        
        logger.info(f"Dataset created with {len(real_images)} real and {len(ai_images)} AI images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new("RGB", (224, 224), color="white")
        
        # Prepare target text
        target_text = self.label_to_text[label]
        
        # Create the full prompt for training
        full_prompt = f"{self.question} {target_text}"
        
        # Process image and text with InstructBLIP processor
        inputs = self.processor(
            images=image,
            text=full_prompt,
            max_length=512,  # Set a reasonable max length
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        return_dict = {}
        for k, v in inputs.items():
            return_dict[k] = v.squeeze(0)
        
        # Create labels for causal language modeling
        # Labels should be the same as input_ids, but with -100 for tokens we don't want to predict
        labels = return_dict["input_ids"].clone()
        
        # Find where the answer starts (after the question)
        question_tokens = self.processor.tokenizer.encode(self.question, add_special_tokens=False)
        question_length = len(question_tokens)
        
        # Set question part to -100 (don't compute loss on question)
        labels[:question_length] = -100
        
        # Set padding tokens to -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return_dict["labels"] = labels
        
        return return_dict

def collect_images(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Collect all image files from a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    
    return images

def split_dataset(
    real_images: List[Path],
    ai_images: List[Path],
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """Split dataset into train/val/test sets with balanced real/AI images."""
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle the lists
    real_shuffled = real_images.copy()
    ai_shuffled = ai_images.copy()
    random.shuffle(real_shuffled)
    random.shuffle(ai_shuffled)
    
    # Calculate splits (assuming balanced classes)
    train_real = num_train // 2
    train_ai = num_train - train_real
    val_real = num_val // 2
    val_ai = num_val - val_real
    test_real = num_test // 2
    test_ai = num_test - test_real
    
    # Split real images
    real_train = real_shuffled[:train_real]
    real_val = real_shuffled[train_real:train_real + val_real]
    real_test = real_shuffled[train_real + val_real:train_real + val_real + test_real]
    
    # Split AI images
    ai_train = ai_shuffled[:train_ai]
    ai_val = ai_shuffled[train_ai:train_ai + val_ai]
    ai_test = ai_shuffled[train_ai + val_ai:train_ai + val_ai + test_ai]
    
    logger.info(f"Dataset split: Train({len(real_train)}+{len(ai_train)}), "
                f"Val({len(real_val)}+{len(ai_val)}), Test({len(real_test)}+{len(ai_test)})")
    
    return real_train, ai_train, real_val, ai_val, real_test, ai_test

class CustomDataCollator:
    """Custom data collator for InstructBLIP with proper batch handling."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Extract components from batch
        batch_dict = {}
        
        # Get all unique keys from the batch items
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())
        
        # Stack each key separately
        for key in all_keys:
            values = [item[key] for item in batch if key in item]
            if values:
                try:
                    # Stack tensors of the same shape
                    stacked = torch.stack(values)
                    batch_dict[key] = stacked
                except Exception as e:
                    logger.warning(f"Could not stack {key}: {e}")
                    # If stacking fails, pad to the same length
                    max_len = max(v.shape[-1] for v in values)
                    padded_values = []
                    for v in values:
                        if v.shape[-1] < max_len:
                            pad_size = max_len - v.shape[-1]
                            if key == 'labels':
                                # Pad labels with -100
                                padded = torch.cat([v, torch.full((pad_size,), -100, dtype=v.dtype)])
                            else:
                                # Pad other tensors with pad_token_id or 0
                                pad_value = getattr(self.processor.tokenizer, 'pad_token_id', 0)
                                padded = torch.cat([v, torch.full((pad_size,), pad_value, dtype=v.dtype)])
                            padded_values.append(padded)
                        else:
                            padded_values.append(v)
                    batch_dict[key] = torch.stack(padded_values)
        
        return batch_dict

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation."""
    predictions, labels = eval_pred
    
    # For text generation, we need to decode predictions first
    # This is a simplified approach - you may need to adjust based on your specific needs
    if len(predictions.shape) > 2:
        # Take the most likely token at each position
        predictions = np.argmax(predictions, axis=-1)
    
    # Flatten predictions and labels, removing -100 labels
    flat_predictions = predictions.flatten()
    flat_labels = labels.flatten()
    
    # Filter out -100 labels (padding tokens)
    mask = flat_labels != -100
    filtered_labels = flat_labels[mask]
    filtered_predictions = flat_predictions[mask]
    
    if len(filtered_labels) == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    # For binary classification, we need to map token IDs to class labels
    # This is a simplified approach - you may need to adjust based on your tokenizer
    # Convert to binary predictions (0 or 1)
    binary_predictions = filtered_predictions % 2  # Simple mapping
    binary_labels = filtered_labels % 2
    
    # Calculate metrics
    try:
        accuracy = accuracy_score(binary_labels, binary_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_predictions, average='binary', zero_division=0
        )
    except Exception as e:
        logger.warning(f"Error computing metrics: {e}")
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def setup_device_and_cuda(gpu_id: int = None) -> torch.device:
    """Setup the computing device and CUDA environment."""
    # Check if CUDA_VISIBLE_DEVICES is already set
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if cuda_visible_devices is not None:
        # If CUDA_VISIBLE_DEVICES is set, use GPU 0 (which maps to the visible device)
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
            logger.info(f"Using GPU via CUDA_VISIBLE_DEVICES={cuda_visible_devices}: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logger.info("CUDA not available, using CPU")
    elif gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            logger.warning(f"GPU {gpu_id} not available, using GPU 0")
            device = torch.device("cuda:0")
            torch.cuda.set_device(0)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        logger.info(f"Using default GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_output_directory(config: Dict[str, Any]) -> Path:
    """Create output directory based on configuration."""
    data_config = config['data']
    model_config = config['model']
    
    # Create a unique run ID
    run_id = (f"instructblip_lora_"
              f"samples_{data_config['num_train_samples']}_"
              f"r_{model_config['lora_params']['r']}_"
              f"alpha_{model_config['lora_params']['alpha']}_"
              f"seed_{data_config['seed']}")
    
    output_dir = Path(config['output']['base_results_dir_root']) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    return output_dir

class CustomTrainer(Trainer):
    """Custom trainer to handle InstructBLIP-specific issues."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation to handle batch size mismatches."""
        
        # Debug print to see tensor shapes
        logger.debug(f"Input shapes:")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                logger.debug(f"  {k}: {v.shape}")
        
        # Ensure labels match input_ids shape
        if 'labels' in inputs and 'input_ids' in inputs:
            input_ids_shape = inputs['input_ids'].shape
            labels_shape = inputs['labels'].shape
            
            if input_ids_shape != labels_shape:
                logger.warning(f"Shape mismatch: input_ids {input_ids_shape} vs labels {labels_shape}")
                # Adjust labels to match input_ids
                if len(labels_shape) == 1 and len(input_ids_shape) == 2:
                    # If labels is 1D and input_ids is 2D, adjust labels
                    batch_size, seq_len = input_ids_shape
                    if labels_shape[0] == seq_len:
                        # Repeat labels for each batch item
                        inputs['labels'] = inputs['labels'].unsqueeze(0).repeat(batch_size, 1)
                    elif labels_shape[0] == batch_size:
                        # Pad labels to match sequence length
                        pad_length = seq_len - 1  # -1 because labels are typically shifted
                        padding = torch.full((batch_size, pad_length), -100, 
                                           dtype=inputs['labels'].dtype, 
                                           device=inputs['labels'].device)
                        inputs['labels'] = torch.cat([inputs['labels'].unsqueeze(-1), padding], dim=1)
                elif labels_shape[1] < input_ids_shape[1]:
                    # If labels sequence is shorter, pad it
                    pad_length = input_ids_shape[1] - labels_shape[1]
                    padding = torch.full((labels_shape[0], pad_length), -100,
                                       dtype=inputs['labels'].dtype,
                                       device=inputs['labels'].device)
                    inputs['labels'] = torch.cat([inputs['labels'], padding], dim=1)
                elif labels_shape[1] > input_ids_shape[1]:
                    # If labels sequence is longer, truncate it
                    inputs['labels'] = inputs['labels'][:, :input_ids_shape[1]]
        
        try:
            # Forward pass
            outputs = model(**inputs)
            
            # Extract loss from outputs
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                # If no loss in outputs, compute it manually
                logits = outputs.logits
                labels = inputs.get('labels')
                
                if labels is not None:
                    # Shift labels for causal LM
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Flatten for cross entropy
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                else:
                    loss = torch.tensor(0.0, requires_grad=True, device=logits.device)
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            logger.error(f"Error in compute_loss: {e}")
            logger.error(f"Final input shapes:")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"  {k}: {v.shape}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune InstructBLIP for AI image detection")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    env_config = config.get('environment', {})
    
    # Setup device and CUDA environment FIRST
    device = setup_device_and_cuda(env_config.get('gpu_id'))
    
    # Set random seeds
    seed = data_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Create output directory
    output_dir = create_output_directory(config)
    
    # Load data
    train_path = Path(data_config['train_path'])
    val_test_path = Path(data_config['val_test_path'])
    
    # Collect images from train directory
    train_real_images = collect_images(train_path / "nature")
    train_ai_images = collect_images(train_path / "ai")
    
    # Collect images from validation directory
    val_real_images = collect_images(val_test_path / "nature")
    val_ai_images = collect_images(val_test_path / "ai")
    
    logger.info(f"Found {len(train_real_images)} real and {len(train_ai_images)} AI training images")
    logger.info(f"Found {len(val_real_images)} real and {len(val_ai_images)} AI validation images")
    
    # Split datasets
    real_train, ai_train, real_val, ai_val, real_test, ai_test = split_dataset(
        train_real_images + val_real_images,
        train_ai_images + val_ai_images,
        data_config['num_train_samples'],
        data_config['num_val_samples'],
        data_config['num_test_samples'],
        seed
    )
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load processor and model
    logger.info("Loading processor and model...")
    processor = InstructBlipProcessor.from_pretrained(
        model_config['name_pretrained'],
        use_fast=False  # Explicitly use slow processor to avoid warnings
    )
    
    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model directly to the correct device
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_config['name_pretrained'],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device}  # This ensures model loads directly to our device
    )
    
    # Setup LoRA
    if model_config['finetune_method'] == 'lora':
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=model_config['lora_params']['r'],
            lora_alpha=model_config['lora_params']['alpha'],
            lora_dropout=model_config['lora_params']['dropout'],
            target_modules=model_config['lora_params']['target_modules'],
            bias="none",
            inference_mode=False,
        )
        
        # Apply LoRA to the language model
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()
        logger.info("LoRA configuration applied to language model")
    
    # Create datasets
    train_dataset = AIImageDataset(
        real_train, ai_train, processor, training_config['max_target_token_length']
    )
    val_dataset = AIImageDataset(
        real_val, ai_val, processor, training_config['max_target_token_length']
    )
    
    # Create data collator
    data_collator = CustomDataCollator(processor)
    
    # Training arguments
    learning_rate = (training_config['learning_rate_lora'] 
                    if model_config['finetune_method'] == 'lora' 
                    else training_config['learning_rate_full'])
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=training_config['epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        learning_rate=learning_rate,
        warmup_steps=training_config['warmup_steps'],
        weight_decay=training_config['weight_decay'],
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        eval_strategy=training_config['evaluation_strategy'],
        save_strategy=training_config['save_strategy'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],  # Disable wandb
        dataloader_pin_memory=False,
        bf16=training_config.get('use_bfloat16', True) and torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        gradient_checkpointing=training_config.get('gradient_checkpointing', False),
        prediction_loss_only=False,
        include_inputs_for_metrics=False,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=training_config['early_stopping_patience'],
        early_stopping_threshold=training_config['early_stopping_threshold']
    )
    
    # Initialize custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[early_stopping],
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    if training_config['should_train']:
        logger.info("Starting training...")
        try:
            trainer.train()
            
            # Save the final model
            final_model_path = output_dir / "final_model"
            trainer.save_model(str(final_model_path))
            processor.save_pretrained(str(final_model_path))
            
            # Save LoRA adapter separately if using LoRA
            if model_config['finetune_method'] == 'lora':
                lora_path = output_dir / "lora_adapter"
                model.language_model.save_pretrained(str(lora_path))
                logger.info(f"LoRA adapter saved to {lora_path}")
            
            logger.info(f"Training completed. Model saved to {final_model_path}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
    
    # Save configuration for reference
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()