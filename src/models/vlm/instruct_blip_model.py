import argparse
import os
import random
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Is this image AI-generated? Answer 'yes' or 'no'."
YES, NO = "yes", "no"


class AINatureDataset(Dataset):
    def __init__(self, root: str, processor: InstructBlipProcessor, max_samples: Optional[int] = None):
        self.processor = processor
        self.samples = []
        
        # Load samples from both folders
        for lbl_name, lbl_id in [("ai", 1), ("nature", 0)]:
            folder = Path(root) / lbl_name
            if not folder.exists():
                raise FileNotFoundError(f"{folder} not found")
            
            folder_samples = []
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    folder_samples.append((str(p), lbl_id))
            
            logger.info(f"Found {len(folder_samples)} samples in {folder}")
            self.samples.extend(folder_samples)
        
        # Shuffle and limit samples if specified
        random.shuffle(self.samples)
        if max_samples and max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Using {len(self.samples)} total samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {path}: {e}")
            # Return a blank image if loading fails
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        answer_text = YES if label == 1 else NO
        
        # Process image and prompt separately for proper label creation
        inputs = self.processor(
            images=img,
            text=PROMPT,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Process the answer separately to get its token IDs
        answer_inputs = self.processor.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        processed_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                processed_inputs[k] = v.squeeze(0)
            else:
                processed_inputs[k] = v
        
        # Create labels for the answer tokens only
        input_ids = processed_inputs["input_ids"]
        answer_ids = answer_inputs["input_ids"].squeeze(0)
        
        # Create labels: -100 for prompt tokens, actual token IDs for answer tokens
        labels = torch.full_like(input_ids, -100)
        
        # Append answer tokens to input_ids and update labels
        full_input_ids = torch.cat([input_ids, answer_ids])
        full_labels = torch.cat([labels, answer_ids])
        
        # Update attention mask
        answer_attention = torch.ones_like(answer_ids)
        full_attention_mask = torch.cat([processed_inputs["attention_mask"], answer_attention])
        
        processed_inputs["input_ids"] = full_input_ids
        processed_inputs["attention_mask"] = full_attention_mask
        processed_inputs["labels"] = full_labels
        
        return processed_inputs


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_lora(model: InstructBlipForConditionalGeneration, lora_cfg: Dict[str, Any]):
    """Apply LoRA fine-tuning to the model."""
    
    # Prepare model for k-bit training (important for gradient computation)
    model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    
    # Apply LoRA to the language model part
    model.language_model = get_peft_model(model.language_model, lora_config)
    
    # Enable gradients for LoRA parameters and ensure proper setup
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return model


class CustomDataCollator:
    """Custom data collator for InstructBLIP."""
    
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __call__(self, features):
        # Extract all the different input types
        batch = {}
        
        # Handle text inputs
        if "input_ids" in features[0]:
            input_ids = [f["input_ids"] for f in features]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["input_ids"] = input_ids
        
        if "attention_mask" in features[0]:
            attention_masks = [f["attention_mask"] for f in features]
            attention_masks = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            batch["attention_mask"] = attention_masks
        
        # Handle Q-Former inputs
        if "qformer_input_ids" in features[0]:
            qformer_input_ids = [f["qformer_input_ids"] for f in features]
            qformer_input_ids = torch.nn.utils.rnn.pad_sequence(
                qformer_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["qformer_input_ids"] = qformer_input_ids
        
        if "qformer_attention_mask" in features[0]:
            qformer_attention_masks = [f["qformer_attention_mask"] for f in features]
            qformer_attention_masks = torch.nn.utils.rnn.pad_sequence(
                qformer_attention_masks, batch_first=True, padding_value=0
            )
            batch["qformer_attention_mask"] = qformer_attention_masks
        
        # Handle image inputs
        if "pixel_values" in features[0]:
            pixel_values = [f["pixel_values"] for f in features]
            batch["pixel_values"] = torch.stack(pixel_values)
        
        # Handle labels
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )
            batch["labels"] = labels
        
        return batch


def compute_metrics(eval_preds):
    """Compute accuracy metrics."""
    predictions, labels = eval_preds
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Get predicted token IDs (take argmax over vocabulary dimension)
    if predictions.ndim == 3:  # [batch_size, seq_len, vocab_size]
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions
    
    # Calculate accuracy only for non-masked tokens
    mask = labels != -100
    if mask.sum() == 0:
        return {"accuracy": 0.0}
    
    correct = (predicted_ids == labels) & mask
    accuracy = correct.sum() / mask.sum()
    
    return {"accuracy": float(accuracy)}


class CustomTrainer(Trainer):
    """Custom trainer with improved handling."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation to ensure proper gradient flow."""
        outputs = model(**inputs)
        loss = outputs.loss
        
        if loss is None:
            # Fallback loss computation if model doesn't return loss
            logits = outputs.logits
            labels = inputs["labels"]
            
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step for better evaluation."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            
            if prediction_loss_only:
                return (loss, None, None)
            
            logits = outputs.logits if hasattr(outputs, 'logits') else None
            labels = inputs.get("labels")
            
            if logits is not None:
                logits = logits.detach().cpu().numpy()
            if labels is not None:
                labels = labels.detach().cpu().numpy()
            
            return (loss, logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune InstructBLIP for AI image detection")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Load configuration
    cfg = load_yaml(args.config)
    set_seed(cfg["data"].get("seed", 42))
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Using model: {cfg['model']['name_pretrained']}")

    # Load processor and model
    logger.info("Loading processor and model...")
    processor = InstructBlipProcessor.from_pretrained(cfg["model"]["name_pretrained"])
    
    # Ensure tokenizer has pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Load model with proper device mapping
    model = InstructBlipForConditionalGeneration.from_pretrained(
        cfg["model"]["name_pretrained"],
        torch_dtype=torch.bfloat16 if cfg["training"]["use_bfloat16"] else torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Apply fine-tuning method
    if cfg["model"]["finetune_method"] == "lora":
        logger.info("Applying LoRA fine-tuning...")
        model = apply_lora(model, cfg["model"]["lora_params"])
    else:
        raise ValueError("Only LoRA fine-tuning is currently supported")

    # Prepare datasets
    logger.info("Loading datasets...")
    train_ds = AINatureDataset(
        cfg["data"]["train_path"], 
        processor,
        max_samples=cfg["data"].get("num_train_samples")
    )
    val_ds = AINatureDataset(
        cfg["data"]["val_test_path"], 
        processor,
        max_samples=cfg["data"].get("num_val_samples")
    )
    
    # Debug: Check a sample
    logger.info("Checking sample data...")
    sample = train_ds[0]
    logger.info(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        if torch.is_tensor(value):
            logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
    # Custom data collator
    collator = CustomDataCollator(processor)

    # Setup output directory
    out_dir = Path(cfg["output"]["base_results_dir_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"].get("eval_batch_size", 1),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate_lora"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["use_bfloat16"],
        fp16=not cfg["training"]["use_bfloat16"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        eval_strategy=cfg["training"]["evaluation_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[] if cfg["environment"]["disable_wandb"] else ["wandb"],
        dataloader_num_workers=1,  # Reduced to avoid issues
        logging_steps=cfg["training"].get("logging_steps", 10),
        eval_steps=cfg["training"].get("eval_steps", 500),
        save_steps=cfg["training"].get("save_steps", 500),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        prediction_loss_only=False,
        dataloader_drop_last=False,
        # Disable cache to avoid issues with gradient checkpointing
        use_cache=False,
    )

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if cfg["training"]["should_train"] else None,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,  # Use processing_class instead of tokenizer
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                early_stopping_threshold=cfg["training"]["early_stopping_threshold"],
            )
        ] if cfg["training"]["should_train"] else [],
    )

    # Training
    if cfg["training"]["should_train"]:
        logger.info("Starting training...")
        
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
        
        # Save model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Save processor
        processor.save_pretrained(out_dir)

    # Final evaluation
    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    
    # Save metrics
    metrics_file = out_dir / "eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Final validation metrics: {metrics}")
    logger.info(f"Training completed successfully! Results saved to {out_dir}")


if __name__ == "__main__":
    main()