import argparse
import os
import random
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from peft import get_peft_model, PromptTuningConfig, TaskType
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Is this image AI-generated? Answer 'yes' or 'no'."
YES, NO = "yes", "no"


class AINatureDataset(Dataset):
    """Dataset for AI vs Nature image classification with Prompt Tuning setup."""
    
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
            img = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        answer_text = YES if label == 1 else NO
        
        # Create the full text sequence: prompt + answer
        full_text = PROMPT + " " + answer_text
        
        # Process image and full text together
        inputs = self.processor(
            images=img,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # Remove batch dimension
        processed_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                processed_inputs[k] = v.squeeze(0)
            else:
                processed_inputs[k] = v
        
        # For training, we need to create labels that mask the prompt part
        # and only compute loss on the answer part
        prompt_tokens = self.processor.tokenizer(
            PROMPT, 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        
        answer_tokens = self.processor.tokenizer(
            answer_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        
        # Create labels: -100 for prompt tokens, actual tokens for answer
        full_input_ids = processed_inputs["input_ids"]
        labels = torch.full_like(full_input_ids, -100)
        
        # Find where the answer starts in the full sequence
        # This is a simplified approach - in practice you might want more robust alignment
        prompt_length = len(prompt_tokens)
        if len(full_input_ids) >= prompt_length + len(answer_tokens):
            # Place answer tokens at the expected position
            answer_start = len(full_input_ids) - len(answer_tokens)
            labels[answer_start:] = answer_tokens
        else:
            # Fallback: just use the last few tokens as answer
            labels[-len(answer_tokens):] = answer_tokens
        
        processed_inputs["labels"] = labels
        processed_inputs["answer_text"] = answer_text  # Keep for reference
        
        return processed_inputs


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_prompt_tuning_correctly(model: InstructBlipForConditionalGeneration, prompt_cfg: Dict[str, Any]):
    """Apply Prompt Tuning with proper gradient setup."""
    
    logger.info("Setting up Prompt Tuning configuration...")
    
    # Create Prompt Tuning config
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=prompt_cfg["num_virtual_tokens"],
        prompt_tuning_init=prompt_cfg["init_method"],
        prompt_tuning_init_text=prompt_cfg.get("init_text", None),
        tokenizer_name_or_path=prompt_cfg.get("tokenizer_name_or_path", None),
    )
    
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply Prompt Tuning only to the language model
    logger.info("Applying Prompt Tuning to language model...")
    model.language_model = get_peft_model(model.language_model, prompt_config)
    
    # Enable training mode for Prompt Tuning-enhanced model
    model.language_model.train()
    
    # Explicitly enable gradients for prompt tuning parameters
    prompt_param_count = 0
    for name, param in model.named_parameters():
        if "prompt_embeddings" in name.lower() or "prompt_encoder" in name.lower():
            param.requires_grad = True
            prompt_param_count += param.numel()
            logger.debug(f"Enabled gradients for: {name} - shape: {param.shape}")
    
    # Verify gradient setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Prompt Tuning parameters found: {prompt_param_count:,}")
    logger.info(f"Total trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Prompt Tuning setup failed.")
    
    return model


class InstructBLIPDataCollator:
    """Fixed data collator for InstructBLIP with proper padding."""
    
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        # Ensure pad token is properly set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features):
        batch = {}
        
        # Stack pixel values
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
        # Handle text inputs with proper padding
        text_fields = ["input_ids", "attention_mask"]
        for field in text_fields:
            if field in features[0]:
                tensors = [f[field] for f in features]
                pad_value = self.tokenizer.pad_token_id if field == "input_ids" else 0
                padded = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=pad_value
                )
                batch[field] = padded
        
        # Handle Q-Former inputs
        qformer_fields = ["qformer_input_ids", "qformer_attention_mask"]
        for field in qformer_fields:
            if field in features[0]:
                tensors = [f[field] for f in features]
                pad_value = self.tokenizer.pad_token_id if "input_ids" in field else 0
                padded = torch.nn.utils.rnn.pad_sequence(
                    tensors, batch_first=True, padding_value=pad_value
                )
                batch[field] = padded
        
        # Handle labels - FIXED VERSION
        if "labels" in features[0]:
            # Pad labels to match input_ids length
            labels_list = [f["labels"] for f in features]
            padded_labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100
            )
            batch["labels"] = padded_labels
        
        return batch


def compute_accuracy_metrics(eval_preds):
    """Compute accuracy for the model predictions - FIXED VERSION."""
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert to numpy if they're tensors
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    # Get predicted token IDs from logits
    if predictions.ndim == 3:  # [batch, seq_len, vocab_size]
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions

    # For Causal LMs, logits at position i are predictions for token at position i+1.
    # We need to shift the predictions and labels to align them correctly for comparison.
    shifted_predictions = predicted_ids[..., :-1]
    shifted_labels = labels[..., 1:]

    # Only compute accuracy for non-masked positions in the shifted labels
    valid_positions = shifted_labels != -100

    if valid_positions.sum() == 0:
        return {"accuracy": 0.0}

    # Compare shifted predictions with shifted labels at valid positions
    correct_predictions = (shifted_predictions == shifted_labels) & valid_positions

    # Calculate accuracy
    accuracy = float(correct_predictions.sum()) / float(valid_positions.sum())

    return {"accuracy": accuracy}


class InstructBLIPPromptTuningTrainer(Trainer):
    """Custom trainer for InstructBLIP with Prompt Tuning and proper loss handling."""
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Custom save method to pass `safe_serialization=False` to `save_pretrained`.
        This is necessary for models with shared weights when using an older version
        of the transformers library.
        """
        # We are overriding the an internal method of the Trainer to force `safe_serialization=False`
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        FIXED compute_loss method that handles InstructBLIP properly.
        We pop the labels before the forward pass to compute loss manually,
        which is necessary for prompt tuning. We then add them back for metric computation.
        """
        model.train()
        
        # Pop labels to prevent the model from computing loss internally.
        labels = inputs.pop("labels")
        
        # Forward pass to get logits
        outputs = model(**inputs)

        # Put labels back for other parts of the Trainer that might need it (e.g., prediction_step for metrics)
        inputs["labels"] = labels

        logits = outputs.logits
        
        # CRITICAL FIX: Handle the shape mismatch issue
        # The issue is that logits and labels might have different sequence lengths
        # due to Q-Former outputs and virtual prompt tokens. We align them from the right.
        
        batch_size, logits_seq_len, vocab_size = logits.shape
        labels_seq_len = labels.shape[1]
        
        if logits_seq_len != labels_seq_len:
            # Case 1: logits are longer than labels - align from the right
            if logits_seq_len > labels_seq_len:
                logits = logits[:, -labels_seq_len:, :]
            # Case 2: labels are longer than logits - also align from the right
            else:
                labels = labels[:, -logits_seq_len:]
        
        # Now logits and labels should have matching sequence lengths
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Ensure shapes match exactly
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # DEBUG: Print shapes to verify alignment
        logger.debug(f"flat_logits shape: {flat_logits.shape}")
        logger.debug(f"flat_labels shape: {flat_labels.shape}")
        
        if flat_logits.size(0) != flat_labels.size(0):
            logger.error(f"Shape mismatch: logits {flat_logits.shape} vs labels {flat_labels.shape}")
            # Emergency fix: truncate to minimum size
            min_size = min(flat_logits.size(0), flat_labels.size(0))
            flat_logits = flat_logits[:min_size]
            flat_labels = flat_labels[:min_size]
        
        loss = loss_fct(flat_logits, flat_labels)
        
        # Create output object to maintain compatibility
        class LossOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        outputs = LossOutput(loss, logits)
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step while properly handling DynamicCache objects.
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        
        # For generation/inference, we need to clean the inputs
        inputs = self._prepare_inputs(inputs)
        
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        
        # Override ignore_keys to exclude cache-related keys
        # The 'past_key_values' key is handled by the model internally and should not be in inputs
        # if 'past_key_values' in inputs:
        #     del inputs['past_key_values']
        
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                logits = outputs.logits if hasattr(outputs, 'logits') else None
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

        if prediction_loss_only:
            return (loss, None, None)

        # Handle logits properly
        if logits is not None:
            logits = logits.detach()
        
        # Handle labels
        labels = None
        if has_labels:
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()

        return (loss, logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune InstructBLIP for AI image detection using Prompt Tuning")
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
    
    # Setup tokenizer padding
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    # Load model
    model = InstructBlipForConditionalGeneration.from_pretrained(
        cfg["model"]["name_pretrained"],
        torch_dtype=torch.bfloat16 if cfg["training"]["use_bfloat16"] else torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True
    )

    # Apply Prompt Tuning fine-tuning
    if cfg["model"]["finetune_method"] == "prompt_tuning":
        logger.info("Applying Prompt Tuning fine-tuning...")
        # Set tokenizer for prompt tuning config
        prompt_cfg = cfg["model"]["prompt_tuning_params"].copy()
        prompt_cfg["tokenizer_name_or_path"] = cfg["model"]["name_pretrained"]
        model = apply_prompt_tuning_correctly(model, prompt_cfg)
    else:
        raise ValueError("Only Prompt Tuning fine-tuning is supported in this script")

    # Load datasets
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
    
    # Debug sample
    logger.info("Checking sample data...")
    sample = train_ds[0]
    logger.info(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        if torch.is_tensor(value):
            logger.info(f"{key}: shape={value.shape}, dtype={value.dtype}")
    
    # Data collator
    collator = InstructBLIPDataCollator(processor)

    # Output directory
    out_dir = Path(cfg["output"]["base_results_dir_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"].get("eval_batch_size", 4),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate_prompt_tuning"],
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
        dataloader_num_workers=0,
        logging_steps=cfg["training"].get("logging_steps", 5),
        eval_steps=cfg["training"].get("eval_steps", 25),
        save_steps=cfg["training"].get("save_steps", 25),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        prediction_loss_only=False,
        dataloader_drop_last=False,
        group_by_length=False,
        length_column_name=None,
        include_inputs_for_metrics=False,
    )

    # Initialize trainer
    trainer = InstructBLIPPromptTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if cfg["training"]["should_train"] else None,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_accuracy_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                early_stopping_threshold=cfg["training"]["early_stopping_threshold"],
            )
        ] if cfg["training"]["should_train"] else [],
    )

    # Verify model setup before training
    logger.info("Verifying model setup...")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Number of trainable parameter tensors: {len(trainable_params)}")
    
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Cannot proceed with training.")

    # Training
    if cfg["training"]["should_train"]:
        logger.info("Starting training...")
        
        try:
            if args.resume:
                logger.info(f"Resuming from checkpoint: {args.resume}")
                trainer.train(resume_from_checkpoint=args.resume)
            else:
                trainer.train()
            
            # Save model
            logger.info("Saving final model...")
            trainer.save_model()
            processor.save_pretrained(out_dir)
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        metrics = trainer.evaluate()
        
        # Save metrics
        metrics_file = out_dir / "eval_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Final validation metrics: {metrics}")
        logger.info(f"Training completed successfully! Results saved to {out_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()