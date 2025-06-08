import argparse
import logging
import os
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from peft import get_peft_model, PromptTuningConfig, TaskType
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Is this image AI-generated? Answer 'yes' or 'no'."
YES, NO = "yes", "no"


class AINatureDataset(Dataset):
    def __init__(self, root: str, processor: InstructBlipProcessor, max_samples: Optional[int] = None):
        self.processor = processor
        self.samples = []
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
        full_text = f"{PROMPT} {answer_text}"
        inputs = self.processor(
            images=img,
            text=full_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )
        processed_inputs = {}
        for k, v in inputs.items():
            processed_inputs[k] = v.squeeze(0) if torch.is_tensor(v) else v
        input_ids = processed_inputs["input_ids"]
        answer_tokens = self.processor.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels = input_ids.clone()
        answer_start = len(input_ids) - len(answer_tokens)
        if answer_start > 0:
            labels[:answer_start] = -100
        processed_inputs["labels"] = labels
        return processed_inputs


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_prompt_tuning(model: InstructBlipForConditionalGeneration, pt_cfg: Dict[str, Any], tokenizer_path: str):
    for param in model.parameters():
        param.requires_grad = False
    config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=pt_cfg["num_virtual_tokens"],
        tokenizer_name_or_path=tokenizer_path,
        prompt_tuning_init_text=pt_cfg.get("init_text"),
    )
    model.language_model = get_peft_model(model.language_model, config)
    model.language_model.print_trainable_parameters()
    return model


class InstructBLIPDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features):
        batch = {}
        tensor_fields = ["input_ids", "attention_mask", "qformer_input_ids", "qformer_attention_mask", "labels"]
        for field in tensor_fields:
            if field in features[0]:
                tensors = [f[field] for f in features]
                if field == "labels":
                    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=-100)
                else:
                    pad_value = self.tokenizer.pad_token_id if "input_ids" in field else 0
                    padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=pad_value)
                batch[field] = padded
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        return batch


def compute_accuracy_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == 3:
        predicted_ids = np.argmax(predictions, axis=-1)
    else:
        predicted_ids = predictions
    valid_positions = labels != -100
    if valid_positions.sum() == 0:
        return {"accuracy": 0.0}
    correct = (predicted_ids == labels) & valid_positions
    accuracy = correct.sum() / valid_positions.sum()
    return {"accuracy": float(accuracy)}


class InstructBLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model.train()
        outputs = model(**inputs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.logits
            labels = inputs["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        ignore_keys = list(ignore_keys) if ignore_keys else []
        cache_keys = ["past_key_values", "cache", "past", "mems"]
        ignore_keys.extend(cache_keys)
        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    cleaned_outputs = {k: v for k, v in outputs.items() if k not in ignore_keys and not k.endswith('_cache') and isinstance(v, torch.Tensor)}
                    logits = cleaned_outputs.get("logits")
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    cleaned_outputs = {k: v for k, v in outputs.items() if k not in ignore_keys and not k.endswith('_cache') and isinstance(v, torch.Tensor)}
                    logits = cleaned_outputs.get("logits")
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
        if prediction_loss_only:
            return (loss, None, None)
        if logits is not None:
            logits = logits.detach()
        labels = None
        if has_labels:
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.detach()
        return (loss, logits, labels)


def main():
    parser = argparse.ArgumentParser(description="Prompt tune InstructBLIP")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["data"].get("seed", 42))

    processor = InstructBlipProcessor.from_pretrained(cfg["model"]["name_pretrained"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = InstructBlipForConditionalGeneration.from_pretrained(
        cfg["model"]["name_pretrained"],
        torch_dtype=torch.bfloat16 if cfg["training"]["use_bfloat16"] else torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )

    if cfg["model"]["finetune_method"] != "prompt_tuning":
        raise ValueError("finetune_method must be 'prompt_tuning'")
    model = apply_prompt_tuning(model, cfg["model"]["prompt_tuning_params"], cfg["model"]["name_pretrained"])

    train_ds = AINatureDataset(
        cfg["data"]["train_path"], processor, max_samples=cfg["data"].get("num_train_samples")
    )
    val_ds = AINatureDataset(
        cfg["data"]["val_test_path"], processor, max_samples=cfg["data"].get("num_val_samples")
    )

    collator = InstructBLIPDataCollator(processor)

    out_dir = Path(cfg["output"]["base_results_dir_root"])
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"].get("eval_batch_size", 1),
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate_prompt"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["use_bfloat16"],
        fp16=not cfg["training"]["use_bfloat16"],
        eval_strategy=cfg["training"]["evaluation_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[] if cfg["environment"]["disable_wandb"] else ["wandb"],
        dataloader_num_workers=0,
        logging_steps=cfg["training"].get("logging_steps", 10),
        eval_steps=cfg["training"].get("eval_steps", 500),
        save_steps=cfg["training"].get("save_steps", 500),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        prediction_loss_only=False,
        dataloader_drop_last=False,
        group_by_length=False,
        length_column_name=None,
        include_inputs_for_metrics=False,
    )

    trainer = InstructBLIPTrainer(
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

    if cfg["training"]["should_train"]:
        logger.info("Starting training...")
        if args.resume:
            trainer.train(resume_from_checkpoint=args.resume)
        else:
            trainer.train()
        trainer.save_model()
        processor.save_pretrained(out_dir)

    logger.info("Running final evaluation...")
    metrics = trainer.evaluate()
    metrics_file = out_dir / "eval_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final validation metrics: {metrics}")


if __name__ == "__main__":
    main()
