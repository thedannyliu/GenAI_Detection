import argparse
import os
import random
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import functools

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType
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
    def __init__(self, root: str, processor: InstructBlipProcessor, max_samples: Optional[int] = None):
        self.processor = processor
        self.samples = []
        
        # 支援多種資料夾命名方式 (ai/nature、1_fake/0_real、fake/real)
        label_dirs = {
            1: ["ai", "1_fake", "fake"],   # AI 生成影像
            0: ["nature", "0_real", "real"],  # 真實影像
        }

        for lbl_id, dir_names in label_dirs.items():
            found_any = False
            for dir_name in dir_names:
                folder = Path(root) / dir_name
                if not folder.exists():
                    continue  # 該命名不存在，繼續尋找其他別名

                found_any = True
                folder_samples = []
                for p in folder.rglob("*"):
                    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                        folder_samples.append((str(p), lbl_id))

                logger.info(f"Found {len(folder_samples)} samples in {folder}")
                self.samples.extend(folder_samples)

            if not found_any:
                logger.warning(
                    f"No folder found for label {lbl_id} under aliases {dir_names} in {root}"
                )
        
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
        
        # 分別處理輸入提示和完整回答
        input_text = PROMPT
        target_text = answer_text
        
        # 處理圖像和輸入文字
        inputs = self.processor(
            images=img,
            text=input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        
        # 為目標答案創建標籤
        target_inputs = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            padding=False,
            add_special_tokens=False
        )
        
        # Remove batch dimension
        processed_inputs = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                processed_inputs[k] = v.squeeze(0)
            else:
                processed_inputs[k] = v
        
        # 創建標籤 - 只有答案部分需要計算損失
        input_ids = processed_inputs["input_ids"]
        target_ids = target_inputs["input_ids"].squeeze(0)
        
        # 將輸入和目標拼接
        full_input_ids = torch.cat([input_ids, target_ids], dim=0)
        
        # 創建對應的attention mask
        full_attention_mask = torch.ones_like(full_input_ids)
        
        # 創建標籤：輸入部分用-100屏蔽，只對答案部分計算損失
        labels = torch.full_like(full_input_ids, -100)
        labels[len(input_ids):] = target_ids  # 只有答案部分的標籤
        
        processed_inputs.update({
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels
        })
        
        # 處理其他可能的輸入
        if "qformer_input_ids" in processed_inputs:
            qformer_attention_mask = torch.ones_like(processed_inputs["qformer_input_ids"])
            processed_inputs["qformer_attention_mask"] = qformer_attention_mask
        
        return processed_inputs


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_lora_correctly(model: InstructBlipForConditionalGeneration, lora_cfg: Dict[str, Any]):
    """Apply LoRA fine-tuning with proper gradient setup."""
    
    logger.info("Setting up LoRA configuration...")
    
    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_rslora=False,
    )
    
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA only to the language model
    logger.info("Applying LoRA to language model...")
    model.language_model = get_peft_model(model.language_model, lora_config)
    
    # Enable training mode for LoRA model
    model.language_model.train()
    
    # Explicitly enable gradients for LoRA parameters
    lora_param_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name.lower():
            param.requires_grad = True
            lora_param_count += param.numel()
            logger.debug(f"Enabled gradients for: {name} - shape: {param.shape}")
    
    # Verify gradient setup
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"LoRA parameters found: {lora_param_count:,}")
    logger.info(f"Total trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! LoRA setup failed.")
    
    return model


class InstructBLIPDataCollator:
    """Optimized data collator for InstructBLIP."""
    
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        # Ensure pad token is properly set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, features):
        batch = {}
        
        # Collect all tensor fields
        tensor_fields = ["input_ids", "attention_mask", "qformer_input_ids", 
                        "qformer_attention_mask", "labels"]
        
        for field in tensor_fields:
            if field in features[0]:
                tensors = [f[field] for f in features]
                if field == "labels":
                    # Pad labels with -100
                    padded = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=-100
                    )
                else:
                    # Pad other sequences with appropriate padding values
                    pad_value = self.tokenizer.pad_token_id if "input_ids" in field else 0
                    padded = torch.nn.utils.rnn.pad_sequence(
                        tensors, batch_first=True, padding_value=pad_value
                    )
                batch[field] = padded
        
        # Handle pixel values (stack them)
        if "pixel_values" in features[0]:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
        return batch


def compute_accuracy_metrics(eval_preds, tokenizer):
    """計算模型預測準確率的修正版本，基於解碼後的文字比較"""
    predictions, labels = eval_preds
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # 處理預測結果 (logits)
    if predictions.ndim == 3:  # [batch, seq_len, vocab_size]
        predicted_ids_all = np.argmax(predictions, axis=-1) # 取得所有位置的預測 token ID
    else: # 如果直接是 token IDs
        predicted_ids_all = predictions

    # 獲取真實標籤 (排除 -100 的部分)
    # labels 的形狀也是 [batch, seq_len]

    batch_size = labels.shape[0]
    sample_accuracies = []

    for i in range(batch_size):
        # 取得單個樣本的真實標籤和預測
        label_ids_sample = labels[i]
        pred_ids_sample = predicted_ids_all[i]

        # 找出標籤中答案的部分 (非 -100)
        actual_answer_token_ids = label_ids_sample[label_ids_sample != -100]
        
        if len(actual_answer_token_ids) == 0:
            # 如果真實答案為空 (不應該發生在我們的案例中)
            sample_accuracies.append(0.0) # 視為不正確
            logger.debug(f"Sample {i}: No valid actual answer tokens found.")
            continue

        # 找出預測中對應答案位置的 token
        # 我們需要知道答案在完整序列中的起始位置
        # 在 AINatureDataset 中，labels 的 -100 部分對應輸入提示，非 -100 部分對應答案
        # 所以，答案的長度是 len(actual_answer_token_ids)
        # 答案的起始索引是 input_ids 的長度，結束索引是 input_ids 長度 + 答案長度
        
        # 預測的答案 token IDs
        # 我們需要找到與 actual_answer_token_ids 長度相同的部分來比較
        # 假設答案總是在序列的末尾，長度與 actual_answer_token_ids 相同
        # 這是基於我們在 Dataset 中的構造方式：input_ids + target_ids -> full_input_ids
        # labels 也是基於 full_input_ids，其中提示部分被 mask
        # 因此，預測的 logits (或 ids) 也對應 full_input_ids 的結構
        
        # 找到有效標籤的起始和結束位置
        valid_label_indices = np.where(label_ids_sample != -100)[0]
        if len(valid_label_indices) == 0:
            sample_accuracies.append(0.0)
            logger.debug(f"Sample {i}: No valid label indices (all -100).")
            continue
            
        start_answer_idx = valid_label_indices[0]
        end_answer_idx = valid_label_indices[-1] + 1 # 切片不包含尾部
        
        # 從預測中提取對應答案位置的 token IDs
        predicted_answer_token_ids = pred_ids_sample[start_answer_idx:end_answer_idx]

        # 解碼真實答案和預測答案
        # 使用 skip_special_tokens=True 來移除如 <eos> 等特殊 token
        actual_answer_text = tokenizer.decode(actual_answer_token_ids, skip_special_tokens=True).strip().lower()
        predicted_answer_text = tokenizer.decode(predicted_answer_token_ids, skip_special_tokens=True).strip().lower()
        
        # 確保在評估時打印詳細的樣本比較信息，但僅限前5個樣本
        if i < 5:
            logger.info(f"Sample {i}: Actual Text='{actual_answer_text}', Predicted Text='{predicted_answer_text}'")
            logger.info(f"  Actual Tokens: {actual_answer_token_ids.tolist()}")
            logger.info(f"  Predicted Tokens: {predicted_answer_token_ids.tolist()}")

        # 比較解碼後的文字
        if actual_answer_text == predicted_answer_text and actual_answer_text in [YES, NO]:
            sample_accuracies.append(1.0)
        else:
            sample_accuracies.append(0.0)

    if sample_accuracies:
        accuracy = np.mean(sample_accuracies)
    else:
        accuracy = 0.0
        logger.warning("No samples processed for accuracy computation in compute_accuracy_metrics.")
    
    logger.info(f"Computed accuracy by text comparison: {accuracy:.4f} from {len(sample_accuracies)} samples")
    return {"accuracy": float(accuracy)}


class InstructBLIPTrainer(Trainer):
    """Custom trainer for InstructBLIP with proper loss and cache handling."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """計算損失，確保只對答案部分計算損失"""
        # Ensure model is in training mode
        model.train()
        
        # Forward pass
        outputs = model(**inputs)
        
        # Extract loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Manual loss computation
            logits = outputs.logits
            labels = inputs["labels"]
            
            # Flatten for loss computation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss only on non-masked tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """執行預測步驟，正確處理DynamicCache對象"""
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
                    cleaned_outputs = {}
                    for key, value in outputs.items():
                        if key not in ignore_keys and not key.endswith('_cache'):
                            if isinstance(value, torch.Tensor):
                                cleaned_outputs[key] = value
                    logits = cleaned_outputs.get("logits")
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
            else:
                loss = None
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    cleaned_outputs = {}
                    for key, value in outputs.items():
                        if key not in ignore_keys and not key.endswith('_cache'):
                            if isinstance(value, torch.Tensor):
                                cleaned_outputs[key] = value
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

    # Apply LoRA fine-tuning
    if cfg["model"]["finetune_method"] == "lora":
        logger.info("Applying LoRA fine-tuning...")
        model = apply_lora_correctly(model, cfg["model"]["lora_params"])
    else:
        raise ValueError("Only LoRA fine-tuning is currently supported")

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
            if key == "labels":
                non_masked = value[value != -100]
                logger.info(f"  Non-masked labels: {non_masked}")
                logger.info(f"  Label tokens: {processor.tokenizer.decode(non_masked)}")
    
    # Data collator
    collator = InstructBLIPDataCollator(processor)

    # Output directory
    out_dir = Path(cfg["output"]["base_results_dir_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Training arguments with corrected settings
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
        gradient_checkpointing=False,
        eval_strategy=cfg["training"]["evaluation_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[] if cfg["environment"]["disable_wandb"] else ["wandb"],
        dataloader_num_workers=0,
        logging_steps=cfg["training"].get("logging_steps", 100),
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
        # 添加重要參數來改善訓練穩定性
        max_grad_norm=1.0,  # 梯度裁剪
        lr_scheduler_type="cosine",  # 使用餘弦學習率調度
    )

    # Initialize trainer
    trainer = InstructBLIPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if cfg["training"]["should_train"] else None,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=functools.partial(compute_accuracy_metrics, tokenizer=processor.tokenizer),
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