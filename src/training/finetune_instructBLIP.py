import argparse
import os
import random
import yaml
import json
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)

# Suppress tokenizers parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Is this image AI-generated? Answer 'yes' or 'no'."
YES, NO = "yes", "no"


class AINatureDataset(Dataset):
    def __init__(self, root: str, processor: InstructBlipProcessor):
        self.processor = processor
        self.samples = []
        for lbl_name, lbl_id in [("ai", 1), ("nature", 0)]:
            folder = Path(root) / lbl_name
            if not folder.exists():
                raise FileNotFoundError(f"{folder} not found")
            for p in folder.rglob("*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    self.samples.append((str(p), lbl_id))
        random.shuffle(self.samples)

        tok = processor.tokenizer
        self.answer_ids = {
            1: tok(YES, add_special_tokens=False).input_ids + [tok.eos_token_id],
            0: tok(NO,  add_special_tokens=False).input_ids + [tok.eos_token_id],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        answer_tok = self.answer_ids[label]
        text = f"{PROMPT} {YES if label == 1 else NO}"

        proc = self.processor(img, text, return_tensors="pt", truncation=True)
        proc = {k: v.squeeze(0) for k, v in proc.items()}

        # build labels: mask prompt tokens as -100
        labels = proc["input_ids"].clone()
        prompt_len = labels.shape[0] - len(answer_tok)
        labels[:prompt_len] = -100
        proc["labels"] = labels.long()
        return proc


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_lora(model: InstructBlipForConditionalGeneration, lora_cfg: Dict[str, Any]):
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    # wrap only the language_model module
    model.language_model = get_peft_model(model.language_model, cfg)
    # freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad_(False)
    return model


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    acc = (preds[:, 0] == labels[:, 0]).float().mean().item()
    return {"accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["data"].get("seed", 42))

    # load processor and model
    processor = InstructBlipProcessor.from_pretrained(cfg["model"]["name_pretrained"])
    model = InstructBlipForConditionalGeneration.from_pretrained(
        cfg["model"]["name_pretrained"],
        torch_dtype=torch.bfloat16 if cfg["training"]["use_bfloat16"] else torch.float16,
        low_cpu_mem_usage=True,
    )

    # apply LoRA
    if cfg["model"]["finetune_method"] == "lora":
        model = apply_lora(model, cfg["model"]["lora_params"])
    else:
        raise ValueError("Only LoRA fine-tune is implemented")

    # prepare datasets and collator
    train_ds = AINatureDataset(cfg["data"]["train_path"], processor)
    val_ds = AINatureDataset(cfg["data"]["val_test_path"], processor)
    collator = DataCollatorForSeq2Seq(processor.tokenizer, padding="longest")

    # output directory
    out_dir = Path(cfg["output"]["base_results_dir_root"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # training arguments
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate_lora"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        bf16=cfg["training"]["use_bfloat16"],
        gradient_checkpointing=False,  # disable to avoid in-place errors
        eval_strategy=cfg["training"]["evaluation_strategy"],
        save_strategy=cfg["training"]["save_strategy"],
        save_total_limit=cfg["training"]["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=[] if cfg["environment"]["disable_wandb"] else ["wandb"],
        dataloader_num_workers=4,
        logging_steps=10,
        seed=cfg["data"].get("seed", 42),
    )

    # Trainer with Early Stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if cfg["training"]["should_train"] else None,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg["training"]["early_stopping_patience"],
                early_stopping_threshold=cfg["training"]["early_stopping_threshold"],
            )
        ],
    )

    # train & save
    if cfg["training"]["should_train"]:
        trainer.train()
        trainer.save_model()

    # final evaluation
    metrics = trainer.evaluate()
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Validation metrics:", metrics)


if __name__ == "__main__":
    main()
