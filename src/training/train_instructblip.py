import os
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import contextlib
import yaml
import argparse
from torch.utils.data import Dataset

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    Trainer,
    TrainingArguments,
    GenerationConfig,
    AutoProcessor, 
    AutoModelForVision2Seq,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

# Define the prompts structure (can be reused or adapted)
USER_PROMPTS = [
    {
        "id": "ft_default", # Default prompt used during fine-tuning
        "question": "Is this image real or fake?",
        "options": {"real": "This is a real photograph.", "fake": "This is an AI-generated image."}
    },
    { 
        "id": "alt_1",
        "question": "Provide a verdict: authentic photograph or AI fabrication?",
        "options": {"real": "Authentic photograph.", "fake": "AI fabrication."}
    }
]

# --- 1. Dataset Class for Fine-tuning ---
class FinetuneImageDataset(Dataset):
    def __init__(self, image_paths, labels, processor, prompt, target_texts, max_length=128):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.prompt = prompt 
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
            return None 

        target_text = self.target_texts[label]
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt")
        
        labels_tokenized = self.processor.tokenizer(
            text=target_text,
            padding="max_length", 
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs["labels"] = labels_tokenized.input_ids.squeeze() 
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.ndim > 1 else v for k, v in inputs.items()}
        return inputs

def collate_fn_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Data Preparation Functions ---
def _load_images_from_path_typed(base_path: Path, category: str, file_extensions: list = None) -> list[Path]:
    if file_extensions is None:
        file_extensions = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    paths = []
    folder = base_path / category
    if not folder.is_dir():
        print(f"Warning: Directory not found: {folder}")
        return []
    for ext in file_extensions:
        paths.extend(list(folder.glob(f"*.{ext}")))
    return paths

def load_custom_train_val_test_data(
    train_base_path: Path,
    val_test_base_path: Path, 
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)

    train_real_paths = _load_images_from_path_typed(train_base_path, "nature")
    train_fake_paths = _load_images_from_path_typed(train_base_path, "ai")

    if not train_real_paths and not train_fake_paths:
        print(f"Error: No training images found in {train_base_path}/nature or {train_base_path}/ai.")
        return {"train_paths": [], "train_labels": [], "val_paths": [], "val_labels": [], "test_paths": [], "test_labels": []}

    num_train_real_to_sample = num_train_samples // 2
    num_train_fake_to_sample = num_train_samples - num_train_real_to_sample
    sampled_train_real = random.sample(train_real_paths, min(num_train_real_to_sample, len(train_real_paths)))
    sampled_train_fake = random.sample(train_fake_paths, min(num_train_fake_to_sample, len(train_fake_paths)))
    
    actual_train_samples = len(sampled_train_real) + len(sampled_train_fake)
    print(f"Requested {num_train_samples} training samples. Actual sampled: {actual_train_samples} (Real: {len(sampled_train_real)}, Fake: {len(sampled_train_fake)}) from {train_base_path}")

    train_paths = sampled_train_real + sampled_train_fake
    train_labels = ([0] * len(sampled_train_real)) + ([1] * len(sampled_train_fake))
    
    if train_paths: 
        train_combined = list(zip(train_paths, train_labels))
        random.shuffle(train_combined)
        train_paths, train_labels = zip(*train_combined)
    else: 
        train_paths, train_labels = list(train_paths), list(train_labels)

    val_test_all_real_paths = _load_images_from_path_typed(val_test_base_path, "nature")
    val_test_all_fake_paths = _load_images_from_path_typed(val_test_base_path, "ai")

    if not val_test_all_real_paths and not val_test_all_fake_paths:
         print(f"Warning: No validation/test images found in {val_test_base_path}/nature or {val_test_base_path}/ai.")

    random.shuffle(val_test_all_real_paths)
    random.shuffle(val_test_all_fake_paths)

    num_val_real_to_sample = num_val_samples // 2
    num_val_fake_to_sample = num_val_samples - num_val_real_to_sample
    sampled_val_real = val_test_all_real_paths[:min(num_val_real_to_sample, len(val_test_all_real_paths))]
    sampled_val_fake = val_test_all_fake_paths[:min(num_val_fake_to_sample, len(val_test_all_fake_paths))]
    
    actual_val_samples = len(sampled_val_real) + len(sampled_val_fake)
    print(f"Requested {num_val_samples} validation samples. Actual sampled: {actual_val_samples} (Real: {len(sampled_val_real)}, Fake: {len(sampled_val_fake)}) from {val_test_base_path}")

    val_paths = sampled_val_real + sampled_val_fake
    val_labels = ([0] * len(sampled_val_real)) + ([1] * len(sampled_val_fake))

    if val_paths:
        val_combined = list(zip(val_paths, val_labels))
        random.shuffle(val_combined) 
        val_paths, val_labels = zip(*val_combined)
    else:
        val_paths, val_labels = list(val_paths), list(val_labels)

    val_paths_set = set(map(str, val_paths)) 
    remaining_real_for_test = [p for p in val_test_all_real_paths if str(p) not in val_paths_set]
    remaining_fake_for_test = [p for p in val_test_all_fake_paths if str(p) not in val_paths_set]
    
    num_test_real_to_sample = num_test_samples // 2
    num_test_fake_to_sample = num_test_samples - num_test_real_to_sample
    sampled_test_real = random.sample(remaining_real_for_test, min(num_test_real_to_sample, len(remaining_real_for_test)))
    sampled_test_fake = random.sample(remaining_fake_for_test, min(num_test_fake_to_sample, len(remaining_fake_for_test)))

    actual_test_samples = len(sampled_test_real) + len(sampled_test_fake)
    print(f"Requested {num_test_samples} test samples. Actual sampled: {actual_test_samples} (Real: {len(sampled_test_real)}, Fake: {len(sampled_test_fake)}) from remaining images in {val_test_base_path}")

    test_paths = sampled_test_real + sampled_test_fake
    test_labels = ([0] * len(sampled_test_real)) + ([1] * len(sampled_test_fake))

    if test_paths:
        test_combined = list(zip(test_paths, test_labels))
        random.shuffle(test_combined) 
        test_paths, test_labels = zip(*test_combined)
    else:
        test_paths, test_labels = list(test_paths), list(test_labels)
        
    return {
        "train_paths": list(train_paths), "train_labels": list(train_labels),
        "val_paths": list(val_paths), "val_labels": list(val_labels),
        "test_paths": list(test_paths), "test_labels": list(test_labels),
    }

# --- Fine-tuning Functions ---
def finetune_model(
    model_name: str,
    train_dataset: FinetuneImageDataset,
    val_dataset: FinetuneImageDataset,
    output_dir: Path, # Where Trainer saves its checkpoints, including the best model
    logging_dir: Path,
    epochs: int = 3,
    batch_size: int = 4, 
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    early_stopping_patience: int = 10,
    early_stopping_threshold: float = 0.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f"\n--- Starting Full Fine-tuning for model: {model_name} ---")
    print(f"Using device: {device}")

    model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=str(logging_dir),
        logging_steps=10, 
        eval_strategy=evaluation_strategy if val_dataset is not None else "no",
        save_strategy=save_strategy, 
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_dataset is not None else False, 
        metric_for_best_model="loss" if val_dataset is not None else None, 
        greater_is_better=False, 
        fp16=False,
        report_to="tensorboard", 
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=collate_fn_skip_none,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)] if val_dataset is not None else [],
    )

    print("Starting full model training...")
    trainer.train()

    # The best model checkpoint is already saved by the Trainer due to load_best_model_at_end=True.
    # The path to this best model is output_dir/checkpoint-xxx (if save_strategy is by steps)
    # or output_dir (if save_strategy is epoch and it saves the best one there, or via save_model call)
    # Hugging Face Trainer by default saves the final best model to output_dir (the root) when load_best_model_at_end=True
    # Let's explicitly save the final model to a known subfolder "final_model" within output_dir.
    final_model_save_path = output_dir / "final_model"
    final_model_save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Training complete. Saving final model to: {final_model_save_path}")
    trainer.save_model(str(final_model_save_path)) # Saves model and tokenizer/processor if any
    
    # Ensure processor is saved if not already handled by save_model
    if hasattr(train_dataset, 'processor') and train_dataset.processor is not None:
        train_dataset.processor.save_pretrained(str(final_model_save_path))
        print(f"Processor saved to {final_model_save_path}")

    return str(final_model_save_path)


def finetune_model_with_lora(
    base_model_name: str,
    train_dataset: FinetuneImageDataset,
    val_dataset: FinetuneImageDataset,
    output_dir_checkpoints: Path, 
    final_lora_adapter_dir: Path, # Explicit directory for the final LoRA adapter
    logging_dir: Path,
    lora_config_params: dict, 
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4, 
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    evaluation_strategy: str = "epoch",
    save_strategy: str = "epoch",
    save_total_limit: int = 3,
    early_stopping_patience: int = 10,
    early_stopping_threshold: float = 0.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    print(f"\n--- Starting LoRA Fine-tuning for model: {base_model_name} ---")
    print(f"Using device: {device}")

    model = InstructBlipForConditionalGeneration.from_pretrained(base_model_name)
    
    lora_config = LoraConfig(
        r=lora_config_params.get("r", 16),
        lora_alpha=lora_config_params.get("lora_alpha", 32),
        target_modules=lora_config_params.get("target_modules", ["q", "v"]),
        lora_dropout=lora_config_params.get("lora_dropout", 0.05),
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM 
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 
    model.to(device) 

    training_args = TrainingArguments(
        output_dir=str(output_dir_checkpoints), 
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=str(logging_dir),
        logging_steps=20, 
        eval_strategy=evaluation_strategy if val_dataset is not None else "no",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_dataset is not None else False, 
        metric_for_best_model="loss" if val_dataset is not None else None,
        greater_is_better=False,
        fp16=False,
        report_to="tensorboard",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        data_collator=collate_fn_skip_none,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience, early_stopping_threshold=early_stopping_threshold)] if val_dataset is not None else [],
    )

    print("Starting LoRA training...")
    trainer.train()

    print("LoRA Training complete. Saving LoRA adapter...")
    final_lora_adapter_dir.mkdir(parents=True, exist_ok=True) 
    
    # Save only the LoRA adapter weights
    model.save_pretrained(str(final_lora_adapter_dir)) 
    
    # Save the processor used (important for inference consistency)
    if hasattr(train_dataset, 'processor') and train_dataset.processor is not None:
        train_dataset.processor.save_pretrained(str(final_lora_adapter_dir))
    print(f"LoRA adapter and processor saved to {final_lora_adapter_dir}")
    
    return str(final_lora_adapter_dir)

# --- Main Script Execution (Training) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune an InstructBLIP model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_instructblip_config.yaml",
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # --- Configuration (from YAML) ---
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    output_config = config.get('output', {})
    env_config = config.get('environment', {}) # Added for environment settings
    
    TRAIN_DATA_PATH = Path(data_config.get("train_path", "/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/train"))
    VAL_TEST_DATA_PATH = Path(data_config.get("val_test_path", "/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/val"))
    
    NUM_TRAIN_SAMPLES = data_config.get("num_train_samples", 100)    
    NUM_VAL_SAMPLES = data_config.get("num_val_samples", 50)     
    NUM_TEST_SAMPLES = data_config.get("num_test_samples", 50)    
    SEED = data_config.get("seed", 42)                 

    MODEL_NAME_PRETRAINED = model_config.get("name_pretrained", "Salesforce/instructblip-flan-t5-xl")
    FINETUNE_METHOD = model_config.get("finetune_method", "lora") 
    
    lora_params_config = model_config.get("lora_params", {})
    LORA_R = lora_params_config.get("r", 16) 
    LORA_ALPHA = lora_params_config.get("alpha", 32) 
    LORA_DROPOUT = lora_params_config.get("dropout", 0.05)
    LORA_TARGET_MODULES = lora_params_config.get("target_modules", ["q", "v"]) 

    SHOULD_TRAIN = training_config.get("should_train", True)
    EPOCHS = training_config.get("epochs", 3) 
    TRAIN_BATCH_SIZE = training_config.get("batch_size", 4) 
    
    LEARNING_RATE_FULL = training_config.get("learning_rate_full", 5e-5)
    LEARNING_RATE_LORA = training_config.get("learning_rate_lora", 1e-4)
    LEARNING_RATE = LEARNING_RATE_LORA if FINETUNE_METHOD == "lora" else LEARNING_RATE_FULL
    
    WARMUP_STEPS = training_config.get("warmup_steps", 50)
    WEIGHT_DECAY = training_config.get("weight_decay", 0.01)
    MAX_TOKEN_LENGTH_TARGET = training_config.get("max_target_token_length", 64)
    PROMPT_CONFIG_IDX = training_config.get("prompt_config_idx", 0)
    EVALUATION_STRATEGY = training_config.get("evaluation_strategy", "epoch")
    SAVE_STRATEGY = training_config.get("save_strategy", "epoch")
    SAVE_TOTAL_LIMIT = training_config.get("save_total_limit", 3)
    EARLY_STOPPING_PATIENCE = training_config.get("early_stopping_patience", 10)
    EARLY_STOPPING_THRESHOLD = training_config.get("early_stopping_threshold", 0.0)

    # GPU Configuration from environment settings
    specified_gpu_id = env_config.get("gpu_id", None) 

    BASE_RESULTS_DIR_ROOT = Path(output_config.get("base_results_dir_root", "results/instructblip_finetune"))
    
    # --- Dynamic RUN_ID and Output Paths ---
    run_id_parts = [
        FINETUNE_METHOD,
        f"train{NUM_TRAIN_SAMPLES}",
        f"val{NUM_VAL_SAMPLES}",
        f"test{NUM_TEST_SAMPLES}", 
        MODEL_NAME_PRETRAINED.split('/')[-1]
    ]
    if FINETUNE_METHOD == "lora":
        run_id_parts.append(f"lora_r{LORA_R}_alpha{LORA_ALPHA}")
    run_id_parts.append(f"seed{SEED}")
    RUN_ID = "_".join(run_id_parts)
        
    BASE_RESULTS_DIR = BASE_RESULTS_DIR_ROOT / RUN_ID
    TRAINER_CHECKPOINTS_DIR = BASE_RESULTS_DIR / "trainer_checkpoints" 
    FINAL_MODEL_ARTIFACTS_DIR = BASE_RESULTS_DIR / "final_model_artifacts" 
    LOGGING_DIR = BASE_RESULTS_DIR / "training_logs"
    
    # --- Prompt Configuration ---
    if not (0 <= PROMPT_CONFIG_IDX < len(USER_PROMPTS)):
        print(f"Error: prompt_config_idx {PROMPT_CONFIG_IDX} is out of range for USER_PROMPTS (len: {len(USER_PROMPTS)}). Using default index 0.")
        PROMPT_CONFIG_IDX = 0
    FINETUNE_PROMPT_CONFIG = USER_PROMPTS[PROMPT_CONFIG_IDX] 

    FINETUNE_QUESTION = FINETUNE_PROMPT_CONFIG["question"]
    FINETUNE_TARGET_TEXTS = {
        0: FINETUNE_PROMPT_CONFIG["options"]["real"], 
        1: FINETUNE_PROMPT_CONFIG["options"]["fake"]  
    }
    
    # Determine device based on configuration and availability
    if specified_gpu_id is not None and isinstance(specified_gpu_id, int):
        if specified_gpu_id == -1:
            device = torch.device("cpu")
            print("Configuration explicitly set to use CPU.")
        elif torch.cuda.is_available():
            if 0 <= specified_gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{specified_gpu_id}")
                torch.cuda.set_device(device) # Set default CUDA device
                print(f"Using specified GPU ID: {specified_gpu_id}")
            else:
                print(f"Warning: Specified GPU ID {specified_gpu_id} is out of range (0-{torch.cuda.device_count()-1}). Using default GPU (cuda:0) if available, otherwise CPU.")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            print(f"Warning: Specified GPU ID {specified_gpu_id} but CUDA is not available. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
             print(f"No GPU ID specified or invalid value, using default available GPU: cuda:{torch.cuda.current_device()}")
        else:
             print("No GPU ID specified or invalid value, CUDA not available. Using CPU.")

    print(f"Starting training script for RUN_ID: {RUN_ID}")
    print(f"Configuration loaded from: {args.config}")
    print(f"Fine-tuning method: {FINETUNE_METHOD}")
    print(f"Model artifacts will be saved to: {FINAL_MODEL_ARTIFACTS_DIR}")
    print(f"Using device: {device}")

    BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Preparing Data ---")
    data_splits = load_custom_train_val_test_data(
        train_base_path=TRAIN_DATA_PATH,
        val_test_base_path=VAL_TEST_DATA_PATH,
        num_train_samples=NUM_TRAIN_SAMPLES,
        num_val_samples=NUM_VAL_SAMPLES,
        num_test_samples=NUM_TEST_SAMPLES, # Ensure test data is accounted for in split
        seed=SEED
    )
    
    if not data_splits['train_paths']:
        print("Error: No training data loaded. Exiting."); exit()

    print("\n--- Initializing Processor and Creating Datasets ---")
    try:
        processor = InstructBlipProcessor.from_pretrained(MODEL_NAME_PRETRAINED)
        train_dataset = FinetuneImageDataset(
            image_paths=data_splits['train_paths'], 
            labels=data_splits['train_labels'], 
            processor=processor, prompt=FINETUNE_QUESTION, 
            target_texts=FINETUNE_TARGET_TEXTS, max_length=MAX_TOKEN_LENGTH_TARGET
        )
        
        val_dataset = None
        if data_splits['val_paths']:
            val_dataset = FinetuneImageDataset(
                image_paths=data_splits['val_paths'], 
                labels=data_splits['val_labels'], 
                processor=processor, prompt=FINETUNE_QUESTION, 
                target_texts=FINETUNE_TARGET_TEXTS, max_length=MAX_TOKEN_LENGTH_TARGET
            )
            print(f"Train dataset: {len(train_dataset)} samples. Val dataset: {len(val_dataset)} samples.")
        else:
            print(f"Train dataset: {len(train_dataset)} samples. No validation dataset created.")

    except Exception as e:
        print(f"Error initializing processor or datasets: {e}. Exiting."); exit()

    if SHOULD_TRAIN:
        print(f"\n--- Starting Fine-tuning (Method: {FINETUNE_METHOD}) --- ")
        TRAINER_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True) 
        LOGGING_DIR.mkdir(parents=True, exist_ok=True)

        if FINETUNE_METHOD == "full":
            trained_model_path = finetune_model(
                model_name=MODEL_NAME_PRETRAINED,
                train_dataset=train_dataset, val_dataset=val_dataset,
                output_dir=FINAL_MODEL_ARTIFACTS_DIR, # Save final model directly to artifacts dir
                logging_dir=LOGGING_DIR,
                epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS, weight_decay=WEIGHT_DECAY, device=device,
                evaluation_strategy=EVALUATION_STRATEGY,
                save_strategy=SAVE_STRATEGY,
                save_total_limit=SAVE_TOTAL_LIMIT,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD
            )
            print(f"Full Fine-tuning complete. Model artifacts saved to: {trained_model_path}")
        
        elif FINETUNE_METHOD == "lora":
            lora_params = {
                "r": LORA_R, "lora_alpha": LORA_ALPHA, 
                "lora_dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES
            }
            adapter_path = finetune_model_with_lora(
                base_model_name=MODEL_NAME_PRETRAINED,
                train_dataset=train_dataset, val_dataset=val_dataset,
                output_dir_checkpoints=TRAINER_CHECKPOINTS_DIR, 
                final_lora_adapter_dir=FINAL_MODEL_ARTIFACTS_DIR,
                logging_dir=LOGGING_DIR, lora_config_params=lora_params,
                epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, learning_rate=LEARNING_RATE,
                warmup_steps=WARMUP_STEPS, weight_decay=WEIGHT_DECAY, device=device,
                evaluation_strategy=EVALUATION_STRATEGY,
                save_strategy=SAVE_STRATEGY,
                save_total_limit=SAVE_TOTAL_LIMIT,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD
            )
            print(f"LoRA Fine-tuning complete. Adapter saved to: {adapter_path}")
    else:
        print("SHOULD_TRAIN is False. Skipping training.")
        if not FINAL_MODEL_ARTIFACTS_DIR.exists() or not any(FINAL_MODEL_ARTIFACTS_DIR.iterdir()):
            print(f"Warning: Skipping training, but no existing artifacts found in {FINAL_MODEL_ARTIFACTS_DIR}")


    print("\n--- Training Script Finished ---") 