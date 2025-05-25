import os
import random
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split # Not strictly needed for eval but load_custom might use it
import matplotlib.pyplot as plt
import seaborn as sns
import json
import contextlib

from transformers import (
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    GenerationConfig,
    AutoProcessor, 
    AutoModelForVision2Seq,
)

from peft import PeftModel

# Define the prompts structure (can be reused or adapted)
USER_PROMPTS = [
    {
        "id": "ft_default", 
        "question": "Is this image real or fake?",
        "options": {"real": "This is a real photograph.", "fake": "This is an AI-generated image."}
    },
    { 
        "id": "alt_1",
        "question": "Provide a verdict: authentic photograph or AI fabrication?",
        "options": {"real": "Authentic photograph.", "fake": "AI fabrication."}
    }
]

# --- Data Loading (copied from training script for now, can be moved to utils) ---
def _load_images_from_path_typed(base_path: Path, category: str, file_extensions: list = None) -> list[Path]:
    if file_extensions is None:
        file_extensions = ["jpg", "png", "jpeg"]
    paths = []
    folder = base_path / category
    if not folder.is_dir():
        print(f"Warning: Directory not found: {folder}")
        return []
    for ext in file_extensions:
        paths.extend(list(folder.glob(f"*.{ext}")))
    return paths

def load_custom_train_val_test_data(
    train_base_path: Path, # Not used by eval script directly for loading, but part of the structure
    val_test_base_path: Path, 
    num_train_samples: int, # Not used by eval script directly for loading
    num_val_samples: int,   # Not used by eval script directly for loading
    num_test_samples: int,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    # In evaluation, we are primarily interested in the test set.
    # We need to replicate how the test set was created during training to ensure consistency.
    # This means we need to simulate the sampling of train and val to get the correct remaining test images.

    # Load all potential val/test images
    val_test_all_real_paths = _load_images_from_path_typed(val_test_base_path, "nature")
    val_test_all_fake_paths = _load_images_from_path_typed(val_test_base_path, "ai")

    if not val_test_all_real_paths and not val_test_all_fake_paths:
         print(f"Error: No validation/test images found in {val_test_base_path}/nature or {val_test_base_path}/ai for evaluation.")
         return {"train_paths": [], "train_labels": [], "val_paths": [], "val_labels": [], "test_paths": [], "test_labels": []}

    random.shuffle(val_test_all_real_paths) # Shuffle to replicate randomness of val selection
    random.shuffle(val_test_all_fake_paths)

    # Simulate val sampling to correctly identify test images
    num_val_real_to_sample = num_val_samples // 2
    num_val_fake_to_sample = num_val_samples - num_val_real_to_sample
    
    # These are the paths that WOULD have been used for validation
    _ = val_test_all_real_paths[:min(num_val_real_to_sample, len(val_test_all_real_paths))]
    _ = val_test_all_fake_paths[:min(num_val_fake_to_sample, len(val_test_all_fake_paths))]
    simulated_val_paths = _ + _ 
    val_paths_set = set(map(str, simulated_val_paths))

    # Identify remaining images for the test set from the *original* val_test_all lists before slicing for val
    # Re-seed before sampling for test set to ensure consistency if called multiple times for test only
    random.seed(seed) # Ensure consistent sampling for test set
    np.random.seed(seed)
    
    # Get all images again and shuffle them in the same way to ensure test set is derived correctly
    # This re-shuffling with the same seed ensures that if we sample for val, the remainder for test is consistent.
    val_test_all_real_paths_for_test_derivation = _load_images_from_path_typed(val_test_base_path, "nature")
    val_test_all_fake_paths_for_test_derivation = _load_images_from_path_typed(val_test_base_path, "ai")
    random.shuffle(val_test_all_real_paths_for_test_derivation)
    random.shuffle(val_test_all_fake_paths_for_test_derivation)

    # The validation paths that would have been chosen
    val_chosen_real = val_test_all_real_paths_for_test_derivation[:min(num_val_real_to_sample, len(val_test_all_real_paths_for_test_derivation))]
    val_chosen_fake = val_test_all_fake_paths_for_test_derivation[:min(num_val_fake_to_sample, len(val_test_all_fake_paths_for_test_derivation))]
    val_paths_chosen_set = set(map(str, val_chosen_real + val_chosen_fake))

    # Test paths are those not in the chosen val set
    remaining_real_for_test = [p for p in val_test_all_real_paths_for_test_derivation if str(p) not in val_paths_chosen_set]
    remaining_fake_for_test = [p for p in val_test_all_fake_paths_for_test_derivation if str(p) not in val_paths_chosen_set]
        
    num_test_real_to_sample = num_test_samples // 2
    num_test_fake_to_sample = num_test_samples - num_test_real_to_sample
    
    # Now sample the test set from these remaining images
    # Re-seed again just before sampling for the test set itself IF this function might be called
    # multiple times in a complex pipeline and we need this specific sampling to be isolated.
    # However, since we derive from the remainder, the previous shuffle of val_test_all_* lists is key.
    # For simplicity, the global seed at the start of the function should suffice if this function is called once for a given seed.

    sampled_test_real = random.sample(remaining_real_for_test, min(num_test_real_to_sample, len(remaining_real_for_test)))
    sampled_test_fake = random.sample(remaining_fake_for_test, min(num_test_fake_to_sample, len(remaining_fake_for_test)))

    actual_test_samples = len(sampled_test_real) + len(sampled_test_fake)
    print(f"(Eval) Requested {num_test_samples} test samples. Actual sampled: {actual_test_samples} (Real: {len(sampled_test_real)}, Fake: {len(sampled_test_fake)}) from remaining images in {val_test_base_path}")

    test_paths = sampled_test_real + sampled_test_fake
    test_labels = ([0] * len(sampled_test_real)) + ([1] * len(sampled_test_fake))

    if test_paths:
        test_combined = list(zip(test_paths, test_labels))
        random.shuffle(test_combined) 
        test_paths, test_labels = zip(*test_combined)
    else:
        test_paths, test_labels = list(test_paths), list(test_labels)
        
    # Return only test paths for evaluation script
    return {"test_paths": list(test_paths), "test_labels": list(test_labels)}


# --- Inference Functions ---
def run_inference_on_finetuned_model(
    finetuned_model_path: str, 
    inference_image_paths: list,
    inference_labels: list,
    prompts_config_for_inference: list, 
    results_dir: Path,
    device: torch.device,
    batch_size: int = 8, 
    max_new_tokens_inf: int = 70 
):
    print(f"\n--- Running Inference on Fine-tuned Model from: {finetuned_model_path} ---")
    results_dir.mkdir(parents=True, exist_ok=True)
    try:
        processor = AutoProcessor.from_pretrained(finetuned_model_path)
        model = AutoModelForVision2Seq.from_pretrained(finetuned_model_path)
        model.to(device)
        model.eval() 
    except Exception as e:
        print(f"Error loading fine-tuned model or processor from {finetuned_model_path}: {e}")
        return None

    all_detailed_results = {}
    num_images = len(inference_image_paths)
    for i in tqdm(range(0, num_images, batch_size), desc="Processing images for inference"):
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
                print(f"Error opening image {img_path_str}: {e}. Skipping this image.")
                path_key = str(img_path_str) 
                all_detailed_results[path_key] = {
                    "true_label": batch_labels[idx] if idx < len(batch_labels) else -1, 
                    "model_outputs": {prompt_conf["id"]: {"score": 0.5, "raw_output": f"Image load error: {e}", "error": True}
                                      for prompt_conf in prompts_config_for_inference}
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
                generated_ids = model.generate(**inputs, generation_config=generation_config)
            raw_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for batch_idx, raw_text in enumerate(raw_generated_texts):
                img_path_str_current = original_batch_paths_valid[batch_idx]
                true_label_current = original_batch_labels_valid[batch_idx]
                path_key = str(img_path_str_current)
                if path_key not in all_detailed_results: all_detailed_results[path_key] = {"true_label": true_label_current, "model_outputs": {}}
                cleaned_text = raw_text
                if question and question in raw_text: cleaned_text = raw_text.replace(question, "").strip()
                generated_text_lower = cleaned_text.lower().strip()
                score = 0.5 
                target_real_text_lower = prompt_conf["options"]["real"].lower()
                target_fake_text_lower = prompt_conf["options"]["fake"].lower()
                found_fake_keywords = ["fake", "artificial", "generated", "synthetic", "ai-generated", "computer-generated", "not real", "isn\'t real"]
                found_real_keywords = ["real", "authentic", "photograph", "genuine", "natural", "actual", "not fake", "isn\'t fake"]
                has_fake = any(keyword in generated_text_lower for keyword in found_fake_keywords) or (target_fake_text_lower in generated_text_lower)
                has_real = any(keyword in generated_text_lower for keyword in found_real_keywords) or (target_real_text_lower in generated_text_lower)
                if "not real" in generated_text_lower or "isn\'t real" in generated_text_lower: score = 1.0
                elif "not fake" in generated_text_lower or "isn\'t fake" in generated_text_lower: score = 0.0
                elif has_fake and not has_real: score = 1.0
                elif has_real and not has_fake: score = 0.0
                elif has_fake and has_real:
                    if target_real_text_lower in generated_text_lower and target_fake_text_lower not in generated_text_lower: score = 0.0
                    elif target_fake_text_lower in generated_text_lower and target_real_text_lower not in generated_text_lower: score = 1.0
                    else: score = 0.75 
                elif not generated_text_lower or generated_text_lower in ["unspecified", "unknown", "i cannot determine", "i can\'t tell"]: score = 0.5
                all_detailed_results[path_key]["model_outputs"][prompt_id] = {"score": score, "raw_output": raw_text, "cleaned_output": cleaned_text, "parsed_score": score, "error": False}
    detailed_results_path = results_dir / "inference_detailed_results.json"
    with open(detailed_results_path, 'w') as f: json.dump(all_detailed_results, f, indent=4)
    print(f"Inference detailed results saved to {detailed_results_path}")
    post_process_inference_results(all_detailed_results, prompts_config_for_inference, results_dir, model_name_for_report="evaluated_instructblip_full")
    return all_detailed_results

def run_inference_on_lora_model(
    base_model_name: str, 
    lora_adapter_path: str, 
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
        base_model = InstructBlipForConditionalGeneration.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model.to(device)
        model.eval() 
        processor = AutoProcessor.from_pretrained(lora_adapter_path) 
    except Exception as e:
        print(f"Error loading base model, LoRA adapter, or processor: {e}"); return None
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
                all_detailed_results[path_key] = {"true_label": batch_labels[idx] if idx < len(batch_labels) else -1, "model_outputs": {p_conf["id"]: {"score": 0.5, "raw_output": f"Image load error: {e}", "error": True} for p_conf in prompts_config_for_inference}}
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
                generated_ids = model.generate(**inputs, generation_config=generation_config)
            raw_generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for batch_idx, raw_text in enumerate(raw_generated_texts):
                img_path_str_current = original_batch_paths_valid[batch_idx]
                true_label_current = original_batch_labels_valid[batch_idx]
                path_key = str(img_path_str_current)
                if path_key not in all_detailed_results: all_detailed_results[path_key] = {"true_label": true_label_current, "model_outputs": {}}
                cleaned_text = raw_text
                if question and question in raw_text: cleaned_text = raw_text.replace(question, "").strip()
                generated_text_lower = cleaned_text.lower().strip()
                score = 0.5
                target_real_text_lower = prompt_conf["options"]["real"].lower()
                target_fake_text_lower = prompt_conf["options"]["fake"].lower()
                found_fake_keywords = ["fake", "artificial", "generated", "synthetic", "ai-generated", "computer-generated", "not real", "isn\'t real"]
                found_real_keywords = ["real", "authentic", "photograph", "genuine", "natural", "actual", "not fake", "isn\'t fake"]
                has_fake = any(keyword in generated_text_lower for keyword in found_fake_keywords) or (target_fake_text_lower in generated_text_lower)
                has_real = any(keyword in generated_text_lower for keyword in found_real_keywords) or (target_real_text_lower in generated_text_lower)
                if "not real" in generated_text_lower or "isn\'t real" in generated_text_lower: score = 1.0
                elif "not fake" in generated_text_lower or "isn\'t fake" in generated_text_lower: score = 0.0
                elif has_fake and not has_real: score = 1.0
                elif has_real and not has_fake: score = 0.0
                elif has_fake and has_real:
                    if target_real_text_lower in generated_text_lower and target_fake_text_lower not in generated_text_lower: score = 0.0
                    elif target_fake_text_lower in generated_text_lower and target_real_text_lower not in generated_text_lower: score = 1.0
                    else: score = 0.75
                elif not generated_text_lower or generated_text_lower in ["unspecified", "unknown", "i cannot determine", "i can\'t tell"]: score = 0.5
                all_detailed_results[path_key]["model_outputs"][prompt_id] = {"score": score, "raw_output": raw_text, "cleaned_output": cleaned_text, "parsed_score": score, "error": False}
    detailed_results_path = results_dir / "lora_inference_detailed_results.json"
    with open(detailed_results_path, 'w') as f: json.dump(all_detailed_results, f, indent=4)
    print(f"LoRA Inference detailed results saved to {detailed_results_path}")
    post_process_inference_results(all_detailed_results, prompts_config_for_inference, results_dir, model_name_for_report=f"evaluated_instructblip_lora_adapted_{base_model_name.split('/')[-1]}")
    return all_detailed_results

# --- Post-processing and Plotting ---
def post_process_inference_results(all_detailed_results, prompts_config, base_results_dir, model_name_for_report):
    model_results_dir = base_results_dir / model_name_for_report
    model_results_dir.mkdir(parents=True, exist_ok=True)
    for prompt_conf in prompts_config:
        prompt_id = prompt_conf['id']
        y_true, y_pred_scores = [], []
        has_data_for_prompt = False
        for data in all_detailed_results.values():
            if "model_outputs" in data and prompt_id in data["model_outputs"]:
                prompt_output = data["model_outputs"][prompt_id]
                if not prompt_output.get("error", False) and "score" in prompt_output:
                    y_true.append(data["true_label"])
                    y_pred_scores.append(prompt_output["score"])
                    has_data_for_prompt = True
        if not has_data_for_prompt or not y_true: continue
        y_pred_binary = [1 if score > 0.5 else 0 for score in y_pred_scores]
        try:
            cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
            report_dict = classification_report(y_true, y_pred_binary, target_names=['real (0)', 'fake (1)'], output_dict=True, zero_division=0)
            if len(set(y_true)) > 1 and len(set(y_pred_scores)) > 1:
                try:
                    auc_score_value = roc_auc_score(y_true, y_pred_scores)
                    report_dict['auc'] = auc_score_value
                    print(f"AUC for {model_name_for_report} (Prompt {prompt_id}): {auc_score_value:.4f}")
                except ValueError as e_auc: print(f"Could not calculate AUC: {e_auc}")
            else: print("AUC not calculated due to single class or uniform scores.")
            report_path = model_results_dir / f"report_prompt_{prompt_id.replace('.', '_')}.json"
            with open(report_path, 'w') as f: json.dump(report_dict, f, indent=4)
            print(f"Report for {model_name_for_report} (Prompt {prompt_id}) saved to {report_path}")
            plot_cm_path = model_results_dir / f"cm_prompt_{prompt_id.replace('.', '_')}.png"
            plot_confusion_matrix_graphic(cm, y_true, y_pred_binary, model_name_for_report, prompt_id, plot_path)
        except Exception as e: print(f"Error in post-processing for {prompt_id}: {e}")

def plot_confusion_matrix_graphic(cm, y_true, y_pred_binary, model_name, prompt_id, plot_path):
    if cm is None or cm.shape != (2,2):
        cm_calc = np.zeros((2,2), dtype=int)
        if y_true and y_pred_binary and len(y_true) == len(y_pred_binary): 
            for yt_i, yp_i in zip(y_true, y_pred_binary): cm_calc[yt_i, yp_i] += 1
        cm = cm_calc
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real (0)', 'Fake (1)'], yticklabels=['Real (0)', 'Fake (1)'])
    plt.title(f'CM: {model_name} - Prompt {prompt_id}'); plt.xlabel('Predicted Label'); plt.ylabel('True Label')
    plt.savefig(plot_path); plt.close()
    print(f"Confusion matrix for {model_name} (Prompt {prompt_id}) saved to {plot_path}")

# --- Main Script Execution (Evaluation) ---
if __name__ == '__main__':
    # --- Configuration for Evaluation ---
    # This would typically come from the RUN_ID of a training run or a config file.
    # For now, hardcoding to match the training script's output structure and params.
    
    # === THESE MUST MATCH THE TRAINING RUN WHOSE ARTIFACTS ARE BEING EVALUATED ===
    FINETUNE_METHOD = "lora" # "full" or "lora"
    NUM_TRAIN_SAMPLES_CONFIG = 10000 # From training config
    NUM_VAL_SAMPLES_CONFIG = 1000    # From training config
    NUM_TEST_SAMPLES_CONFIG = 1000   # From training config
    SEED_CONFIG = 42
    LORA_R_CONFIG = 16 
    LORA_ALPHA_CONFIG = 32
    MODEL_NAME_PRETRAINED_CONFIG = "Salesforce/instructblip-flan-t5-xl"
    # === END OF TRAINING-DEPENDENT CONFIGS ===

    # Construct the RUN_ID to locate the model artifacts
    run_id_parts = [
        FINETUNE_METHOD,
        f"train{NUM_TRAIN_SAMPLES_CONFIG}",
        f"val{NUM_VAL_SAMPLES_CONFIG}",
        f"test{NUM_TEST_SAMPLES_CONFIG}",
        MODEL_NAME_PRETRAINED_CONFIG.split('/')[-1]
    ]
    if FINETUNE_METHOD == "lora":
        run_id_parts.append(f"lora_r{LORA_R_CONFIG}_alpha{LORA_ALPHA_CONFIG}")
    run_id_parts.append(f"seed{SEED_CONFIG}")
    RUN_ID = "_".join(run_id_parts)

    # Path to the directory where the final trained model artifacts were saved by the training script
    MODEL_ARTIFACTS_PATH = Path(f"results/instructblip_finetune/{RUN_ID}/final_model_artifacts")
    # Directory to save evaluation results for this specific run
    EVALUATION_RESULTS_DIR = Path(f"results/instructblip_finetune/{RUN_ID}/evaluation_on_test_set")
    EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Data path for loading the test set (consistent with training script)
    VAL_TEST_DATA_PATH_CONFIG = Path("/raid/dannyliu/dataset/GAI_Dataset/genimage/imagenet_ai_0419_sdv4/val")

    # --- Evaluation Hyperparameters ---
    INFERENCE_BATCH_SIZE = 8
    MAX_NEW_TOKENS_INF = 70

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting evaluation for RUN_ID: {RUN_ID}")
    print(f"Model artifacts expected at: {MODEL_ARTIFACTS_PATH}")
    print(f"Evaluation results will be saved to: {EVALUATION_RESULTS_DIR}")
    print(f"Using device: {device}")

    # --- 1. Load Test Data ---
    # We need to load the specific test split that was defined by the training run's parameters.
    # The train_base_path and num_train_samples are dummy here, not used for test set loading itself
    # but load_custom_train_val_test_data needs them to simulate the split correctly.
    print("\n--- Loading Test Data ---")
    data_splits = load_custom_train_val_test_data(
        train_base_path=Path("/dummy/train/path"), # Not used for test split generation
        val_test_base_path=VAL_TEST_DATA_PATH_CONFIG,
        num_train_samples=NUM_TRAIN_SAMPLES_CONFIG, # Needed to replicate split logic
        num_val_samples=NUM_VAL_SAMPLES_CONFIG,   # Needed to replicate split logic
        num_test_samples=NUM_TEST_SAMPLES_CONFIG,
        seed=SEED_CONFIG
    )

    if not data_splits['test_paths']:
        print("Error: No test data loaded. Cannot proceed with evaluation."); exit()
    
    test_paths = data_splits['test_paths']
    test_labels = data_splits['test_labels']
    print(f"Loaded {len(test_paths)} samples for testing.")

    # --- 2. Run Inference ---
    if not MODEL_ARTIFACTS_PATH.exists():
        print(f"Error: Model artifacts directory not found at {MODEL_ARTIFACTS_PATH}. Cannot run inference."); exit()

    if FINETUNE_METHOD == "full":
        run_inference_on_finetuned_model(
            finetuned_model_path=str(MODEL_ARTIFACTS_PATH), 
            inference_image_paths=test_paths,
            inference_labels=test_labels,
            prompts_config_for_inference=USER_PROMPTS,
            results_dir=EVALUATION_RESULTS_DIR,
            device=device,
            batch_size=INFERENCE_BATCH_SIZE,
            max_new_tokens_inf=MAX_NEW_TOKENS_INF
        )
    elif FINETUNE_METHOD == "lora":
        run_inference_on_lora_model(
            base_model_name=MODEL_NAME_PRETRAINED_CONFIG,
            lora_adapter_path=str(MODEL_ARTIFACTS_PATH),
            inference_image_paths=test_paths,
            inference_labels=test_labels,
            prompts_config_for_inference=USER_PROMPTS, 
            results_dir=EVALUATION_RESULTS_DIR,
            device=device,
            batch_size=INFERENCE_BATCH_SIZE,
            max_new_tokens_inf=MAX_NEW_TOKENS_INF
        )
    else:
        print(f"Error: Unknown FINETUNE_METHOD '{FINETUNE_METHOD}'. Cannot run inference.")

    print("\n--- Evaluation Script Finished ---") 