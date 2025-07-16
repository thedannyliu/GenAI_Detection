# Models Module

## Overview
Contains CNN baselines and Vision-Language Model wrappers used across experiments.

This module contains model architectures for AI-generated image detection. It includes baseline CNN classifiers and Vision-Language Model (VLM) wrappers with support for zero-shot inference, fine-tuning, and parameter-efficient adaptation.

## Key Files and Structure

- `baseline_classifiers.py`: Traditional CNN models like ResNet.
- `vlm/`: A sub-directory dedicated to Vision-Language Models.
    - `base_vlm.py`: Defines an abstract base class `BaseVLM` for all VLM implementations, ensuring a consistent interface for loading models and making predictions.
    - `clip_model.py`: Wrapper for CLIP models.
    - `blip_model.py`: Wrapper for BLIP models.
    - `instruct_blip_model.py`: Wrapper for InstructBLIP models.
    - `llava_model.py`: Wrapper for LLaVA models.
    - `README.md`: (To be created) Documentation specific to the VLM submodule.
- `vlm_models.py`: May contain older VLM wrappers or utility classes for VLMs like `PromptTuningWrapper` and `AdapterEnhancedModel`. (Note: Core VLM implementations are now in the `vlm/` subdirectory).

## Available Models

### GXMA Fusion Detector

The GXMA detector fuses *frequency fingerprints* (FFT, DCT, Wavelet) with
CLIP semantics.  Two fusion strategies are available:

1. **Tier-1 (single-stream)** – `CrossAttentionFusion` (1 Query ⇆ 1 KV).
2. **Tier-2 (parallel streams)** – `ParallelCrossAttentionFusion` (1 Query ⇆ 3 KV)
   introduced in **2025-06-16**.

Refer to `src/models/gxma/README.md` for architecture details and to
`configs/gxma/poc_stage1/` for ready-made training configs.

---

### Baseline Models

#### ResNet50Classifier

A standard ResNet-50 backbone with a classification head for binary detection:

```python
from src.models.baseline_classifiers import ResNet50Classifier

# Initialize with ImageNet pre-training
model = ResNet50Classifier(
    pretrained=True,            # Use ImageNet weights
    freeze_backbone=False,      # Fine-tune the entire network
    num_classes=2               # Binary classification
)

# Forward pass
logits = model(images)          # shape: [batch_size, 2]
```

### Vision-Language Models (New Structure)

The core VLM implementations are now located under the `src.models.vlm` submodule and inherit from `BaseVLM`. These models are designed for zero-shot inference using official pre-trained weights and can be extended for fine-tuning or other adaptations.

#### `BaseVLM` (Abstract Class)
Located in `src.models.vlm.base_vlm`, this class provides the blueprint for VLM wrappers:
```python
from src.models.vlm.base_vlm import BaseVLM
# Subclasses implement _load_model() and predict()

# Example usage (conceptual):
# class MyVLM(BaseVLM):
#     def _load_model(self): # ...
#     def predict(self, image, text_prompts): # ...

# model_instance = MyVLM(model_name="my_vlm_identifier", config=model_specific_config)
# results = model_instance.predict(pil_image, ["prompt1", "prompt2"])
```

#### CLIPModelWrapper
Wrapper for OpenAI's CLIP model, primarily for zero-shot classification.
Located in `src.models.vlm.clip_model`.
```python
from src.models.vlm.clip_model import CLIPModelWrapper
from PIL import Image

# Configuration for the model (passed during instantiation)
clip_config = {"model_id": "openai/clip-vit-large-patch14"} 

# Initialize model
clip_vlm = CLIPModelWrapper(model_name="CLIP", config=clip_config)

# Example image and prompts
img = Image.open("path/to/your/image.jpg")
prompts = ["a real photograph", "an AI-generated image"]

# Get zero-shot predictions (dictionary of prompt: score)
probabilities = clip_vlm.predict(img, prompts)
print(probabilities)
```

#### BlipModelWrapper
Wrapper for Salesforce's BLIP model, configured for image-text matching for zero-shot classification.
Located in `src.models.vlm.blip_model`.
```python
from src.models.vlm.blip_model import BlipModelWrapper
from PIL import Image

blip_config = {
    "model_id": "Salesforce/blip-itm-large-eval", # For image-text matching
    "task": "image-text-matching"
}
blip_vlm = BlipModelWrapper(model_name="BLIP-ITM", config=blip_config)

img = Image.open("path/to/your/image.jpg")
prompts = ["a real photograph", "an AI-generated image"]
probabilities = blip_vlm.predict(img, prompts)
print(probabilities)
```

#### InstructBlipModelWrapper
Wrapper for Salesforce's InstructBLIP model. Zero-shot classification is performed by asking a question and checking if the generated response contains any of the target prompts.
Located in `src.models.vlm.instruct_blip_model`.
```python
from src.models.vlm.instruct_blip_model import InstructBlipModelWrapper
from PIL import Image

instruct_blip_config = {
    "model_id": "Salesforce/instructblip-vicuna-7b",
    "instructblip_question": "Is this image a real photograph or an AI-generated image? Answer:",
    "max_new_tokens": 50
}
instruct_blip_vlm = InstructBlipModelWrapper(model_name="InstructBLIP", config=instruct_blip_config)

img = Image.open("path/to/your/image.jpg")
prompts_to_check = ["real photograph", "ai-generated image"] # keywords to find in response
scores = instruct_blip_vlm.predict(img, prompts_to_check) # returns {prompt: 1.0 or 0.0}
print(scores)
```

#### LlavaModelWrapper
Wrapper for LLaVA models (e.g., `llava-hf/llava-1.5-7b-hf`). Zero-shot classification involves prompting with a question and checking the generated response.
Located in `src.models.vlm.llava_model`.
```python
from src.models.vlm.llava_model import LlavaModelWrapper
from PIL import Image

llava_config = {
    "model_id": "llava-hf/llava-1.5-7b-hf",
    # The prompt template might need adjustment based on the specific LLaVA version
    "llava_prompt_template": "USER: <image>\n{} ASSISTANT:", 
    "llava_question": "Is this image a real photograph or an AI-generated image? Please answer with 'real photograph' or 'AI-generated image'.",
    "max_new_tokens": 70 
}
llava_vlm = LlavaModelWrapper(model_name="LLaVA", config=llava_config)

img = Image.open("path/to/your/image.jpg")
# Prompts here are the expected phrases in the answer
phrases_to_check_in_answer = ["real photograph", "ai-generated image"] 
scores = llava_vlm.predict(img, phrases_to_check_in_answer) # returns {phrase: 1.0 or 0.0}
print(scores)
```

### Parameter-Efficient Adaptation

#### PromptTuningWrapper

Adds learnable prompt parameters to a VLM:

```python
from src.models.vlm_models import PromptTuningWrapper, CLIPModel

# Create base CLIP model
base_model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="zero-shot"
)

# Wrap with prompt tuning
prompt_tuned_model = PromptTuningWrapper(
    model=base_model,
    prompt_length=10,           # Number of learnable tokens
    prompt_initialization="random"  # or "text" to initialize from text
)

# Only the prompt parameters will be updated during training
```

#### AdapterEnhancedModel

Adds lightweight adapter modules to transformer layers:

```python
from src.models.vlm_models import AdapterEnhancedModel, CLIPModel

# Create base CLIP model
base_model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="zero-shot"
)

# Add adapters
adapter_model = AdapterEnhancedModel(
    model=base_model,
    adapter_type="bottleneck",   # Bottleneck adapter architecture
    reduction_factor=16,         # Reduction in adapter dimension
    use_scaling=True             # Scale adapter outputs
)

# Only adapter parameters will be updated during training
```

## Model Selection Guidelines

- **For baseline comparison**: Use `ResNet50Classifier`
- **For zero-shot evaluation**: Use `CLIPModel` in zero-shot mode
- **For full fine-tuning**: Use `CLIPModel` with `freeze_backbone=False`
- **For parameter-efficient tuning**:
  - Use `PromptTuningWrapper` when you want to only learn text prompts
  - Use `AdapterEnhancedModel` when you want to adapt both vision and text encoders

## Model Saving and Loading

Model saving and loading will generally use `torch.save` and `model.load_state_dict()`.
For the new VLM wrappers, you would instantiate the wrapper first, then load the state dict.
Note that only parameters of `self.model` (the Hugging Face model) are typically saved/loaded unless you have custom trainable parameters in the wrapper itself.

```python
from src.models.vlm.clip_model import CLIPModelWrapper
import torch

# Example for saving (assuming you've trained parts of the HF model or a head)
# For our zero-shot wrappers, direct saving/loading of the wrapper itself might not be common
# as they load pre-trained weights. If fine-tuning is added to these wrappers, then saving is key.

# To save the underlying Hugging Face model if it were fine-tuned through the wrapper:
# clip_vlm.model.save_pretrained("path/to/save/my_finetuned_clip_hf_model")
# clip_vlm.processor.save_pretrained("path/to/save/my_finetuned_clip_hf_model")

# If the wrapper itself has trainable parameters (e.g., a custom classification head not part of BaseVLM yet):
# torch.save(clip_vlm.state_dict(), "results/saved_models/my_clip_wrapper.pth")

# Loading example:
# clip_config = {"model_id": "path/to/save/my_finetuned_clip_hf_model"} # if HF model was saved
# loaded_clip_vlm = CLIPModelWrapper(model_name="CLIP", config=clip_config)
# If wrapper state was saved:
# loaded_clip_vlm.load_state_dict(torch.load("results/saved_models/my_clip_wrapper.pth"))
# loaded_clip_vlm.eval()
```
The original README showed saving `model.state_dict()`. This applies if `model` is an `nn.Module`.
Our `BaseVLM` subclasses *are not* `nn.Module` themselves (they *contain* an `nn.Module` at `self.model`).
If you intend to fine-tune these VLMs and save them, you'd typically save the state of `self.model` (the Hugging Face model).

If `CLIPModel` from `vlm_models.py` (which *is* an `nn.Module`) is used for fine-tuning:
```python
# Assuming CLIPModel from vlm_models.py is used for fine-tuning
# from src.models.vlm_models import CLIPModel
# model_to_save_load = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune", num_classes=2)
# ... training ...
# torch.save(model_to_save_load.state_dict(), "results/saved_models/clip_finetuned_from_vlm_models_py.pth")

# model_to_load = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune", num_classes=2)
# model_to_load.load_state_dict(torch.load("results/saved_models/clip_finetuned_from_vlm_models_py.pth"))
# model_to_load.eval()
```

## Creating Custom Models

To add a new VLM based on the new structure:
1. Create a new Python file in `src/models/vlm/` (e.g., `my_new_vlm.py`).
2. Define a class that inherits from `src.models.vlm.base_vlm.BaseVLM`.
3. Implement the `__init__` method:
    - Call `super().__init__(model_name, config)`.
    - The `config` dictionary can be used to pass model-specific parameters like `model_id` from Hugging Face, or any other operational parameters.
4. Implement the `_load_model(self)` method:
    - Load your pre-trained model and any necessary processors (e.g., from Hugging Face Transformers).
    - Store them as `self.model` and `self.processor`.
    - Ensure the model is in evaluation mode (`self.model.eval()`) if primarily used for inference.
    - Handle device placement (e.g., move to CUDA if available).
5. Implement the `predict(self, image: PIL.Image.Image, text_prompts: List[str]) -> Dict[str, float]` method:
    - Take a PIL Image and a list of text prompts as input.
    - Perform the VLM's prediction logic.
    - Return a dictionary where keys are the input `text_prompts` (or derived concepts) and values are their corresponding scores (e.g., probabilities, logits, or binary indicators).

Example for a new VLM in `src/models/vlm/custom_vlm_model.py`:
```python
# In src/models/vlm/custom_vlm_model.py
from PIL import Image
from typing import List, Dict, Any
# from transformers import AutoModel, AutoProcessor # Example imports
from .base_vlm import BaseVLM

class CustomVLMModelWrapper(BaseVLM):
    def _load_model(self):
        model_id = self.config.get("model_id", "some-default-custom-vlm")
        # self.processor = AutoProcessor.from_pretrained(model_id)
        # self.model = AutoModel.from_pretrained(model_id)
        # if torch.cuda.is_available():
        #     self.model.to("cuda")
        # self.model.eval()
        print(f"CustomVLM {self.model_name} loaded with id {model_id}") # Placeholder
        pass # Replace with actual loading logic

    def predict(self, image: Image.Image, text_prompts: List[str]) -> Dict[str, float]:
        # inputs = self.processor(text=text_prompts, images=image, return_tensors="pt", padding=True)
        # if torch.cuda.is_available():
        #     inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # with torch.no_grad():
        #     outputs = self.model(**inputs) # This depends on the model type
        print(f"CustomVLM {self.model_name} predicting for image and prompts: {text_prompts}") # Placeholder
        results = {prompt: 0.5 for prompt in text_prompts} # Placeholder
        return results
```

## Model Requirements

- All models should return consistent output formats:
  - For classifiers: logits of shape `[batch_size, num_classes]`
  - For zero-shot models: probabilities of shape `[batch_size]`
- Models should handle both training and evaluation modes appropriately
- Documentation should specify input requirements (size, normalization, etc.)

## Hardware Considerations

- Most VLMs require a GPU with at least 8GB VRAM
- For larger models (e.g., CLIP-L/14), 16GB or more is recommended
- Use mixed precision training (`torch.cuda.amp`) for memory efficiency 

## R-VFiD (Semantic–Frequency–Prompt Detector)

The `src.models.r_vfid` sub-module contains an initial, **parameter-efficient** implementation of the R-VFiD architecture described in `src/models/R-VFiD/R-VFiD.md`.

Key points:

* Frozen CLIP ViT-B/16 vision backbone and text encoder.
* Read-only *Router Prompt* tokens drive a cross-attention router that produces per-image weights α over multiple **frequency experts** (LoRA adapters).
* Lightweight Squeeze-and-Excite fusion combines high-level semantics (CLS token) with the α-weighted frequency vector, followed by a LoRA-adapted binary head.
* **LoRA Frequency Experts** now implemented via lightweight `LoRALinear` (rank 4).  Each expert learns only ~24 k parameters.
* Router upgraded to `MultiheadCrossAttentionRouter` (PyTorch MHA), providing ⍺-weights in a single pass.
* `RouterPromptPool` provides **read-only learnable tokens**; `add_domain_prompt_and_expert()` enables continual expansion without touching past prompts.
* Built-in `compute_loss()` implements BCE + Entropy (+ placeholder InfoNCE) for quick training prototype.
* **InfoNCE + Entropy loss** implemented via `RvfidModel.compute_loss()`; CLS token 對比 Router Prompt，並含 α 熵正則。
* Three *true* frequency experts (`npr`, `dncnn`, `noiseprint`) with low-level filters and LoRA projection (rank-4) are registered in `frequency_expert.py`.
* `add_domain_prompt_and_expert(mode=...)` 可在訓練中即時擴充新 prompt + expert 並自動凍結舊權重。

Usage example:

```python
import torch
from src.models.r_vfid import RvfidModel

# dummy batch of normalised images (B,3,224,224)
images = torch.randn(8, 3, 224, 224)

model = RvfidModel(num_experts=3)
logits = model(images)  # shape: (8, 2)
probs  = logits.softmax(dim=-1)[:, 1]  # P(fake)
print(probs)
```

The current implementation uses *stub* frequency experts.  To plug in real LoRA-based experts, extend `SimpleFrequencyExpert` or replace it with a PEFT-injected ViT block wrapper. 

* Basic shape integrity tests live in `tests/test_r_vfid.py` (pytest).  Run `pytest -q` after installing dev dependencies.
