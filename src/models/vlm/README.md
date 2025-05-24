# Vision-Language Model (VLM) Submodule

This directory (`src/models/vlm/`) houses the core implementations for various Vision-Language Models, designed with extensibility and direct use of official pre-trained weights in mind.

## Key Design Principles

- **Abstract Base Class**: All VLM wrappers inherit from `base_vlm.BaseVLM`. This ensures a consistent interface for model loading (`_load_model` method) and prediction (`predict` method).
- **Official Pre-trained Weights**: Wrappers primarily focus on loading models directly from sources like Hugging Face Hub, facilitating the use of well-tested, official checkpoints.
- **Zero-Shot Focus**: The initial design prioritizes zero-shot inference capabilities, where models make predictions on images based on text prompts without specific task fine-tuning.
- **Configuration Driven**: Model specifics (like Hugging Face model IDs, task types, or inference parameters such as custom questions for generative VLMs) are passed via a configuration dictionary during instantiation.
- **Extensibility**: The structure allows for easy addition of new VLMs by creating a new class inheriting from `BaseVLM` and implementing the required methods.
- **Decoupling from `nn.Module`**: The wrappers themselves are not `torch.nn.Module` subclasses. Instead, they manage an underlying `nn.Module` (the actual VLM from Hugging Face). This simplifies their use for direct inference and allows flexibility in how they might be incorporated into larger PyTorch training workflows if needed (e.g., by accessing `wrapper.model`).

## Underlying PyTorch Model Access

It is important to note that while the VLM wrappers (e.g., `CLIPModelWrapper`, `BlipModelWrapper`) are not themselves `torch.nn.Module` instances, they load and manage the actual Hugging Face PyTorch model (which *is* an `nn.Module`) as an attribute, typically named `self.model` (for the main model) and `self.processor` (for the associated preprocessor).

This design means that if you need to interact directly with the underlying PyTorch model—for example, to move it to a specific device (`.to(device)`), set it to evaluation mode (`.eval()`), or access its parameters for fine-tuning—you should operate on this `self.model` attribute.

**Example**: 
To set a wrapped CLIP model to evaluation mode and move it to a CUDA device:
```python
# Assuming 'vlm_wrapper' is an instance of CLIPModelWrapper
# and 'device' is your target torch.device

if hasattr(vlm_wrapper, 'model') and isinstance(vlm_wrapper.model, torch.nn.Module):
    vlm_wrapper.model.eval()      # Set the underlying Hugging Face model to eval mode
    vlm_wrapper.model.to(device)  # Move the underlying model to the target device

if hasattr(vlm_wrapper, 'processor'):
    # Processors don't typically have an eval mode but might need device context for some operations
    # For many Hugging Face processors, they don't need to be explicitly moved to a device 
    # unless they create tensors internally during preprocessing without a device argument.
    pass 
```
This is particularly relevant when using these wrappers in scripts like `src/experiments/zero_shot_vlm_eval.py`, where such operations are necessary.

## Available Model Wrappers

- `clip_model.py`: Implements `CLIPModelWrapper` for CLIP models.
- `blip_model.py`: Implements `BlipModelWrapper` for BLIP models (configurable for tasks like Image-Text Matching).
- `instruct_blip_model.py`: Implements `InstructBlipModelWrapper` for InstructBLIP models.
- `llava_model.py`: Implements `LlavaModelWrapper` for LLaVA models.

Each wrapper handles the specifics of loading its corresponding model and processor, and provides a `predict(self, image: PIL.Image.Image, text_prompts: List[str]) -> Dict[str, float]` method.

- For models like CLIP and BLIP (ITM), `text_prompts` are usually class names or descriptive phrases, and the output dictionary contains similarity scores.
- For generative models like InstructBLIP and LLaVA, `text_prompts` are often keywords to search for in the model's generated response to a predefined (but configurable) question. The score indicates the presence of these keywords.

## Usage Example (Conceptual)

```python
from PIL import Image
# Assuming a function to get the correct wrapper based on config
# from .get_vlm import get_vlm_instance 

# # Example: Load CLIP
# clip_config = {"name": "CLIP", "model_id": "openai/clip-vit-large-patch14"}
# clip_vlm = get_vlm_instance(clip_config)

# img = Image.open("path/to/image.jpg")
# prompts = ["a genuine photograph", "a computer-generated artwork"]
# predictions = clip_vlm.predict(img, prompts)
# print(f"CLIP Predictions: {predictions}")

# # Example: Load LLaVA
# llava_config = {
#     "name": "LLaVA", 
#     "model_id": "llava-hf/llava-1.5-7b-hf",
#     "llava_question": "Is this image real or fake?",
#     "max_new_tokens": 50
# }
# llava_vlm = get_vlm_instance(llava_config)
# keywords_to_find = ["real", "fake"]
# predictions = llava_vlm.predict(img, keywords_to_find)
# print(f"LLaVA Predictions (keyword presence): {predictions}")

# Example: Load BLIP
blip_config = {
    "model_id": "Salesforce/blip-itm-large-coco", # For image-text matching (COCO-trained)
    "task": "image-text-matching"
}
blip_vlm = BlipModelWrapper(model_name="BLIP-ITM-L-COCO", config=blip_config)

img = Image.open("path/to/your/image.jpg")
prompts = ["a genuine photograph", "a computer-generated artwork"]
predictions = blip_vlm.predict(img, prompts)
print(f"BLIP Predictions: {predictions}")
```

## Future Extensions

While the current focus is zero-shot, the loaded `self.model` within each wrapper is the official Hugging Face model, which can be used for:
- Fine-tuning (by extracting `wrapper.model` and using it in a PyTorch training loop).
- Integration with parameter-efficient fine-tuning (PEFT) techniques.
- More complex inference strategies.

Refer to the main `src/models/README.md` for how these VLM wrappers fit into the broader model ecosystem of this project, including fine-tuning and PEFT approaches which might leverage other files like `src/models/vlm_models.py`. 