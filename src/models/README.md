# Models Module

This module contains model architectures for AI-generated image detection. It includes baseline CNN classifiers and Vision-Language Model (VLM) wrappers with support for zero-shot inference, fine-tuning, and parameter-efficient adaptation.

## Key Files

- `baseline_classifiers.py`: Traditional CNN models like ResNet
- `vlm_models.py`: Wrappers for CLIP, BLIP, and other VLMs

## Available Models

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

### Vision-Language Models

#### CLIPModel

Wrapper for OpenAI's CLIP model, supporting both zero-shot and fine-tuning:

```python
from src.models.vlm_models import CLIPModel

# Zero-shot mode
zero_shot_model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="zero-shot"
)

# Get zero-shot predictions
prompts = ["a real photo", "an AI-generated image"]
probabilities = zero_shot_model.predict_zero_shot(images, prompts)

# Fine-tuning mode
fine_tuned_model = CLIPModel(
    model_name="openai/clip-vit-base-patch32",
    mode="fine-tune",
    freeze_backbone=True,       # Only train the classification head
    num_classes=2
)

# Forward pass for fine-tuned model
logits = fine_tuned_model(images)  # shape: [batch_size, 2]
```

#### BLIPModel

Wrapper for Salesforce's BLIP with VQA-style adaptation:

```python
from src.models.vlm_models import BLIPModel

model = BLIPModel(
    model_name="Salesforce/blip-image-captioning-base",
    mode="vqa",                 # Use visual question answering head
    freeze_backbone=True
)

# Forward pass
logits = model(images)
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

```python
# Save a model
torch.save(model.state_dict(), "results/saved_models/clip_finetuned.pth")

# Load a model
model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune")
model.load_state_dict(torch.load("results/saved_models/clip_finetuned.pth"))
model.eval()
```

## Creating Custom Models

To add a new model, follow these steps:

1. Create a new class that inherits from `nn.Module`
2. Implement `__init__()` and `forward()` methods
3. Ensure the forward method returns logits for classifier models or provides a prediction interface for zero-shot models

Example:

```python
class CustomVLMModel(nn.Module):
    def __init__(self, model_name, mode="fine-tune", freeze_backbone=True):
        super().__init__()
        # Initialize your model components
        
    def forward(self, images):
        # Process images and return logits
        return logits
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