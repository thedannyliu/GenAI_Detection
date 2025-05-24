# Prompt Strategies Module

This directory (`src/prompts/`) defines strategies for generating text prompts to be used with Vision-Language Models (VLMs) for tasks like zero-shot image classification (e.g., distinguishing real vs. AI-generated images).

## Core Concepts

- **`BasePromptStrategy`**: An abstract base class (`base_prompt_strategy.BasePromptStrategy`) that all specific prompt strategies should inherit from. It defines a common interface:
    - `get_prompts(class_names: List[str] = None, image_info: Dict[str, Any] = None) -> List[str]`: 
        Generates a list of text prompts. These are typically used by models like CLIP or BLIP (in Image-Text Matching mode) for similarity scoring against an image.
    - `get_vlm_question() -> str`:
        Returns a specific question string. This is intended for use with generative VLMs (e.g., InstructBLIP, LLaVA), which are prompted with an image and a question.
    - `get_keywords_for_response_check() -> List[str]`:
        Provides a list of keywords. For generative VLMs, the model's textual response to the question is checked for the presence of these keywords to make a classification decision.

- **Configuration**: Prompt strategies can be configured via a dictionary passed to their constructor, allowing for customization of prompts, questions, and keywords without code changes.

## Available Strategies

- **`generic_prompts.GenImageDetectPrompts`**: 
    A strategy specifically designed for the GenImage real vs. AI detection task. It provides:
    - A default set of discriminative prompts (e.g., "a real photograph", "an AI-generated image").
    - A default question for generative models ("Is this image a real photograph or an AI-generated image?...").
    - A default list of keywords (e.g., "real photograph", "ai-generated image") to check in responses.
    These can be overridden via configuration.

- **`generic_prompts.SimpleBinaryPrompts`**: 
    A simpler strategy that can take two class names (e.g., "real", "fake") and construct basic prompts around them. Useful for quick setup of binary classification tasks.

## How to Use

1.  **Instantiate a strategy**: Choose a strategy and initialize it, optionally passing a configuration dictionary.
    ```python
    from src.prompts.generic_prompts import GenImageDetectPrompts

    prompt_config = {
        "discriminative_prompts": ["this is a genuine photo", "this is a computer artifact"],
        "generative_question": "Tell me about the origin of this visual data.",
        "generative_response_keywords": ["genuine photo", "computer artifact"]
    }
    prompt_strategy = GenImageDetectPrompts(config=prompt_config)
    ```

2.  **Get prompts/questions for your VLM**:

    *For CLIP-like models:*
    ```python
    prompts_for_clip = prompt_strategy.get_prompts(class_names=["real", "ai"])
    # -> e.g., ["a real photograph", "an AI-generated image"]
    # (or from config if provided)
    ```

    *For LLaVA/InstructBLIP-like models:*
    ```python
    question_for_llava = prompt_strategy.get_vlm_question()
    # -> "Tell me about the origin of this visual data."
    
    keywords = prompt_strategy.get_keywords_for_response_check()
    # -> ["genuine photo", "computer artifact"]
    
    # Then, in your VLP wrapper:
    # llava_vlm.config["llava_question"] = question_for_llava # (or pass directly)
    # llava_vlm.config["llava_prompt_template"] = "USER: <image>\n{} ASSISTANT:"
    # response = llava_vlm.predict(image, keywords_to_check_against=keywords)
    ```
    Note: The `predict` method of the generative VLM wrappers (`InstructBlipModelWrapper`, `LlavaModelWrapper`) already expects `text_prompts` to be the keywords to check for. The question they use can be set via their own `config` (e.g., `config["llava_question"]`). So, the prompt strategy's `get_vlm_question()` output would be fed into the VLM wrapper's configuration, and `get_keywords_for_response_check()` would be passed as the `text_prompts` argument to `predict()`.

## Configuration

- **Prompt-to-class mapping (`prompt_to_class_map`)**: 
    This mapping is crucial when using discriminative VLMs (like CLIP) within the `zero_shot_vlm_eval.py` script, particularly with the `GenImageDetectPrompts` strategy.
    The `prompt_to_class_map` is defined within the `prompt_config` section of your YAML configuration file (e.g., `configs/vlm_zero_shot_custom.yaml`).
    It dictates how the text prompt that a VLM finds most similar to an image is converted into a class label (e.g., `0` for 'real/natural' and `1` for 'AI-generated').

    **Important Considerations**:
    - **Completeness**: This map *must* include every single text prompt that your chosen VLM might select as the "best" or highest-scoring prompt from the list provided by `GenImageDetectPrompts.get_prompts_for_image()`.
    - **Consistency**: The prompts in this map should exactly match the strings returned by the prompt strategy.
    - **Default Fallback**: If `zero_shot_vlm_eval.py` encounters a "best prompt" from the VLM that is *not* present in your `prompt_to_class_map`, the `GenImageDetectPrompts` strategy will typically fall back to using the `default_label_if_no_match` (which itself can be configured, often defaulting to a specific class like 'AI-generated'). This can lead to unexpected classification behavior or skewed metrics if your map is incomplete.

    Example YAML structure:
```yaml
# In your main experiment configuration file (e.g., configs/vlm_zero_shot_custom.yaml)
# ...
vlm:
  # ...
  prompt_config:
    module: "src.prompts.generic_prompts"
    class: "GenImageDetectPrompts"
    params:
      config: # This 'config' key is important for GenImageDetectPrompts
        discriminative_prompts: # Optional: override default prompts
          - "a natural image from a camera"
          - "a photograph taken by a person"
          - "a synthetic picture generated by AI"
          - "an image created by a computer algorithm"
        prompt_to_class_map: # The critical mapping
          "a natural image from a camera": 0
          "a photograph taken by a person": 0
          "a real photograph": 0        # Default prompt
          "an authentic image": 0       # Example custom prompt
          "a natural image": 0          # Example custom prompt
          "an AI-generated image": 1    # Default prompt
          "a computer-generated artwork": 1 # Example custom prompt
          "a synthetic image": 1        # Example custom prompt
          "a synthetic picture generated by AI": 1
          "an image created by a computer algorithm": 1
        default_label_if_no_match: 2 # Optional: set a fallback label (e.g., 0 for real, 1 for AI, 2 for unknown/unclassified)
# ...
```

- **Troubleshooting**: If you observe warnings in the `zero_shot_vlm_eval.py` output like `Warning for ...: Best prompt '...' not in prompt_to_class_map. Using default label.`, or if many samples are classified with the `default_label_if_no_match` (e.g., label 2), it indicates that either your `prompt_to_class_map` is incomplete for discriminative VLMs, or that generative VLMs like InstructBLIP are not consistently producing the exact text phrases expected by their wrapper's parsing logic. For generative VLMs, ensure the VLM's specific question (e.g., `instructblip_question` in its config) and the answer parsing logic within its wrapper (e.g., in `InstructBlipModelWrapper.predict`) are aligned with the VLM's typical response format and the target phrases (e.g., "real photograph", "ai generated").

## Adding New Strategies

1.  Create a new Python file in `src/prompts/` (e.g., `my_custom_prompts.py`).
2.  Define a class that inherits from `BasePromptStrategy`.
3.  Implement the `get_prompts`, `get_vlm_question`, and `get_keywords_for_response_check` methods according to your new strategy's logic.
4.  Consider making your strategy configurable through the `self.config` dictionary. 