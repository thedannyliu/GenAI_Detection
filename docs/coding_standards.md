# Coding Standards for AI-Generated Image Detection Project

This document outlines the coding standards and best practices for contributing to the VLM-based AI-Generated Image Detection project. Following these guidelines ensures code consistency, readability, and maintainability across the project.

## Python Style Guidelines

### General Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) standards for Python code
- Use 4 spaces for indentation (no tabs)
- Keep line length to a maximum of 100 characters
- Use consistent capitalization (e.g., `ClassName`, `function_name`, `variable_name`)
- Maintain logical sections in files with appropriate spacing (two blank lines between top-level functions/classes)
- Include docstrings for all modules, functions, classes, and methods

### Naming Conventions

- **Variables**: Use lowercase with underscores (`image_size`, `batch_count`)
- **Functions**: Use lowercase with underscores (`load_model`, `compute_accuracy`)
- **Classes**: Use CapWords/CamelCase (`ResNetClassifier`, `CLIPModel`)
- **Constants**: Use uppercase with underscores (`MAX_BATCH_SIZE`, `DEFAULT_LEARNING_RATE`)
- **Private methods/variables**: Prefix with underscore (`_internal_helper`, `_cached_data`)

### Docstrings

- Use [Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- Include parameter descriptions, return types, and exceptions where applicable
- Add examples for complex or non-obvious functionality

```python
def compute_metrics(predictions, labels):
    """Computes accuracy, precision, recall, and F1 score for binary classification.
    
    Args:
        predictions (numpy.ndarray): Model predictions as probabilities [0, 1]
        labels (numpy.ndarray): Ground truth labels (0 or 1)
    
    Returns:
        dict: Dictionary containing metrics:
            - 'accuracy': Overall accuracy
            - 'precision': Precision for positive class
            - 'recall': Recall for positive class
            - 'f1': F1 score
            - 'auc': Area under ROC curve
    
    Example:
        >>> preds = np.array([0.1, 0.9, 0.8, 0.2])
        >>> labels = np.array([0, 1, 1, 0])
        >>> compute_metrics(preds, labels)
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'auc': 1.0}
    """
```

## Code Organization

### Module Structure

- Each Python module should focus on a specific functionality
- Group related functions/classes together in a logical manner
- Import statements should be at the top of the file, grouped as:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library-specific imports
- Add a meaningful docstring at the beginning of each file explaining its purpose

### Class Design

- Follow the Single Responsibility Principle - classes should have one primary purpose
- Use inheritance appropriately but avoid deep inheritance hierarchies
- Define clear interfaces for classes that are meant to be subclassed
- Use class-level attributes for configuration parameters (e.g., default values)

## PyTorch-Specific Guidelines

### Model Definitions

- All models should inherit from `nn.Module`
- Implement `forward()` method with clear input/output documentation
- Initialize layers in `__init__` method, not in `forward()`
- Include model parameter counts and architecture summary in docstring

```python
class ResNet50Classifier(nn.Module):
    """ResNet-50 model adapted for binary image classification.
    
    Architecture:
        - ResNet-50 backbone (pretrained on ImageNet)
        - Global average pooling
        - Fully connected layer (2048 -> 2)
    
    Total params: ~23.5M
    Trainable params: ~23.5M
    """
```

### Data Loading

- Dataset classes should inherit from `torch.utils.data.Dataset`
- Implement `__len__()` and `__getitem__()` methods
- Use data transforms from `torchvision.transforms` when possible
- Document the expected data format and transformations

### Training Loops

- Use `model.train()` and `model.eval()` modes appropriately
- Zero gradients before backward pass (`optimizer.zero_grad()`)
- Use context managers for specific training modes (e.g., `with torch.no_grad():` for evaluation)
- Apply gradient clipping when necessary to stabilize training

## Version Control Practices

### Commits

- Write clear and concise commit messages
- Begin commit messages with a short summary (50 chars or less)
- Focus each commit on a single logical change
- Reference issue numbers in commit messages when applicable

### Branching

- `main` branch should always be stable
- Create feature branches for new functionality (`feature/prompt-tuning`)
- Create bugfix branches for fixes (`fix/dataset-loading`)
- Merge via pull requests after code review

## Testing Guidelines

### Test Structure

- Place tests in the `tests/` directory mirroring the structure of `src/`
- Name test files with `test_` prefix (e.g., `test_models.py`)
- Group tests by functionality and class being tested

### Test Content

- Test both normal operation and edge cases
- Write assertions that clearly indicate what's being tested
- Keep tests independent of each other
- Use fixtures or setup/teardown methods for common test data

```python
def test_clip_zero_shot_prediction():
    """Test that CLIP zero-shot prediction works with valid inputs."""
    model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="zero-shot")
    image = torch.randn(1, 3, 224, 224)
    prompts = ["a real photo", "an AI-generated image"]
    
    # Should return a prediction without errors
    prediction = model.predict_zero_shot(image, prompts)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert 0 <= prediction[0] <= 1
```

## Documentation

### Code Comments

- Use comments sparingly - focus on "why", not "what"
- Comment complex algorithms or non-obvious decisions
- Update comments when changing related code

### README Files

- Each major directory should have a README.md explaining its purpose
- Include setup instructions, usage examples, and architecture details
- Document dependencies and any environment configurations needed

### Jupyter Notebooks

- Use notebooks primarily for exploration, visualization, and demos
- Keep code in notebooks clean and well-commented
- Extract reusable functionality from notebooks into proper Python modules

## Dependency Management

- Clearly document all dependencies in `requirements.txt`
- Specify version numbers to ensure reproducibility
- Minimize dependencies when possible
- Document any non-PyPI dependencies and their installation process

## Performance Considerations

- Profile code for bottlenecks before optimizing
- Use vectorized operations when possible (NumPy/PyTorch)
- Consider GPU memory usage when designing models and batch sizes
- Use appropriate data types (e.g., float32 vs float16)
- Enable mixed precision training for performance when applicable

## Logging and Debugging

- Use the logging module instead of print statements
- Set appropriate log levels for different types of messages
- Include relevant context in log messages

```python
import logging

# Instead of print statements
logging.info(f"Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}")
```

## Config Management

- Store configuration in YAML files rather than hardcoding values
- Document each configuration parameter with a comment
- Provide sensible defaults
- Use a consistent structure across configuration files

## Error Handling

- Use explicit exception types rather than bare `except`
- Handle errors at the appropriate level
- Provide informative error messages
- Fail gracefully with user-friendly messages

By following these coding standards, we can ensure that our project remains maintainable, extensible, and accessible to all contributors. Remember that consistency is key - when in doubt, follow the conventions already established in the existing codebase. 