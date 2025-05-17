# Tests

This directory contains unit tests for the AI-Generated Image Detection project. The tests validate functionality, ensure code quality, and help catch regressions when modifying code.

## Test Structure

The test directory mirrors the structure of the `src` directory:

```
tests/
├── test_data_processing.py   # Tests for data loading/processing
├── test_models.py            # Tests for model functionality
├── test_training.py          # Tests for training pipeline
├── test_evaluation.py        # Tests for evaluation methods
└── test_utils.py             # Tests for utility functions
```

## Running Tests

### Run All Tests

To run all tests, use pytest from the project root:

```bash
pytest
```

### Run Tests for a Specific Module

To run tests for a particular module:

```bash
pytest tests/test_models.py
```

### Run a Specific Test

To run a specific test function:

```bash
pytest tests/test_models.py::test_clip_zero_shot
```

### Run with Verbosity

For more detailed output:

```bash
pytest -v
```

### Run with Coverage Report

To check test coverage:

```bash
pytest --cov=src
# For HTML report
pytest --cov=src --cov-report=html
```

## Test Configuration

The tests use configurations in `tests/conftest.py` including:

- Pytest fixtures for shared test resources
- Test environment setup/teardown
- Mock data generation

## Test Dependencies

Tests require additional packages specified in `requirements-dev.txt`:

- pytest: Testing framework
- pytest-cov: Coverage measurement
- pytest-mock: Mocking functionality

Install these with:

```bash
pip install -r requirements-dev.txt
```

## Writing Tests

### Test File Organization

Each test file follows a consistent structure:

1. Imports
2. Fixtures specific to that file
3. Helper functions
4. Test functions grouped by feature

### Test Function Naming

Tests are named descriptively:

- `test_feature_scenario_expectation` (e.g., `test_dataset_loading_correct_labels`)

### Test Categories

#### Unit Tests

Test individual functions and methods in isolation:

```python
def test_compute_accuracy_perfect_predictions():
    predictions = np.array([0, 1, 1, 0])
    labels = np.array([0, 1, 1, 0])
    accuracy = compute_accuracy(predictions, labels)
    assert accuracy == 1.0
```

#### Integration Tests

Test interactions between components:

```python
def test_model_dataset_compatibility(sample_dataset):
    # Test that model correctly processes data from dataset
    model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="fine-tune")
    dataloader = DataLoader(sample_dataset, batch_size=2)
    batch = next(iter(dataloader))
    images, labels = batch
    outputs = model(images)
    assert outputs.shape == (2, 2)  # [batch_size, num_classes]
```

#### Mocking External Dependencies

For tests that require external resources:

```python
def test_clip_loading(mocker):
    # Mock the clip loading function to avoid actual model download
    mock_clip = mocker.patch("src.models.vlm_models.clip_from_pretrained")
    mock_clip.return_value = (mocker.MagicMock(), mocker.MagicMock())
    
    # Test model initialization
    model = CLIPModel(model_name="openai/clip-vit-base-patch32", mode="zero-shot")
    assert model is not None
```

## Common Test Fixtures

Reusable components defined in `conftest.py`:

```python
@pytest.fixture
def sample_dataset():
    """Create a small dummy dataset for testing."""
    images = torch.randn(10, 3, 224, 224)
    labels = torch.randint(0, 2, (10,))
    return TensorDataset(images, labels)

@pytest.fixture
def model_config():
    """Return a model configuration for testing."""
    return {
        "type": "clip",
        "name": "openai/clip-vit-base-patch32",
        "mode": "fine-tune",
        "freeze_backbone": True
    }
```

## Testing GPU-Dependent Code

For tests requiring GPU:

```python
@pytest.mark.gpu
def test_model_gpu_compatibility():
    # This test only runs if GPUs are available
    if not torch.cuda.is_available():
        pytest.skip("Test requires GPU")
    
    model = CLIPModel(...)
    model.to("cuda")
    # Test GPU operations
```

Run GPU tests specifically:

```bash
pytest -m gpu
```

## Test Workflow Integration

### Continuous Integration

Tests automatically run on pull requests to ensure code quality. The CI pipeline:

1. Installs dependencies
2. Runs all tests
3. Generates coverage report
4. Fails if tests fail or coverage drops below threshold

### Pre-commit Testing

Before committing, it's recommended to run:

```bash
# Run fast tests
pytest -xvs

# Run full test suite with coverage
pytest --cov=src
```

## Troubleshooting Common Test Issues

- **Test hangs**: Check for infinite loops or deadlocks
- **Missing dependencies**: Ensure all requirements-dev.txt packages are installed
- **Inconsistent results**: Check if test depends on random operations, and ensure seeds are fixed
- **Slow tests**: Use the `--durations=10` flag to identify and optimize slow tests

## Mock Data

For tests that need sample images:

```python
def create_dummy_image(size=(224, 224)):
    """Create a dummy PIL image for testing."""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([(50, 50), (size[0]-50, size[1]-50)], fill='black')
    return img
``` 