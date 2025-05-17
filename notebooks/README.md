# Notebooks

This directory contains Jupyter notebooks for exploration, visualization and analysis related to AI-generated image detection using VLMs.

## Available Notebooks

### Data Exploration

- `data_exploration.ipynb`: Examines the genimage dataset, visualizes sample images, and analyzes the distribution of classes and generators.

### Prompt Engineering

- `prompt_engineering.ipynb`: Interactive notebook for testing different prompt formulations for zero-shot CLIP. Evaluate which prompts yield the best classification performance.

### Results Analysis

- `results_analysis.ipynb`: Visualizes and analyzes training results, generates plots, and compares performance across different models.

## Usage Instructions

1. Ensure you have Jupyter installed:
   ```
   pip install jupyter
   ```

2. Launch Jupyter:
   ```
   jupyter notebook
   ```

3. Navigate to the notebook you wish to use and open it.

## Creating New Notebooks

When creating new notebooks for analysis, please follow these guidelines:

- Use clear, descriptive names
- Include markdown cells explaining the purpose and methodology
- Structure code in a logical, step-by-step manner
- Add comments for complex operations
- Keep visualization code separate from analysis logic
- Consider extracting reusable code into the `src/` modules

## Dependencies

Notebooks depend on the project's core libraries plus visualization packages:

- matplotlib
- seaborn
- pandas
- plotly (optional for interactive visualizations)

These are included in the main `requirements.txt` file. 