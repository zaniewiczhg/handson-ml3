# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Jupyter notebooks for the book "Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)" by Aurélien Géron. It includes example code and exercise solutions covering fundamental to advanced machine learning topics.

## Environment Setup

### Using Conda (Recommended)
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate homl3

# Register kernel with Jupyter
python3 -m ipykernel install --user --name=python3

# Start Jupyter
jupyter notebook
```

### Using pip (Alternative)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Register kernel and start Jupyter
python3 -m ipykernel install --user --name=python3
jupyter notebook
```

### Docker Setup
For Docker-based setup, see the `docker/` directory and `docker/README.md`.

## Notebook Structure

The repository is organized as follows:

- **Main chapters (01-19)**: Core machine learning topics, from basics to deployment at scale
- **Tools notebooks**: NumPy, Matplotlib, and Pandas tutorials (`tools_*.ipynb`)
- **Math notebooks**: Linear algebra and differential calculus refreshers (`math_*.ipynb`)
- **Extra notebooks**: Supplementary material like autodiff and gradient descent comparisons (`extra_*.ipynb`)
- **index.ipynb**: Navigation hub for all notebooks

## Key Dependencies

- **Python 3.10** (recommended, ≥3.7 works)
- **Core ML**: scikit-learn, TensorFlow (~2.14), Keras
- **Scientific Computing**: NumPy, Pandas, SciPy, Matplotlib
- **Deep Learning**: TensorFlow Hub, TensorFlow Datasets, Transformers
- **Specialized**: XGBoost, Gymnasium (reinforcement learning), Keras Tuner
- **Cloud**: Google Cloud AI Platform and Storage (chapter 19 only)

## Working with Notebooks

### Running Notebooks
- Start from `index.ipynb` for an overview
- Notebooks are designed to run sequentially within each chapter
- Each notebook is self-contained with necessary imports and data loading
- Output cells show expected results; re-run cells to regenerate

### Data Handling
- Datasets are typically downloaded automatically via code in notebooks (e.g., `load_housing_data()` in chapter 02)
- Some datasets are stored in the `datasets/` directory
- Test sets should be kept separate and not examined during exploration (avoid data snooping)

### Common Patterns
- Data is loaded at the start of each notebook
- Notebooks include both conceptual explanations (markdown) and executable code
- Visualizations use Matplotlib; models use Scikit-Learn and TensorFlow/Keras
- Most notebooks follow the ML project workflow from `ml-project-checklist.md`

## ML Project Workflow

When working on ML tasks, follow the 8-step process from `ml-project-checklist.md`:
1. Frame the problem and look at the big picture
2. Get the data
3. Explore the data to gain insights
4. Prepare the data (cleaning, feature engineering)
5. Explore models and short-list the best ones
6. Fine-tune models and combine them
7. Present the solution
8. Launch, monitor, and maintain

## Updating the Repository

```bash
# Pull latest changes
git pull

# If you have local changes, commit them first
git checkout -b my_branch
git add -u
git commit -m "describe your changes"
git checkout main
git pull

# Update conda environment
conda update -c defaults -n base conda
conda activate base
conda env remove -n homl3
conda env create -f environment.yml
conda activate homl3
```

## GPU Support

- To enable GPU support, install TensorFlow-compatible GPU drivers (NVIDIA with Compute Capability ≥3.5)
- Install CUDA and cuDNN (handled automatically by Anaconda)
- Replace `tensorflow-serving-api` with `tensorflow-serving-api-gpu` in dependencies
- See TensorFlow's GPU installation docs for detailed instructions

## Chapter-Specific Notes

- **Chapter 02**: End-to-end ML project with California housing dataset
- **Chapter 06**: Decision trees use Graphviz for visualization
- **Chapter 07**: Ensemble methods include XGBoost
- **Chapter 10**: Neural networks with Keras; uses Keras Tuner for hyperparameter optimization
- **Chapter 15**: Time series analysis with statsmodels
- **Chapter 16**: NLP with transformers library
- **Chapter 18**: Reinforcement learning with Gymnasium (requires Box2D, Atari ROMs)
- **Chapter 19**: Model deployment with Google Cloud AI Platform

## Troubleshooting

- **SSL errors on macOS**: Run `/Applications/Python\ 3.10/Install\ Certificates.command`
- **load_housing_data() errors**: Check network configuration; may be SSL-related
- **Memory issues with large notebooks**: Use kernel restart; consider running on Colab for GPU-intensive tasks
- **Windows Box2D installation**: Requires Swig and Microsoft C++ Build Tools; use Anaconda instead of pip