# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course materials for **Neural Circuits for Navigation: Anatomy, Physiology, and Computational Methods** at NCBS. This is a 6-week course covering neural data analysis techniques for navigation research.

**Instructors**: Eric Denovellis and Abhilasha Joshi

## Course Structure

Notebooks are in `notebooks/` and numbered by week:

- `01-loading-nwb-data.ipynb` - Week 1: Loading and exploring NWB data
- `02a-spike-stimulus-analysis.ipynb` - Week 2a: Spike-Stimulus Analysis
- `02b-poisson-regression.ipynb` - Week 2b: Poisson Regression / GLMs
- `02c-model-evaluation.ipynb` - Week 2c: Model Evaluation & Diagnostics
- `03a-spectral-lfp.ipynb` - Week 3a: LFP and Spectral Analysis
- `04-decoding.ipynb` - Week 4: Decoding + Open data use and visualization
- `05-clusterless-decoding.ipynb` - Week 5: Clusterless Decoding Approaches
- Week 6: Project + Presentation

## Development Commands

This project uses **uv** for Python package management.

```bash
# Install dependencies
uv sync

# Run JupyterLab
uv run jupyter lab

# Build Jupyter Book site
uv run jupyter-book build .

# Preview site locally
uv run myst start

# Add a new dependency
uv add <package-name>
```

## Jupyter Book

The course uses Jupyter Book 2 (MyST) to generate a navigable website from notebooks. Configuration is in `myst.yml`.

Students can access materials three ways:

1. **Static website** - Read-only view (after deploying to GitHub Pages)
2. **Google Colab** - Interactive notebooks, no installation required
3. **Local** - Clone repo and run `uv sync`

## Key Dependencies

- **pynwb/hdmf**: Neurodata Without Borders format for neural data
- **numpy/scipy**: Numerical computing and signal processing
- **matplotlib**: Visualization
- **pandas**: Data manipulation
- **statsmodels/patsy**: GLM fitting and formula interface
- **jupyterlab**: Interactive notebooks
- **jupyter-book**: Build course website

## Data Format

This course uses NWB (Neurodata Without Borders) files for neural data, streamed from DANDI. NWB is a standardized format for neurophysiology data that includes:

- Spike times and waveforms
- LFP/continuous signals
- Behavioral data (position, velocity)
- Stimulus information

## Coding Style

This is a teaching repository. Code should demonstrate best practices for scientific Python:

**General principles:**

- Prefer clarity over cleverness
- Use vectorized NumPy/SciPy operations over explicit loops when appropriate
- Use list/dict comprehensions when they improve readability

**Matplotlib:**

- Use `plt.eventplot()` for spike rasters (not scatter)
- Use `ax.set()` to set multiple properties at once
- Use `ax.spines[["top", "right"]].set_visible(False)` for clean plots
- Prefer the object-oriented interface (`fig, ax = plt.subplots()`)

**Python idioms:**

- Use `.get()` for safe dictionary access with defaults
- Use `next()` with generator expressions to find first matching item
- Use tuple unpacking for readable coordinate extraction
- Use NumPy boolean masking for filtering arrays

**Documentation:**

- Include docstrings with NumPy-style parameter documentation
- Add comments explaining *why*, not *what*

## Notebook Editing

Notebooks (`.ipynb`) are paired with Python scripts (`.py`) via **jupytext** for reliable editing and version control. Use the `jupyter-notebook-editor` skill for editing notebooks.

**File structure:**

```
notebooks/
├── 01-loading-nwb-data.ipynb
├── 02a-spike-stimulus-analysis.ipynb
├── ...
└── scripts/
    ├── 01-loading-nwb-data.py
    ├── 02a-spike-stimulus-analysis.py
    └── ...
```

**To edit a notebook manually:**

```bash
# Edit the .py file in notebooks/scripts/, then sync from notebooks/ directory
cd notebooks
jupytext --sync scripts/01-loading-nwb-data.py
```

**Key points:**

- Edit the `.py` file (not the `.ipynb` directly) for reliable text-based editing
- The `.py` files use jupytext "percent" format with `# %%` cell markers
- Run `jupytext --sync` from the `notebooks/` directory to update the `.ipynb`
- Both files should be committed to git
