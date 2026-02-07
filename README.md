# Neural Circuits for Navigation

Course materials for **Neural Circuits for Navigation: Anatomy, Physiology, and Computational Methods** at NCBS.

**Instructors**: Eric Denovellis and Abhilasha Joshi

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/edeno/ncbs-neural-circuits-navigation/HEAD?labpath=notebooks)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/01-loading-nwb-data.ipynb)

## Course Structure

| Week | Topic | Notebook |
|------|-------|----------|
| 1 | Loading and Exploring NWB Data | [01-loading-nwb-data.ipynb](notebooks/01-loading-nwb-data.ipynb) |
| 2a | Spike-Stimulus Analysis | [02a-spike-stimulus-analysis.ipynb](notebooks/02a-spike-stimulus-analysis.ipynb) |
| 2b | Poisson Regression / GLMs | [02b-poisson-regression.ipynb](notebooks/02b-poisson-regression.ipynb) |
| 3 | Spectral properties of LFP | [03-spectral-lfp.ipynb](notebooks/03-spectral-lfp.ipynb) |
| 4 | Decoding + Open data use and visualization | [04-decoding.ipynb](notebooks/04-decoding.ipynb) |
| 5 | Clusterless Decoding Approaches | [05-clusterless-decoding.ipynb](notebooks/05-clusterless-decoding.ipynb) |
| 6 | Project + Presentation | — |

## Getting Started

**Option 1: Binder (no installation)**

Click the Binder badge above to launch notebooks in your browser. Note: Binder has limited memory (~2GB), which may cause issues with large datasets.

**Option 2: Google Colab (recommended for more memory)**

Click the Colab badge above, or open any notebook directly in Colab. Run this cell first to install dependencies:

```python
!pip install -q pynwb hdmf dandi remfile h5py fsspec aiohttp requests spectral-connectivity statsmodels patsy xarray
```

**Option 3: Local installation**

```bash
# Clone the repository
git clone https://github.com/edeno/ncbs-neural-circuits-navigation.git
cd ncbs-neural-circuits-navigation

# Install dependencies with uv
uv sync

# Launch JupyterLab
uv run jupyter lab
```

## Data Format

This course uses [NWB (Neurodata Without Borders)](https://www.nwb.org/) files for neural data.
