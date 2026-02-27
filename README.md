# Neural Circuits for Navigation

Course materials for **Neural Circuits for Navigation: Anatomy, Physiology, and Computational Methods** at NCBS.

**Instructors**: Eric Denovellis and Abhilasha Joshi

## Course Structure

| Week | Topic | Notebook | Colab |
|------|-------|----------|-------|
| 1 | Loading and Exploring NWB Data | [01-loading-nwb-data.ipynb](notebooks/01-loading-nwb-data.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/01-loading-nwb-data.ipynb) |
| 2a | Spike-Stimulus Analysis | [02a-spike-stimulus-analysis.ipynb](notebooks/02a-spike-stimulus-analysis.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/02a-spike-stimulus-analysis.ipynb) |
| 2b | Poisson Regression / GLMs | [02b-poisson-regression.ipynb](notebooks/02b-poisson-regression.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/02b-poisson-regression.ipynb) |
| 2c | Model Evaluation & Diagnostics | [02c-model-evaluation.ipynb](notebooks/02c-model-evaluation.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/02c-model-evaluation.ipynb) |
| 3a | LFP and Spectral Analysis | [03a-spectral-lfp.ipynb](notebooks/03a-spectral-lfp.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/03a-spectral-lfp.ipynb) |
| 3b | Filtering, Hilbert Transform, and Coherence | [03b-filtering-coherence.ipynb](notebooks/03b-filtering-coherence.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/03b-filtering-coherence.ipynb) |
| 3c | Spike-Field Coupling, Phase-Based Measures, and Cross-Frequency Coupling | [03c-spike-field-coupling.ipynb](notebooks/03c-spike-field-coupling.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/03c-spike-field-coupling.ipynb) |
| 4 | Decoding + Open data use and visualization | [04-decoding.ipynb](notebooks/04-decoding.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/04-decoding.ipynb) |
| 5 | Clusterless Decoding Approaches | [05-clusterless-decoding.ipynb](notebooks/05-clusterless-decoding.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edeno/ncbs-neural-circuits-navigation/blob/main/notebooks/05-clusterless-decoding.ipynb) |
| 6 | Project + Presentation | — | — |

## Getting Started

### Google Colab (recommended, no installation)

Click the Colab badge next to any notebook in the table above. Run this cell first to install dependencies:

```python
!pip install -q pynwb hdmf dandi remfile h5py fsspec aiohttp requests statsmodels patsy time-rescale spectral-connectivity
```

### Local installation

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
