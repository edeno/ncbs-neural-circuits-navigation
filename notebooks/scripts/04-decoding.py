# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 4: Decoding + Open Data Use and Visualization
#
# This notebook introduces neural decoding methods and working with open neuroscience data.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pynwb import NWBHDF5IO
