# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 3c: Spike-Field Coupling, Phase-Based Measures, and Cross-Frequency Coupling
#
# This notebook builds on the filtering, Hilbert transform, and coherence
# techniques from Weeks 3a–3b. There, we analyzed relationships *between LFP
# signals*. Now we ask three new questions:
#
# 1. **Are LFP coherence measures reliable?** Coherence can be inflated by
#    volume conduction and common sources. We introduce alternative coupling
#    measures — Phase Locking Value (PLV) and Phase Lag Index (PLI) — that
#    address these issues.
# 2. **How do spikes relate to LFP oscillations?** Neurons often fire at
#    preferred phases of ongoing rhythms like theta. We quantify this with
#    circular statistics, spike-field coherence, and harmonic Poisson regression.
# 3. **How do different frequency bands interact?** Phase-amplitude coupling
#    (PAC) reveals cross-frequency organization, such as gamma bursts locked
#    to theta phase.
#
# Using the same dataset from Weeks 1–3b ([Petersen & Buzsáki, 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7442698/)),
# we combine spike times with LFP to study these relationships.
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# 1. Understand the limitations of coherence (common sources, volume conduction)
# 2. Compute and interpret Phase Locking Value (PLV) and Phase Lag Index (PLI)
# 3. Construct phase tuning curves for spike-LFP relationships
# 4. Apply circular statistics (circular mean, mean resultant length, Rayleigh test)
# 5. Compute non-parametric spike-field coherence
# 6. Build a harmonic Poisson regression model for spike-phase coupling
# 7. Measure phase-amplitude coupling (PAC) between theta and gamma

# %% [markdown]
# ## Setup

# %%
# Install dependencies (required for Google Colab)
import subprocess
import sys

if "google.colab" in sys.modules:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "pynwb",
            "hdmf",
            "dandi",
            "remfile",
            "h5py",
            "fsspec",
            "aiohttp",
            "requests",
            "spectral-connectivity",
            "statsmodels",
            "patsy",
        ]
    )

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from remfile import File as RemoteFile
from scipy.signal import filtfilt, firwin, hilbert
from scipy.stats import chi2, circmean, circvar
from spectral_connectivity import Connectivity, Multitaper

# %% [markdown]
# ## Load Data from DANDI
#
# We need **both** NWB files for this notebook:
# 1. The **behavior + spike file** containing position, speed, spike times,
#    and electrode metadata
# 2. The **raw LFP file** containing electrophysiology at high sampling rate
#
# This is the first notebook where we analyze spikes and LFP together.

# %%
# Define the dataset location on DANDI
DANDISET_ID = "000059"
DANDISET_VERSION = "0.230907.2101"

# Behavior file (contains speed, spike times, electrodes)
BEHAVIOR_ASSET_PATH = (
    "sub-MS22/"
    "sub-MS22_ses-Peter-MS22-180629-110319-concat_desc-processed_behavior+ecephys.nwb"
)

# LFP file (contains raw electrophysiology)
LFP_ASSET_PATH = (
    "sub-MS22/"
    "sub-MS22_ses-Peter-MS22-180629-110319-concat_desc-raw_ecephys.nwb"
)

# %% [markdown]
# ### Load Behavior and Spike Data

# %%
# Connect to DANDI and get the streaming URL for behavior file
with DandiAPIClient() as client:
    dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
    behavior_asset = dandiset.get_asset_by_path(BEHAVIOR_ASSET_PATH)
    behavior_s3_url = behavior_asset.get_content_url(follow_redirects=1, strip_query=True)

print(f"Behavior file: {BEHAVIOR_ASSET_PATH}")
print(f"Streaming from: {behavior_s3_url[:80]}...")

# %%
# Open the behavior NWB file
behavior_remote_file = RemoteFile(behavior_s3_url)
behavior_h5_file = h5py.File(behavior_remote_file, "r")
behavior_io = NWBHDF5IO(file=behavior_h5_file, load_namespaces=True)
behavior_nwbfile = behavior_io.read()

print(f"Session: {behavior_nwbfile.identifier}")

# %%
# Extract speed data
behavior_module = behavior_nwbfile.processing["behavior"]

speed_interface = next(
    interface
    for name, interface in behavior_module.data_interfaces.items()
    if "speed" in name.lower()
)

speed_data = speed_interface.data[:]
speed_timestamps = speed_interface.timestamps[:]

print(f"Speed data shape: {speed_data.shape}")
print(f"Speed time range: {speed_timestamps[0]:.1f} - {speed_timestamps[-1]:.1f} s")

# %%
# Get electrodes info (for theta reference channel)
electrodes_df = behavior_nwbfile.electrodes.to_dataframe()

if "theta_reference" in electrodes_df.columns:
    theta_reference_indices = np.nonzero(electrodes_df["theta_reference"].values)[0]
    if len(theta_reference_indices) > 0:
        theta_reference_channel = theta_reference_indices[0]
        print(f"Theta reference channel: {theta_reference_channel}")
else:
    theta_reference_channel = 0
    print(f"No theta reference found, using channel {theta_reference_channel}")

# %%
# Extract spike times from a good-quality unit
units_df = behavior_nwbfile.units.to_dataframe()
good_unit_mask = units_df["quality"].isin(["good", "good2"])
good_unit_indices = np.where(good_unit_mask)[0]

# Select a unit (same unit used in notebooks 02a/02b)
unit_index = good_unit_indices[3]
all_spike_times = behavior_nwbfile.units["spike_times"][unit_index]

print(f"Number of good units: {len(good_unit_indices)}")
print(f"Selected unit {unit_index}: {len(all_spike_times)} total spikes")

# %% [markdown]
# ### Load LFP Data

# %%
# Get streaming URL for LFP file
with DandiAPIClient() as client:
    dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
    lfp_asset = dandiset.get_asset_by_path(LFP_ASSET_PATH)
    lfp_s3_url = lfp_asset.get_content_url(follow_redirects=1, strip_query=True)

print(f"LFP file: {LFP_ASSET_PATH}")
print(f"Streaming from: {lfp_s3_url[:80]}...")

# %%
# Open the LFP NWB file
lfp_remote_file = RemoteFile(lfp_s3_url)
lfp_h5_file = h5py.File(lfp_remote_file, "r")
lfp_io = NWBHDF5IO(file=lfp_h5_file, load_namespaces=True)
lfp_nwbfile = lfp_io.read()

print(f"LFP Session: {lfp_nwbfile.identifier}")

# %%
# Get LFP electrical series
lfp_electrical_series = next(iter(lfp_nwbfile.acquisition.values()))

lfp_sampling_rate = lfp_electrical_series.rate
n_channels = lfp_electrical_series.data.shape[1]

print(f"LFP data shape: {lfp_electrical_series.data.shape}")
print(f"LFP sampling rate: {lfp_sampling_rate} Hz")
print(f"Number of channels: {n_channels}")

# %%
# Load a 60-second segment of LFP aligned to the behavior epoch
ANALYSIS_DURATION = 60  # seconds
behavior_start_sample = int(speed_timestamps[0] * lfp_sampling_rate)
n_analysis_samples = int(ANALYSIS_DURATION * lfp_sampling_rate)

# Load two channels: the theta reference and a nearby channel for coherence
second_channel = min(theta_reference_channel + 1, n_channels - 1)

lfp_channel_1 = lfp_electrical_series.data[
    behavior_start_sample : behavior_start_sample + n_analysis_samples,
    theta_reference_channel,
]
lfp_channel_2 = lfp_electrical_series.data[
    behavior_start_sample : behavior_start_sample + n_analysis_samples,
    second_channel,
]
lfp_time_original = np.arange(n_analysis_samples) / lfp_sampling_rate

print(f"Loaded {ANALYSIS_DURATION}s of LFP from channels {theta_reference_channel} and {second_channel}")

# %% [markdown]
# ### Downsample and Filter LFP
#
# The raw LFP is sampled at ~30 kHz — much higher than needed for theta
# (4–12 Hz) or gamma (30–80 Hz) analysis. We downsample to 1250 Hz, which
# preserves frequencies up to 625 Hz (well above our bands of interest).
#
# **Note**: Proper downsampling should include a low-pass anti-aliasing filter
# before decimation to prevent high-frequency content from folding into the
# passband. We skip this step because any aliased content that falls outside
# the theta (4–12 Hz) and gamma (30–80 Hz) bands will be removed by the
# subsequent bandpass filters. In practice, neural signal power above 625 Hz
# is very weak, so aliasing artifacts are negligible.

# %%
# Downsample LFP for computational efficiency
TARGET_SAMPLING_RATE = 1250  # Hz — standard LFP rate
downsample_factor = int(lfp_sampling_rate / TARGET_SAMPLING_RATE)
actual_sampling_rate = lfp_sampling_rate / downsample_factor

lfp_channel_1_downsampled = lfp_channel_1[::downsample_factor]
lfp_channel_2_downsampled = lfp_channel_2[::downsample_factor]
lfp_time = np.arange(len(lfp_channel_1_downsampled)) / actual_sampling_rate

# Remove DC offset
lfp_channel_1_detrended = lfp_channel_1_downsampled - np.mean(lfp_channel_1_downsampled)
lfp_channel_2_detrended = lfp_channel_2_downsampled - np.mean(lfp_channel_2_downsampled)

print(f"Downsampled: {lfp_sampling_rate} Hz → {actual_sampling_rate} Hz (factor {downsample_factor})")
print(f"Samples: {len(lfp_channel_1_detrended)}")

# %%
# Apply theta bandpass filter and extract instantaneous phase/amplitude
# (same pipeline as notebook 03b)
THETA_LOW = 4  # Hz
THETA_HIGH = 12  # Hz

# Design FIR bandpass filter: at least 3 cycles of the lowest frequency
n_filter_taps = int(3 * actual_sampling_rate / THETA_LOW)
if n_filter_taps % 2 == 0:
    n_filter_taps += 1

theta_filter_coefficients = firwin(
    n_filter_taps,
    [THETA_LOW, THETA_HIGH],
    pass_zero=False,
    fs=actual_sampling_rate,
)

# Zero-phase filtering preserves phase information
theta_filtered_lfp = filtfilt(theta_filter_coefficients, 1.0, lfp_channel_1_detrended)

# Hilbert transform → instantaneous phase and amplitude
theta_analytic_signal = hilbert(theta_filtered_lfp)
theta_instantaneous_phase = np.angle(theta_analytic_signal)
theta_instantaneous_amplitude = np.abs(theta_analytic_signal)

print(f"Theta filter: {n_filter_taps} taps ({n_filter_taps / actual_sampling_rate * 1000:.0f} ms)")

# %% [markdown]
# ## Part 1: Beyond Coherence — Robust Measures of Neural Coupling
#
# ### Communication Through Coherence and Its Limitations
#
# The **communication through coherence** hypothesis (Fries, 2005, 2015)
# proposes that neural communication between brain regions is gated by
# oscillatory coherence. When two regions' LFPs are phase-aligned, they
# create time windows of high excitability so that spikes arriving from one
# region are more effective at driving responses in the other.
#
# In Notebook 3b, we measured coherence between LFP channels. But coherence
# has important limitations:
#
# 1. **Common sources**: A third neural generator can drive correlated signals
#    at multiple electrodes, creating spurious coherence.
# 2. **Volume conduction**: Electrical fields spread through brain tissue, so
#    nearby electrodes pick up the same source even without direct neural
#    communication. Volume conduction produces **zero-lag** (0° or 180°)
#    phase relationships.
#
# We need measures that can distinguish genuine neural coupling from these
# artifacts.

# %% [markdown]
# ### Phase Locking Value (PLV)
#
# Coherence depends on both the **phase relationship** and the **power** of
# each signal. The Phase Locking Value (Lachaux et al., 1999) removes the
# power dependence by normalizing each cross-spectral estimate to unit
# magnitude before averaging:
#
# $$\text{PLV} = \left| \left\langle \frac{S_{xy}}{|S_{xy}|} \right\rangle \right| = \left| \left\langle e^{i\Phi} \right\rangle \right|$$
#
# where $\Phi$ is the phase difference between signals $x$ and $y$.
#
# **Intuition**: PLV asks "do the phase vectors point in the same direction?"
# regardless of their length (power).
#
# **Problem**: By normalizing to unit length, PLV gives **equal weight to
# noisy, low-power trials**. A trial where the signal is buried in noise
# gets the same influence as a trial with a clean, strong signal.

# %%
# Synthetic demo: PLV vs coherence with varying signal power
synth_sampling_rate = 1000  # Hz
trial_duration = 1.0  # seconds
n_trials = 20
time_trial = np.arange(0, trial_duration, 1 / synth_sampling_rate)
n_samples_per_trial = len(time_trial)

np.random.seed(42)

# Create trials where some have high power (clean) and others have low power (noisy)
signal_x = np.zeros((n_trials, n_samples_per_trial))
signal_y = np.zeros((n_trials, n_samples_per_trial))

constant_phase_lag = np.pi / 4  # 45 degrees

for trial in range(n_trials):
    # Alternate between high-power and low-power trials
    if trial < n_trials // 2:
        amplitude = 1.0  # Strong signal
        noise_level = 0.2
    else:
        amplitude = 0.1  # Weak signal — phase estimate is noisy
        noise_level = 1.0

    signal_x[trial] = (
        amplitude * np.cos(2 * np.pi * 10 * time_trial)
        + noise_level * np.random.randn(n_samples_per_trial)
    )
    signal_y[trial] = (
        amplitude * np.cos(2 * np.pi * 10 * time_trial - constant_phase_lag)
        + noise_level * np.random.randn(n_samples_per_trial)
    )

# Compute coherence and PLV using spectral_connectivity
mixed_power_data = np.stack([signal_x.T, signal_y.T], axis=-1)

multitaper_mixed = Multitaper(
    mixed_power_data,
    sampling_frequency=synth_sampling_rate,
    time_halfbandwidth_product=4,
)
connectivity_mixed = Connectivity.from_multitaper(multitaper_mixed)
coherence_mixed = connectivity_mixed.coherence_magnitude()
plv_mixed = connectivity_mixed.phase_locking_value()
frequencies_mixed = connectivity_mixed.frequencies

# %%
# Plot coherence vs PLV
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

ax.plot(
    frequencies_mixed,
    coherence_mixed[:, :, 0, 1].squeeze(),
    color="steelblue",
    linewidth=2,
    label="Coherence",
)
ax.plot(
    frequencies_mixed,
    plv_mixed[:, :, 0, 1].squeeze(),
    color="coral",
    linewidth=2,
    label="PLV",
)
ax.axvline(10, color="gray", linestyle="--", alpha=0.5, label="10 Hz (signal)")
ax.set(
    xlabel="Frequency (Hz)",
    ylabel="Coupling strength",
    title="Coherence vs. PLV with Mixed Signal Power",
    xlim=(0, 50),
    ylim=(0, 1),
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# PLV differs from coherence because it treats all trials equally regardless
# of signal strength. When some trials have very low power (buried in noise),
# PLV upweights their noisy phase estimates, which can either inflate or
# deflate the coupling measure depending on the noise pattern.

# %% [markdown]
# ### Phase Lag Index (PLI)
#
# The **Phase Lag Index** (Stam et al., 2007) takes a fundamentally different
# approach: it uses only the **imaginary part** of the cross-spectrum to
# detect coupling.
#
# The key insight is that **volume conduction produces zero-lag relationships**:
# signals arrive at electrodes simultaneously, giving a cross-spectral phase
# of exactly 0 or $\pi$. The imaginary part of these zero-lag cross-spectra
# is exactly zero: $\text{Im}(e^{i \cdot 0}) = 0$ and
# $\text{Im}(e^{i\pi}) = 0$.
#
# Therefore, any non-zero imaginary component must reflect **genuine lagged
# coupling**. The PLI is defined as:
#
# $$\text{PLI} = \left| \left\langle \text{sign}(\text{Im}(S_{xy})) \right\rangle \right|$$
#
# | Value | Interpretation |
# |-------|---------------|
# | PLI ≈ 0 | No lagged coupling (or volume conduction artifact) |
# | PLI ≈ 1 | Perfect lagged coupling |
#
# **Advantage**: Completely robust to zero-lag sources (volume conduction).
#
# **Disadvantage**: Discards any genuine coupling that happens to occur at
# exactly 0 or $\pi$ phase lag, reducing sensitivity.

# %%
# Synthetic demo: three coupling scenarios
n_trials_demo = 30
time_demo = np.arange(0, 1.0, 1 / synth_sampling_rate)
n_samples_demo = len(time_demo)

np.random.seed(123)
noise_level_demo = 0.5

scenarios = {}

# Scenario A: Genuine 45-degree lagged coupling
genuine_x = np.zeros((n_trials_demo, n_samples_demo))
genuine_y = np.zeros((n_trials_demo, n_samples_demo))
for trial in range(n_trials_demo):
    genuine_x[trial] = np.cos(2 * np.pi * 10 * time_demo) + noise_level_demo * np.random.randn(n_samples_demo)
    genuine_y[trial] = np.cos(2 * np.pi * 10 * time_demo - np.pi / 4) + noise_level_demo * np.random.randn(n_samples_demo)
scenarios["Genuine lag (45°)"] = (genuine_x, genuine_y)

# Scenario B: Volume conduction — same zero-lag source at both electrodes
common_source_x = np.zeros((n_trials_demo, n_samples_demo))
common_source_y = np.zeros((n_trials_demo, n_samples_demo))
for trial in range(n_trials_demo):
    shared_signal = np.cos(2 * np.pi * 10 * time_demo + np.random.uniform(0, 2 * np.pi))
    common_source_x[trial] = shared_signal + noise_level_demo * np.random.randn(n_samples_demo)
    common_source_y[trial] = 0.8 * shared_signal + noise_level_demo * np.random.randn(n_samples_demo)
scenarios["Volume conduction (0°)"] = (common_source_x, common_source_y)

# Scenario C: Independent noise — no coupling
independent_x = np.zeros((n_trials_demo, n_samples_demo))
independent_y = np.zeros((n_trials_demo, n_samples_demo))
for trial in range(n_trials_demo):
    independent_x[trial] = np.cos(2 * np.pi * 10 * time_demo) + noise_level_demo * np.random.randn(n_samples_demo)
    independent_y[trial] = np.cos(2 * np.pi * 10 * time_demo + np.random.uniform(0, 2 * np.pi)) + noise_level_demo * np.random.randn(n_samples_demo)
scenarios["Independent (no coupling)"] = (independent_x, independent_y)

# Compute all three measures for each scenario
results_by_scenario = {}

for scenario_name, (scenario_x, scenario_y) in scenarios.items():
    scenario_data = np.stack([scenario_x.T, scenario_y.T], axis=-1)
    multitaper_scenario = Multitaper(
        scenario_data,
        sampling_frequency=synth_sampling_rate,
        time_halfbandwidth_product=4,
    )
    connectivity_scenario = Connectivity.from_multitaper(multitaper_scenario)
    # Note: phase_lag_index() returns a signed value in [-1, 1].
    # We take np.abs() to get the unsigned PLI in [0, 1] (Stam et al., 2007).
    results_by_scenario[scenario_name] = {
        "coherence": connectivity_scenario.coherence_magnitude()[:, :, 0, 1].squeeze(),
        "plv": connectivity_scenario.phase_locking_value()[:, :, 0, 1].squeeze(),
        "pli": np.abs(connectivity_scenario.phase_lag_index()[:, :, 0, 1].squeeze()),
        "frequencies": connectivity_scenario.frequencies,
    }

# %%
# Compare measures across scenarios
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained", sharey=True)

colors = {"coherence": "steelblue", "plv": "coral", "pli": "forestgreen"}
labels = {"coherence": "Coherence", "plv": "PLV", "pli": "|PLI|"}

for ax, (scenario_name, scenario_results) in zip(axes, results_by_scenario.items()):
    for measure_name in ["coherence", "plv", "pli"]:
        ax.plot(
            scenario_results["frequencies"],
            scenario_results[measure_name],
            color=colors[measure_name],
            linewidth=2,
            label=labels[measure_name],
        )
    ax.axvline(10, color="gray", linestyle="--", alpha=0.5)
    ax.set(
        xlabel="Frequency (Hz)",
        title=scenario_name,
        xlim=(0, 50),
        ylim=(0, 1),
    )
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Coupling strength")
axes[0].legend()

fig.suptitle("Comparing Coupling Measures Across Scenarios", fontsize=12)

# %% [markdown]
# **Key observations:**
#
# - **Genuine lag**: All three measures detect coupling at 10 Hz.
# - **Volume conduction**: Coherence and PLV show high values at 10 Hz — they
#   cannot distinguish this from genuine coupling. **PLI correctly shows low
#   values** because the zero-lag relationship has no imaginary component.
# - **Independent signals**: All measures are low, as expected.
#
# This demonstrates why PLI is preferred when volume conduction is a concern.
#
# ### Summary of Coupling Measures
#
# | Measure | Ignores power? | Robust to volume conduction? | Range |
# |---------|:--------------:|:----------------------------:|:-----:|
# | Coherence | No | No | [0, 1] |
# | PLV | Yes | No | [0, 1] |
# | PLI | Yes | Yes | [0, 1] |
# | Imaginary coherence | No | Yes | [−1, 1] |
# | Weighted PLI | Partial | Yes | [0, 1] |
# | Pairwise phase consistency | Yes (debiased) | No | [0, 1] |
#
# All of these measures are available in the `spectral_connectivity` package.

# %% [markdown]
# ### Comparing Coupling Measures on Real LFP
#
# Let's compute coherence, PLV, and PLI between our two hippocampal LFP
# channels. Since these channels are on the same probe and physically close,
# we expect volume conduction to contribute — so PLI should be lower than
# coherence at some frequencies.

# %%
# Compute coupling measures between two real LFP channels
two_channel_lfp = np.stack(
    [lfp_channel_1_detrended, lfp_channel_2_detrended], axis=-1
)[:, np.newaxis, :]

multitaper_lfp = Multitaper(
    two_channel_lfp,
    sampling_frequency=actual_sampling_rate,
    time_halfbandwidth_product=4,
    time_window_duration=1.0,
    time_window_step=0.5,
    detrend_type="constant",
)

connectivity_lfp = Connectivity.from_multitaper(multitaper_lfp)
lfp_coherence = connectivity_lfp.coherence_magnitude()
lfp_plv = connectivity_lfp.phase_locking_value()
lfp_pli = np.abs(connectivity_lfp.phase_lag_index())
coherence_frequencies = connectivity_lfp.frequencies

# Average across time windows
mean_coherence = np.mean(lfp_coherence[:, :, 0, 1], axis=0)
mean_plv = np.mean(lfp_plv[:, :, 0, 1], axis=0)
mean_pli = np.mean(lfp_pli[:, :, 0, 1], axis=0)

# %%
# Plot all three measures
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

ax.plot(coherence_frequencies, mean_coherence, color="steelblue", linewidth=1.5, label="Coherence")
ax.plot(coherence_frequencies, mean_plv, color="coral", linewidth=1.5, label="PLV")
ax.plot(coherence_frequencies, mean_pli, color="forestgreen", linewidth=1.5, label="|PLI|")
ax.axvspan(THETA_LOW, THETA_HIGH, alpha=0.15, color="orange", label="Theta (4–12 Hz)")
ax.set(
    xlabel="Frequency (Hz)",
    ylabel="Coupling strength",
    title=f"LFP Coupling: Channel {theta_reference_channel} vs. Channel {second_channel}",
    xlim=(0, 100),
    ylim=(0, 1),
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# At low frequencies, coherence and PLV are high between nearby channels on
# the same probe — much of this reflects volume conduction (shared electrical
# sources). PLI is notably lower, confirming that a substantial portion of
# the apparent coupling is due to zero-lag sources rather than genuine
# lagged neural communication.

# %% [markdown]
# ## Part 2: Spike-LFP Relationships
#
# So far, we have related LFP signals to each other. A more fundamental
# question is: **how does the spiking of individual neurons relate to ongoing
# LFP oscillations?**
#
# Hippocampal neurons are known to fire at preferred phases of the theta
# cycle. This "theta phase locking" is thought to coordinate neural activity
# and support memory processes.
#
# We'll build up from simple visualization to rigorous statistical tests:
#
# 1. Phase tuning curves (histogram approach)
# 2. Circular statistics (circular mean, mean resultant length, Rayleigh test)
# 3. Non-parametric spike-field coherence
# 4. Harmonic Poisson regression (connecting to the GLM framework from Week 2b)

# %% [markdown]
# ### Assigning LFP Phase to Each Spike
#
# To study spike-LFP coupling, we need to know the **instantaneous theta
# phase at the time of each spike**. The procedure:
#
# 1. We already have the theta-filtered LFP and its instantaneous phase
# 2. Restrict spikes to the time window of our LFP segment
# 3. Interpolate theta phase at each spike time (since spikes don't fall
#    exactly on LFP sample times)

# %%
# Restrict spike times to the LFP analysis window
lfp_start_time = speed_timestamps[0]
lfp_end_time = lfp_start_time + ANALYSIS_DURATION

spike_in_segment_mask = (all_spike_times >= lfp_start_time) & (all_spike_times < lfp_end_time)
spike_times_in_segment = all_spike_times[spike_in_segment_mask]

# Convert to relative time (matching lfp_time)
spike_times_relative = spike_times_in_segment - lfp_start_time

print(f"Spikes in {ANALYSIS_DURATION}s LFP segment: {len(spike_times_relative)}")
print(f"Mean firing rate: {len(spike_times_relative) / ANALYSIS_DURATION:.1f} Hz")

# %%
# Interpolate theta phase and amplitude at each spike time
spike_theta_phases = np.interp(spike_times_relative, lfp_time, theta_instantaneous_phase)
spike_theta_amplitudes = np.interp(spike_times_relative, lfp_time, theta_instantaneous_amplitude)

print(f"Phase range: [{spike_theta_phases.min():.2f}, {spike_theta_phases.max():.2f}] radians")

# %%
# Visualize spikes on theta-filtered LFP
PLOT_START = 5.0  # seconds
PLOT_DURATION = 2.0  # seconds
plot_time_mask = (lfp_time >= PLOT_START) & (lfp_time < PLOT_START + PLOT_DURATION)
spike_plot_mask = (spike_times_relative >= PLOT_START) & (spike_times_relative < PLOT_START + PLOT_DURATION)

fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, layout="constrained")

# Theta-filtered LFP with spike markers
ax = axes[0]
ax.plot(lfp_time[plot_time_mask], theta_filtered_lfp[plot_time_mask], color="steelblue", linewidth=0.8)
spike_times_to_plot = spike_times_relative[spike_plot_mask]
spike_y_positions = np.interp(spike_times_to_plot, lfp_time, theta_filtered_lfp)
ax.plot(spike_times_to_plot, spike_y_positions, "r|", markersize=15, markeredgewidth=2)
ax.set(ylabel="Voltage (µV)", title=f"Unit {unit_index}: Spikes on Theta-Filtered LFP")
ax.spines[["top", "right"]].set_visible(False)

# Instantaneous phase with spike markers
ax = axes[1]
ax.plot(lfp_time[plot_time_mask], theta_instantaneous_phase[plot_time_mask], color="green", linewidth=0.5)
spike_phases_to_plot = np.interp(spike_times_to_plot, lfp_time, theta_instantaneous_phase)
ax.plot(spike_times_to_plot, spike_phases_to_plot, "r|", markersize=15, markeredgewidth=2)
ax.set(
    xlabel="Time (s)",
    ylabel="Phase (radians)",
    title="Theta Phase at Spike Times",
    yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    yticklabels=["−π", "−π/2", "0", "π/2", "π"],
)
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The red tick marks show when this neuron fires. If spikes cluster at
# certain phases of the theta cycle (e.g., consistently near the trough),
# the neuron is **theta phase-locked**.

# %% [markdown]
# ### Phase Tuning Curves
#
# A **phase tuning curve** is the analog of a place field, but for
# oscillatory phase instead of position. We ask: at which theta phase does
# this neuron fire most?
#
# Steps:
# 1. Divide the full phase circle $[-\pi, \pi]$ into bins
# 2. Count spikes in each phase bin
# 3. Normalize by the time spent at each phase (**occupancy**)
# 4. Plot firing rate as a function of theta phase
#
# **Important considerations:**
# - Need a **narrowband** filter so instantaneous phase is meaningful
# - Only include times with sufficient theta **power** (phase is unreliable
#   when power is low)
# - **Edge artifacts** from filtering can distort phase near segment boundaries
# - Spikes within **bursts** are not independent observations

# %%
# Compute phase tuning curve
N_PHASE_BINS = 36  # 10 degrees per bin
phase_bin_edges = np.linspace(-np.pi, np.pi, N_PHASE_BINS + 1)
phase_bin_centers = (phase_bin_edges[:-1] + phase_bin_edges[1:]) / 2

# Count spikes in each phase bin
spike_counts_by_phase, _ = np.histogram(spike_theta_phases, bins=phase_bin_edges)

# Compute occupancy: how much time was spent at each phase
all_phase_counts, _ = np.histogram(theta_instantaneous_phase, bins=phase_bin_edges)
occupancy_seconds_by_phase = all_phase_counts / actual_sampling_rate

# Firing rate = spike counts / occupancy
with np.errstate(divide="ignore", invalid="ignore"):
    firing_rate_by_phase = spike_counts_by_phase / occupancy_seconds_by_phase
    firing_rate_by_phase[~np.isfinite(firing_rate_by_phase)] = 0

# %%
# Plot phase tuning curve — linear and polar
fig = plt.figure(figsize=(12, 4), layout="constrained")
ax_linear = fig.add_subplot(1, 2, 1)
ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

# Linear plot
ax_linear.bar(phase_bin_centers, firing_rate_by_phase, width=np.diff(phase_bin_edges)[0] * 0.9, color="steelblue", alpha=0.7)
ax_linear.set(
    xlabel="Theta phase (radians)",
    ylabel="Firing rate (Hz)",
    title=f"Unit {unit_index}: Phase Tuning Curve",
    xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    xticklabels=["−π", "−π/2", "0", "π/2", "π"],
)
ax_linear.spines[["top", "right"]].set_visible(False)

# Polar plot — close the circle by appending the first value
polar_angles = np.concatenate([phase_bin_centers, [phase_bin_centers[0] + 2 * np.pi]])
polar_rates = np.concatenate([firing_rate_by_phase, [firing_rate_by_phase[0]]])
ax_polar.plot(polar_angles, polar_rates, color="steelblue", linewidth=2)
ax_polar.fill(polar_angles, polar_rates, color="steelblue", alpha=0.3)
ax_polar.set_title(f"Unit {unit_index}: Polar Tuning Curve", pad=20)

# %% [markdown]
# The phase tuning curve reveals whether this neuron has a preferred theta
# phase. A strong peak indicates phase locking; a flat curve indicates no
# phase preference.

# %% [markdown]
# ### Circular Statistics
#
# Phases are **circular** quantities — 0 and $2\pi$ are the same point.
# Standard statistics (mean, variance) don't work because they don't respect
# this circularity.
#
# **Example**: The arithmetic mean of $-170°$ and $+170°$ is $0°$, but the
# correct circular mean is $180°$ — the angles are actually clustered near
# $\pi$.
#
# We need **circular statistics**:
#
# | Statistic | Description | Python |
# |-----------|-------------|--------|
# | Circular mean | Direction of the mean resultant vector (preferred phase) | `scipy.stats.circmean` |
# | Mean resultant length ($R$) | Concentration of phases (0 = uniform, 1 = identical) | `abs(mean(exp(1j * phases)))` |
# | Circular variance | $1 - R$ (spread of phases) | `scipy.stats.circvar` |
# | Rayleigh test | Significance test for non-uniformity | Manual implementation |

# %%
# Build intuition: three phase distributions with different concentration
np.random.seed(42)

# Von Mises distributions with different concentrations (kappa)
demo_distributions = {
    "Strongly locked\n(κ=3)": np.random.vonmises(mu=np.pi / 4, kappa=3, size=200),
    "Weakly locked\n(κ=0.5)": np.random.vonmises(mu=np.pi / 4, kappa=0.5, size=200),
    "Uniform\n(κ=0)": np.random.uniform(-np.pi, np.pi, size=200),
}

fig, axes = plt.subplots(
    1, 3, figsize=(12, 4), subplot_kw={"projection": "polar"}, layout="constrained",
)

for ax, (distribution_label, demo_phases) in zip(axes, demo_distributions.items()):
    # Histogram
    histogram_counts, histogram_edges = np.histogram(demo_phases, bins=36, range=(-np.pi, np.pi))
    histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2
    histogram_widths = np.diff(histogram_edges)
    ax.bar(histogram_centers, histogram_counts, width=histogram_widths, alpha=0.5, color="steelblue")

    # Mean resultant vector: R = |mean(e^{i*phi})|, direction = angle(mean(e^{i*phi}))
    demo_mean_vector = np.mean(np.exp(1j * demo_phases))
    demo_mean_direction = np.angle(demo_mean_vector)
    demo_resultant_length = np.abs(demo_mean_vector)

    # Draw the mean vector as an arrow
    ax.annotate(
        "",
        xy=(demo_mean_direction, demo_resultant_length * ax.get_ylim()[1]),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", linewidth=2),
    )
    ax.set_title(f"{distribution_label}\nR = {demo_resultant_length:.2f}", pad=15)

fig.suptitle("Circular Distributions and Mean Resultant Vector", fontsize=12, y=1.02)

# %% [markdown]
# The red arrow shows the **mean resultant vector**: its direction is the
# circular mean (preferred phase), and its length ($R$) measures how
# concentrated the phases are. When phases are tightly clustered, $R$ is
# close to 1; when uniformly distributed, $R$ is close to 0.

# %%
# Compute circular statistics on real spike phases
preferred_phase = circmean(spike_theta_phases, high=np.pi, low=-np.pi)

# Mean resultant length: R = |mean(e^{i*phi})|
# This is the standard measure of phase concentration (0 = uniform, 1 = identical)
mean_resultant_length = np.abs(np.mean(np.exp(1j * spike_theta_phases)))

circular_variance = circvar(spike_theta_phases, high=np.pi, low=-np.pi)

print(f"Preferred theta phase: {np.degrees(preferred_phase):.1f}°")
print(f"Mean resultant length (R): {mean_resultant_length:.4f}")
print(f"Circular variance (1 - R): {circular_variance:.4f}")

# %%
# Rayleigh test of circular uniformity
#
# Tests whether phases are significantly non-uniform (clustered).
# - H0: phases are uniformly distributed (no phase preference)
# - H1: phases are unimodally clustered (one preferred phase)


def rayleigh_test(phases):
    """Test whether phases are uniformly distributed on the circle.

    Parameters
    ----------
    phases : np.ndarray
        Array of phase angles in radians.

    Returns
    -------
    z_statistic : float
        Rayleigh's Z statistic (n * R^2), where R is the mean resultant
        length.
    p_value : float
        P-value under the null hypothesis of uniform distribution.

    Notes
    -----
    Uses the approximation from Zar (2009), p. 617, which is the same
    formula used by pycircstat. The test statistic is Z = n * r^2, where
    r is the mean resultant length and n is the number of observations.

    References
    ----------
    Zar, J.H. (2009). Biostatistical Analysis, 5th ed.
    """
    n_observations = len(phases)
    sum_cos = np.sum(np.cos(phases))
    sum_sin = np.sum(np.sin(phases))

    # Total resultant length R = sqrt(C^2 + S^2), and mean resultant r = R/n
    total_resultant_length = np.sqrt(sum_cos**2 + sum_sin**2)

    # Rayleigh's Z statistic: Z = R^2 / n = n * r^2
    z_statistic = total_resultant_length**2 / n_observations

    # P-value approximation (Zar, 2009, p. 617)
    p_value = np.exp(
        np.sqrt(1 + 4 * n_observations + 4 * (n_observations**2 - total_resultant_length**2))
        - (1 + 2 * n_observations)
    )

    return z_statistic, p_value


rayleigh_z, rayleigh_p = rayleigh_test(spike_theta_phases)
print(f"Rayleigh test: Z = {rayleigh_z:.2f}, p = {rayleigh_p:.2e}")

if rayleigh_p < 0.05:
    print("→ Significant phase locking (reject uniform null)")
else:
    print("→ No significant phase locking (cannot reject uniform null)")

# %%
# Polar plot of spike phase distribution with mean vector and statistics
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"}, layout="constrained")

# Phase histogram
histogram_counts, histogram_edges = np.histogram(spike_theta_phases, bins=N_PHASE_BINS, range=(-np.pi, np.pi))
histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2
histogram_widths = np.diff(histogram_edges)
ax.bar(histogram_centers, histogram_counts, width=histogram_widths, alpha=0.5, color="steelblue")

# Mean resultant vector (scaled to histogram max)
max_count = histogram_counts.max()
ax.annotate(
    "",
    xy=(preferred_phase, mean_resultant_length * max_count),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", color="red", linewidth=3),
)

ax.set_title(
    f"Unit {unit_index}: Spike-Theta Phase Distribution\n"
    f"Preferred phase = {np.degrees(preferred_phase):.0f}°, "
    f"R = {mean_resultant_length:.3f}, "
    f"p = {rayleigh_p:.1e}",
    pad=20,
    fontsize=10,
)

# %% [markdown]
# **Important caveats for circular statistics:**
#
# - The Rayleigh test assumes each spike is an **independent observation**.
#   Burst firing violates this — a burst of 5 spikes in one theta cycle
#   counts as 5 observations, but they come from a single independent event.
# - The Rayleigh test assumes a **unimodal** alternative. A neuron that fires
#   at two opposite phases (bimodal locking) would not be detected.
# - With very large $n$ (thousands of spikes), even tiny $R$ values become
#   "significant." Always report $R$ alongside $p$-values as an effect size.

# %% [markdown]
# ### Non-Parametric Spike-Field Coherence
#
# The phase tuning curve approach requires choosing a frequency band *a
# priori* (we chose theta). **Spike-field coherence** (SFC) is a
# non-parametric method that asks: at which frequencies is spiking
# consistently phase-locked to the LFP?
#
# The approach treats:
# - The LFP as a continuous signal
# - The spike train as a point process (converted to a binary time series)
#
# We compute coherence between these two "signals" using the same multitaper
# method from Notebook 3b. Peaks in the SFC spectrum reveal the frequencies
# at which spikes are most phase-locked to the LFP, without pre-specifying
# a frequency band.

# %%
# Create binary spike train at the downsampled LFP sampling rate
# Note: if two spikes fall in the same sample (possible during bursts at 1250 Hz),
# only one is counted. This is a simplification that slightly underestimates SFC
# for bursting neurons.
binary_spike_train = np.zeros(len(lfp_time))

# For each spike, find the nearest LFP sample
spike_sample_indices = np.searchsorted(lfp_time, spike_times_relative)
valid_indices_mask = spike_sample_indices < len(binary_spike_train)
spike_sample_indices = spike_sample_indices[valid_indices_mask]
binary_spike_train[spike_sample_indices] = 1

print(f"Spike train length: {len(binary_spike_train)} samples")
print(f"Spikes placed: {int(binary_spike_train.sum())}")
print(f"Mean rate: {binary_spike_train.sum() / (len(binary_spike_train) / actual_sampling_rate):.2f} Hz")

# %%
# Compute spike-field coherence
spike_field_data = np.stack(
    [lfp_channel_1_detrended, binary_spike_train], axis=-1
)[:, np.newaxis, :]

multitaper_spike_field = Multitaper(
    spike_field_data,
    sampling_frequency=actual_sampling_rate,
    time_halfbandwidth_product=4,
    time_window_duration=1.0,
    time_window_step=0.5,
)
connectivity_spike_field = Connectivity.from_multitaper(multitaper_spike_field)
spike_field_coherence = connectivity_spike_field.coherence_magnitude()
spike_field_frequencies = connectivity_spike_field.frequencies

# Average across time windows
mean_spike_field_coherence = np.mean(spike_field_coherence[:, :, 0, 1], axis=0)

# %%
# Plot spike-field coherence spectrum
fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")

ax.plot(spike_field_frequencies, mean_spike_field_coherence, color="black", linewidth=1)
ax.axvspan(THETA_LOW, THETA_HIGH, alpha=0.15, color="orange", label="Theta (4–12 Hz)")
ax.set(
    xlabel="Frequency (Hz)",
    ylabel="Spike-Field Coherence",
    title=f"Unit {unit_index}: Spike-Field Coherence Spectrum",
    xlim=(0, 100),
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# A peak in the theta band confirms that this neuron's spiking is
# phase-locked to theta — the same conclusion as the phase tuning curve,
# but reached without pre-specifying the frequency band. Spike-field
# coherence also reveals whether the neuron is locked to other rhythms
# (e.g., gamma).
#
# **Note**: Spike-field coherence depends on firing rate — neurons with more
# spikes produce more reliable estimates. This makes it difficult to compare
# SFC across neurons with very different firing rates. The harmonic Poisson
# regression below addresses this by explicitly modeling rate.

# %% [markdown]
# ### Harmonic Poisson Regression
#
# The methods above give descriptive statistics (preferred phase, $R$, SFC),
# but they don't account for other factors that affect firing rate (e.g.,
# position, speed). The **harmonic Poisson regression** extends the GLM
# framework from Week 2b to model phase locking while controlling for
# covariates.
#
# The model adds cosine and sine of the LFP phase as predictors:
#
# $$\log(\lambda) = \beta_0 + \beta_{\cos} \cos(\theta) + \beta_{\sin} \sin(\theta)$$
#
# where $\theta$ is the instantaneous theta phase.
#
# This can be rewritten as a single cosine with amplitude and phase offset:
#
# $$\beta_{\cos} \cos(\theta) + \beta_{\sin} \sin(\theta) = \kappa \cos(\theta - \varphi)$$
#
# where:
# - $\kappa = \sqrt{\beta_{\cos}^2 + \beta_{\sin}^2}$ is the **modulation
#   depth** — how much firing rate varies with theta phase
# - $\varphi = \text{atan2}(\beta_{\sin}, \beta_{\cos})$ is the **preferred
#   phase**
#
# **Advantages over circular statistics:**
# 1. Accounts for baseline firing rate
# 2. Can add other covariates (position, speed, direction)
# 3. Provides confidence intervals on modulation and preferred phase
# 4. Model comparison via AIC and likelihood ratio tests

# %%
# Prepare data for GLM — bin spikes at 10 ms resolution
GLM_BIN_SIZE = 0.01  # seconds (10 ms)

glm_bin_edges = np.arange(0, ANALYSIS_DURATION, GLM_BIN_SIZE)
glm_bin_centers = glm_bin_edges[:-1] + GLM_BIN_SIZE / 2

# Count spikes in each bin
glm_spike_counts, _ = np.histogram(spike_times_relative, bins=glm_bin_edges)

# Interpolate theta phase and amplitude at bin centers
theta_phase_at_bins = np.interp(glm_bin_centers, lfp_time, theta_instantaneous_phase)
theta_amplitude_at_bins = np.interp(glm_bin_centers, lfp_time, theta_instantaneous_amplitude)

# Create DataFrame with cosine and sine of phase
glm_dataframe = pd.DataFrame({
    "spike_count": glm_spike_counts,
    "theta_phase": theta_phase_at_bins,
    "cosine_phase": np.cos(theta_phase_at_bins),
    "sine_phase": np.sin(theta_phase_at_bins),
    "theta_amplitude": theta_amplitude_at_bins,
})

print(f"Number of time bins: {len(glm_dataframe)}")
print(f"Total spikes: {glm_spike_counts.sum()}")
print(f"Mean spikes per bin: {glm_spike_counts.mean():.4f}")

# %%
# Fit the null (constant rate) model
model_null = smf.glm("spike_count ~ 1", data=glm_dataframe, family=sm.families.Poisson())
results_null = model_null.fit()

null_rate_hz = np.exp(results_null.params["Intercept"]) / GLM_BIN_SIZE
print(f"Null model: constant rate = {null_rate_hz:.2f} Hz")
print(f"Null model AIC: {results_null.aic:.1f}")

# %%
# Fit the harmonic phase model
model_phase = smf.glm(
    "spike_count ~ cosine_phase + sine_phase",
    data=glm_dataframe,
    family=sm.families.Poisson(),
)
results_phase = model_phase.fit()

print(results_phase.summary())

# %%
# Extract modulation depth and preferred phase from GLM coefficients
beta_cosine = results_phase.params["cosine_phase"]
beta_sine = results_phase.params["sine_phase"]

modulation_depth = np.sqrt(beta_cosine**2 + beta_sine**2)
preferred_phase_glm = np.arctan2(beta_sine, beta_cosine)

print(f"β_cos = {beta_cosine:.4f}")
print(f"β_sin = {beta_sine:.4f}")
print()
print(f"Modulation depth (κ): {modulation_depth:.4f}")
print(f"Preferred phase (GLM): {np.degrees(preferred_phase_glm):.1f}°")
print(f"Preferred phase (circular mean): {np.degrees(preferred_phase):.1f}°")

# %%
# Model comparison: Does theta phase improve the model?
likelihood_ratio_statistic = 2 * (results_phase.llf - results_null.llf)
likelihood_ratio_p_value = chi2.sf(likelihood_ratio_statistic, df=2)  # 2 extra params (cos, sin)

print("Model Comparison")
print("-" * 45)
print(f"Null model AIC:  {results_null.aic:.1f}")
print(f"Phase model AIC: {results_phase.aic:.1f}")
print(f"ΔAIC: {results_phase.aic - results_null.aic:.1f}")
print(f"\nLikelihood ratio test: χ² = {likelihood_ratio_statistic:.2f}, p = {likelihood_ratio_p_value:.2e}")

# %%
# Plot GLM predictions over the phase tuning curve
phase_grid = np.linspace(-np.pi, np.pi, 100)
prediction_dataframe = pd.DataFrame({
    "cosine_phase": np.cos(phase_grid),
    "sine_phase": np.sin(phase_grid),
})

glm_predictions = results_phase.get_prediction(prediction_dataframe)
predicted_firing_rate = glm_predictions.predicted_mean / GLM_BIN_SIZE
predicted_confidence_interval = glm_predictions.conf_int() / GLM_BIN_SIZE

fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

# Histogram estimate
bar_width = np.diff(phase_bin_edges)[0] * 0.9
ax.bar(
    phase_bin_centers,
    firing_rate_by_phase,
    width=bar_width,
    alpha=0.4,
    color="steelblue",
    label="Histogram estimate",
)

# GLM fit with confidence interval
ax.fill_between(
    phase_grid,
    predicted_confidence_interval[:, 0],
    predicted_confidence_interval[:, 1],
    alpha=0.3,
    color="coral",
)
ax.plot(phase_grid, predicted_firing_rate, color="coral", linewidth=2, label="Harmonic GLM fit")

# Mark preferred phase
ax.axvline(preferred_phase_glm, color="red", linestyle="--", alpha=0.5, label=f"Preferred phase ({np.degrees(preferred_phase_glm):.0f}°)")

ax.set(
    xlabel="Theta phase (radians)",
    ylabel="Firing rate (Hz)",
    title=f"Unit {unit_index}: Harmonic Poisson Regression (κ = {modulation_depth:.3f})",
    xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    xticklabels=["−π", "−π/2", "0", "π/2", "π"],
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The harmonic Poisson regression captures the same phase preference as the
# histogram approach, but additionally provides:
# - A smooth, parametric estimate of the tuning curve
# - Confidence intervals on the predicted firing rate
# - A formal statistical test (likelihood ratio) for phase modulation
# - The modulation depth $\kappa$ as a standardized effect size
#
# **Key advantage**: We can extend this model by adding other covariates
# (position, speed, direction) to ask whether phase locking persists after
# accounting for spatial tuning — something the histogram approach cannot do.

# %%
# Extension: add theta amplitude as a covariate
model_phase_and_amplitude = smf.glm(
    "spike_count ~ cosine_phase + sine_phase + theta_amplitude",
    data=glm_dataframe,
    family=sm.families.Poisson(),
)
results_phase_and_amplitude = model_phase_and_amplitude.fit()

amplitude_coefficient = results_phase_and_amplitude.params["theta_amplitude"]

print("Phase + Amplitude model:")
print(f"  AIC: {results_phase_and_amplitude.aic:.1f} (vs. phase-only: {results_phase.aic:.1f})")
print(f"  Theta amplitude β = {amplitude_coefficient:.4f}")
print(f"  → {100 * (np.exp(amplitude_coefficient) - 1):.1f}% rate change per unit amplitude increase")

# %% [markdown]
# ## Part 3: Cross-Frequency Coupling
#
# So far we have measured coupling *within* the same frequency band: theta
# coherence between channels, spike locking to theta phase. But brain
# rhythms at **different frequencies** can also interact.
#
# **Phase-amplitude coupling (PAC)** measures whether the amplitude (power)
# of a fast oscillation is modulated by the phase of a slow oscillation. In
# the hippocampus, gamma amplitude (~30–80 Hz) is often strongest at a
# particular phase of the theta cycle (~6–10 Hz). This "theta-gamma coupling"
# is thought to organize neural computation into discrete gamma-frequency
# processing cycles nested within each theta period.

# %% [markdown]
# ### Extracting Theta Phase and Gamma Amplitude
#
# The analysis requires two ingredients from the same LFP signal:
# 1. The **phase** of the slow rhythm (theta) — already computed above
# 2. The **amplitude** of the fast rhythm (gamma) — we need to extract this

# %%
# Apply gamma bandpass filter and extract amplitude envelope
GAMMA_LOW = 30  # Hz
GAMMA_HIGH = 80  # Hz

n_gamma_filter_taps = int(3 * actual_sampling_rate / GAMMA_LOW)
if n_gamma_filter_taps % 2 == 0:
    n_gamma_filter_taps += 1

gamma_filter_coefficients = firwin(
    n_gamma_filter_taps,
    [GAMMA_LOW, GAMMA_HIGH],
    pass_zero=False,
    fs=actual_sampling_rate,
)

gamma_filtered_lfp = filtfilt(gamma_filter_coefficients, 1.0, lfp_channel_1_detrended)
gamma_analytic_signal = hilbert(gamma_filtered_lfp)
gamma_instantaneous_amplitude = np.abs(gamma_analytic_signal)

print(f"Gamma filter: {n_gamma_filter_taps} taps ({n_gamma_filter_taps / actual_sampling_rate * 1000:.0f} ms)")

# %%
# Visualize theta phase and gamma amplitude together
PLOT_START_PAC = 10.0  # seconds
PLOT_DURATION_PAC = 1.5  # seconds
pac_time_mask = (lfp_time >= PLOT_START_PAC) & (lfp_time < PLOT_START_PAC + PLOT_DURATION_PAC)

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")

# Theta-filtered LFP
ax = axes[0]
ax.plot(lfp_time[pac_time_mask], theta_filtered_lfp[pac_time_mask], color="steelblue")
ax.set(ylabel="Voltage (µV)", title="Theta-Filtered LFP (4–12 Hz)")
ax.spines[["top", "right"]].set_visible(False)

# Gamma-filtered LFP with amplitude envelope
ax = axes[1]
ax.plot(lfp_time[pac_time_mask], gamma_filtered_lfp[pac_time_mask], color="gray", linewidth=0.5)
ax.plot(lfp_time[pac_time_mask], gamma_instantaneous_amplitude[pac_time_mask], color="coral", linewidth=1.5)
ax.plot(lfp_time[pac_time_mask], -gamma_instantaneous_amplitude[pac_time_mask], color="coral", linewidth=1.5)
ax.set(ylabel="Voltage (µV)", title="Gamma-Filtered LFP (30–80 Hz) with Envelope")
ax.spines[["top", "right"]].set_visible(False)

# Theta phase
ax = axes[2]
ax.plot(lfp_time[pac_time_mask], theta_instantaneous_phase[pac_time_mask], color="green", linewidth=0.8)
ax.set(
    xlabel="Time (s)",
    ylabel="Phase (radians)",
    title="Theta Instantaneous Phase",
    yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    yticklabels=["−π", "−π/2", "0", "π/2", "π"],
)
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# Look for a pattern: does gamma amplitude (red envelope) tend to peak at
# certain phases of the theta cycle? If so, there is phase-amplitude
# coupling.

# %% [markdown]
# ### Quantifying Phase-Amplitude Coupling: Modulation Index
#
# A simple approach to quantify PAC:
# 1. Divide theta phase into bins
# 2. Compute the **mean gamma amplitude** in each phase bin
# 3. If PAC exists, gamma amplitude varies systematically with theta phase
#
# To measure PAC strength, we compute the **Modulation Index** (Tort et al.,
# 2010), which uses Kullback-Leibler divergence to measure how far the
# amplitude distribution across phase bins deviates from uniform:
#
# $$\text{MI} = \frac{D_{KL}(P \| U)}{\log(N)}$$
#
# where $P$ is the normalized amplitude distribution across $N$ phase bins
# and $U$ is the uniform distribution. The denominator normalizes MI to
# lie between 0 (uniform, no PAC) and 1 (all amplitude in one bin, maximum PAC).

# %%
# Compute mean gamma amplitude in each theta phase bin
N_PAC_BINS = 18  # 20 degrees per bin
pac_bin_edges = np.linspace(-np.pi, np.pi, N_PAC_BINS + 1)
pac_bin_centers = (pac_bin_edges[:-1] + pac_bin_edges[1:]) / 2

mean_gamma_amplitude_by_phase = np.zeros(N_PAC_BINS)
for bin_index in range(N_PAC_BINS):
    phase_mask = (theta_instantaneous_phase >= pac_bin_edges[bin_index]) & (
        theta_instantaneous_phase < pac_bin_edges[bin_index + 1]
    )
    if phase_mask.sum() > 0:
        mean_gamma_amplitude_by_phase[bin_index] = gamma_instantaneous_amplitude[phase_mask].mean()

# Compute Modulation Index (Tort et al., 2010)
amplitude_distribution = mean_gamma_amplitude_by_phase / mean_gamma_amplitude_by_phase.sum()
uniform_distribution = np.ones(N_PAC_BINS) / N_PAC_BINS

# KL divergence: sum(P * log(P / Q))
kl_divergence = np.sum(amplitude_distribution * np.log(amplitude_distribution / uniform_distribution))
modulation_index = kl_divergence / np.log(N_PAC_BINS)

print(f"Modulation Index: {modulation_index:.6f}")

# %%
# Plot the PAC diagram
fig = plt.figure(figsize=(12, 4), layout="constrained")
ax_linear = fig.add_subplot(1, 2, 1)
ax_polar = fig.add_subplot(1, 2, 2, projection="polar")

# Linear plot
ax_linear.bar(
    pac_bin_centers,
    mean_gamma_amplitude_by_phase,
    width=np.diff(pac_bin_edges)[0] * 0.9,
    color="coral",
    alpha=0.7,
)
ax_linear.set(
    xlabel="Theta phase (radians)",
    ylabel="Mean gamma amplitude (µV)",
    title=f"Phase-Amplitude Coupling (MI = {modulation_index:.4f})",
    xticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    xticklabels=["−π", "−π/2", "0", "π/2", "π"],
)
ax_linear.spines[["top", "right"]].set_visible(False)

# Polar plot
polar_pac_angles = np.concatenate([pac_bin_centers, [pac_bin_centers[0] + 2 * np.pi]])
polar_pac_amplitudes = np.concatenate([mean_gamma_amplitude_by_phase, [mean_gamma_amplitude_by_phase[0]]])
ax_polar.plot(polar_pac_angles, polar_pac_amplitudes, color="coral", linewidth=2)
ax_polar.fill(polar_pac_angles, polar_pac_amplitudes, color="coral", alpha=0.3)
ax_polar.set_title("Gamma Amplitude by Theta Phase", pad=20)

# %% [markdown]
# ### Surrogate Test for PAC Significance
#
# To assess whether the observed MI is significant, we create **surrogate
# data** by circularly shifting the gamma amplitude time series relative to
# the theta phase. This preserves the autocorrelation structure of each
# signal but destroys their temporal relationship. If the observed MI is
# larger than 95% of surrogate MIs, the coupling is significant.

# %%
# Surrogate distribution for PAC significance
N_SURROGATES = 200
surrogate_modulation_indices = np.zeros(N_SURROGATES)
n_lfp_samples = len(gamma_instantaneous_amplitude)

# Pre-compute bin assignments once (avoids recomputing on every surrogate iteration)
phase_bin_assignments = np.clip(
    np.digitize(theta_instantaneous_phase, pac_bin_edges) - 1, 0, N_PAC_BINS - 1
)

np.random.seed(0)
for surrogate_index in range(N_SURROGATES):
    # Circular shift: shift by at least 1 second to break coupling
    shift_amount = np.random.randint(int(actual_sampling_rate), n_lfp_samples - int(actual_sampling_rate))
    gamma_amplitude_shifted = np.roll(gamma_instantaneous_amplitude, shift_amount)

    # Compute mean amplitude by phase bin for surrogate
    surrogate_amplitude_by_phase = np.array([
        gamma_amplitude_shifted[phase_bin_assignments == bin_index].mean()
        for bin_index in range(N_PAC_BINS)
    ])

    surrogate_distribution = surrogate_amplitude_by_phase / surrogate_amplitude_by_phase.sum()
    surrogate_kl = np.sum(surrogate_distribution * np.log(surrogate_distribution / uniform_distribution))
    surrogate_modulation_indices[surrogate_index] = surrogate_kl / np.log(N_PAC_BINS)

pac_p_value = np.mean(surrogate_modulation_indices >= modulation_index)
print(f"Observed MI: {modulation_index:.6f}")
print(f"Surrogate MI: {np.mean(surrogate_modulation_indices):.6f} ± {np.std(surrogate_modulation_indices):.6f}")
print(f"PAC p-value (surrogate test): {pac_p_value:.4f}")

# %%
# Plot surrogate distribution with observed MI
fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

ax.hist(
    surrogate_modulation_indices,
    bins=30,
    density=True,
    alpha=0.6,
    color="gray",
    label="Surrogate distribution",
)
ax.axvline(
    modulation_index,
    color="coral",
    linewidth=2,
    linestyle="--",
    label=f"Observed MI = {modulation_index:.4f}",
)
ax.axvline(
    np.percentile(surrogate_modulation_indices, 95),
    color="black",
    linewidth=1,
    linestyle=":",
    label="95th percentile",
)
ax.set(
    xlabel="Modulation Index",
    ylabel="Density",
    title=f"PAC Significance Test (p = {pac_p_value:.3f})",
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# ## Exercises
#
# Try these on your own to deepen your understanding:
#
# 1. **Multiple neurons**: Repeat the spike-LFP analysis for several good
#    units. Do all neurons show the same preferred theta phase? Compute $R$
#    and preferred phase for each and plot them as vectors on a polar plot.
#
# 2. **Power threshold**: When computing spike phases, only include spikes
#    that occur during periods of high theta power (e.g., theta amplitude
#    above the median). Does this change the estimated $R$ and preferred
#    phase?
#
# 3. **PLI between distant channels**: Compute the PLI between the theta
#    reference channel and several other channels at increasing distances
#    on the probe. How does PLI change with distance? Compare to coherence.
#
# 4. **Full harmonic regression**: Extend the harmonic Poisson regression
#    to include position (splines), speed, and theta phase simultaneously
#    (combining the model from 02b with the phase terms from this notebook).
#    Does the preferred phase change when controlling for position?
#
# 5. **Higher harmonics**: Add `np.cos(2 * theta_phase)` and
#    `np.sin(2 * theta_phase)` to the GLM to capture bimodal phase tuning.
#    Does AIC improve?
#
# 6. **Gamma sub-bands**: Repeat the PAC analysis using slow gamma
#    (30–50 Hz) and fast gamma (50–80 Hz) separately. Do they couple to
#    different theta phases?

# %% [markdown]
# ## Summary
#
# In this notebook, we learned how to:
#
# 1. **Identify limitations of coherence** and use alternative measures
#    (PLV, PLI) that are robust to volume conduction
# 2. **Construct phase tuning curves** for spike-LFP relationships
# 3. **Apply circular statistics** to quantify phase locking strength
# 4. **Implement the Rayleigh test** for significance of phase locking
# 5. **Compute spike-field coherence** to identify locking frequencies
# 6. **Build harmonic Poisson regression** models for spike-phase coupling
# 7. **Measure phase-amplitude coupling** between theta and gamma
#
# ### Key Concepts
#
# | Concept | Description |
# |---------|-------------|
# | Phase Locking Value (PLV) | Coherence ignoring power; sensitive to noise in low-power trials |
# | Phase Lag Index (PLI) | Coupling measure robust to volume conduction (zero-lag artifacts) |
# | Phase tuning curve | Firing rate as a function of oscillatory phase |
# | Circular mean | Mean direction of phase angles (preferred phase) |
# | Mean resultant length ($R$) | Strength of phase clustering (0 = uniform, 1 = identical) |
# | Rayleigh test | Tests for significant non-uniform phase distribution |
# | Spike-field coherence | Frequency-resolved spike-LFP coupling (no pre-specified band) |
# | Harmonic Poisson regression | GLM with cos/sin phase covariates for rate modeling |
# | Modulation depth ($\kappa$) | GLM-based measure of firing rate modulation by phase |
# | Phase-amplitude coupling | Coupling between slow-rhythm phase and fast-rhythm amplitude |
# | Modulation Index (MI) | KL-divergence-based PAC strength measure (Tort et al., 2010) |
#
# ### Python Techniques Used
#
# - **`spectral_connectivity.Connectivity.phase_locking_value()`** for PLV
# - **`spectral_connectivity.Connectivity.phase_lag_index()`** for PLI
# - **`scipy.stats.circmean`** and **`scipy.stats.circvar`** for circular statistics
# - **`np.exp(1j * phases)`** for computing mean resultant length on the unit circle
# - **`statsmodels.formula.api.glm`** with cos/sin covariates for harmonic regression
# - **`np.interp`** for assigning LFP phase to spike times
# - **`np.searchsorted`** for aligning spike times to LFP samples
#
# ### Next Steps
#
# In Notebook 04, we move from characterizing neural representations to
# **decoding** — using neural activity to reconstruct the animal's position.
# This connects the encoding models we've built (place fields, phase tuning)
# to computational readout of population activity.

# %% [markdown]
# ## Cleanup

# %%
# Close file handles
behavior_io.close()
lfp_io.close()
