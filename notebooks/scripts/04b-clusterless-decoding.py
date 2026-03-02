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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Week 4b: Clusterless Decoding Approaches
#
# In notebook 04a, we built a Bayesian decoder that uses **sorted spike
# counts** — spikes that have been assigned to individual neurons through
# spike sorting. But spike sorting is an imperfect process that can lose
# information. This notebook introduces **clusterless decoding**, which
# bypasses spike sorting entirely by decoding directly from spike waveform
# features.
#
# We'll use **simulated data** throughout to keep things transparent and
# computationally tractable.
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# 1. Understand why spike sorting loses information and how clusterless
#    methods address this
# 2. Explain the marked point process framework for clusterless decoding
# 3. Implement a toy clusterless decoder using kernel density estimation
# 4. Compare sorted vs clusterless decoding on simulated data
# 5. See when clusterless decoding outperforms sorted decoding

# %% [markdown]
# ## Setup

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

# %% [markdown]
# ## Part 1: Why Clusterless?
#
# ### The Spike Sorting Pipeline
#
# Traditional neural decoding relies on spike sorting — a multi-step process:
#
# 1. **Detect** threshold crossings in the raw voltage trace
# 2. **Extract features** from each spike waveform (e.g., peak amplitude,
#    principal components)
# 3. **Cluster** spikes into groups based on waveform similarity
# 4. **Assign** each spike to a putative neuron
# 5. **Compute** firing rates per neuron → build encoding model → decode
#
# ### Where Information Is Lost
#
# Each step in this pipeline can discard useful information:
#
# - **Ambiguous spikes are discarded**: Spikes that fall between clusters
#   (overlapping waveforms from nearby neurons) are often thrown away
# - **Multi-unit activity (MUA) is ignored**: Spikes that can't be cleanly
#   assigned to a single neuron are typically excluded
# - **Cluster boundaries are subjective**: Different experimenters may draw
#   cluster boundaries differently, leading to different neuron assignments
# - **Spike sorting errors**: Spikes can be assigned to the wrong neuron,
#   especially when neurons have similar waveforms
#
# ### The Key Insight
#
# Different neurons produce spikes with different waveform shapes (amplitudes,
# widths, etc.). Different neurons also have different place fields. This means
# **waveform features carry information about position** — even without knowing
# which neuron produced a spike.
#
# The clusterless approach exploits this: instead of asking "which neuron fired
# and where is its place field?", we ask "given a spike with *this* waveform
# feature at *this* time, where is the animal?"

# %% [markdown]
# ## Part 2: Simulated Data — A 1D Linear Track
#
# We'll simulate a simple scenario: an animal running back and forth on a
# linear track, with neurons that have both spatial tuning (place fields)
# and distinct waveform features (spike amplitudes).
#
# Using simulated data lets us:
# - Fully control the ground truth
# - Compare sorted vs clusterless decoding under identical conditions
# - Demonstrate what happens when spike clusters overlap

# %%
# Simulation parameters
N_NEURONS = 10              # Simulated place cells
N_POSITION_BINS = 50        # Spatial resolution of the 1D track
TRACK_LENGTH = 150.0        # cm
N_TIME_STEPS = 2000         # Number of time bins (~50 seconds)
TIME_BIN_SIZE = 0.025       # 25 ms bins (matching notebook 04a)
AMPLITUDE_NOISE_STD = 15.0  # Noise in spike amplitude feature (microvolts)

np.random.seed(42)

# Position bins
position_bin_edges = np.linspace(0, TRACK_LENGTH, N_POSITION_BINS + 1)
position_bin_centers = (position_bin_edges[:-1] + position_bin_edges[1:]) / 2
position_bin_size = position_bin_edges[1] - position_bin_edges[0]


# %%
def simulate_place_cells_1d(
    n_neurons: int,
    track_length: float,
    position_bin_centers: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Simulate 1D Gaussian place fields and characteristic spike amplitudes.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to simulate
    track_length : float
        Length of the linear track in cm
    position_bin_centers : np.ndarray
        Center of each position bin

    Returns
    -------
    place_fields : np.ndarray, shape (n_neurons, n_position_bins)
        Firing rate (Hz) at each position for each neuron
    neuron_amplitudes : np.ndarray, shape (n_neurons,)
        Mean spike amplitude for each neuron (microvolts)
    place_field_centers : np.ndarray, shape (n_neurons,)
        Center of each neuron's place field (cm)
    """
    place_field_centers = np.random.uniform(10, track_length - 10, n_neurons)
    place_field_widths = np.random.uniform(10, 25, n_neurons)
    peak_rates = np.random.uniform(15, 40, n_neurons)

    place_fields = np.zeros((n_neurons, len(position_bin_centers)))
    for i in range(n_neurons):
        place_fields[i] = peak_rates[i] * np.exp(
            -(position_bin_centers - place_field_centers[i]) ** 2
            / (2 * place_field_widths[i] ** 2)
        )

    # Each neuron has a characteristic amplitude, spread across a wide range
    neuron_amplitudes = np.random.uniform(50, 300, n_neurons)

    return place_fields, neuron_amplitudes, place_field_centers


# %%
def simulate_trajectory_1d(
    n_time_steps: int,
    track_length: float,
) -> NDArray[np.floating]:
    """Simulate a smooth back-and-forth trajectory on a 1D track.

    Parameters
    ----------
    n_time_steps : int
        Number of time bins
    track_length : float
        Length of the track in cm

    Returns
    -------
    np.ndarray, shape (n_time_steps,)
        Position at each time step
    """
    t = np.linspace(0, 4 * np.pi, n_time_steps)
    # Sinusoidal traversal covering most of the track
    position = (track_length / 2) * (1 + 0.85 * np.sin(t))
    return np.clip(position, 0, track_length)


# %%
def simulate_spikes_and_marks(
    true_position: NDArray[np.floating],
    place_fields: NDArray[np.floating],
    neuron_amplitudes: NDArray[np.floating],
    position_bin_edges: NDArray[np.floating],
    time_bin_size: float,
    amplitude_noise_std: float,
) -> tuple[
    NDArray[np.int32],
    list[NDArray[np.floating]],
    list[NDArray[np.int32]],
]:
    """Simulate Poisson spikes with noisy amplitude marks.

    For each time bin, each neuron fires according to a Poisson process
    with rate determined by its place field at the current position.
    Each spike gets an amplitude = neuron's mean amplitude + Gaussian noise.

    Parameters
    ----------
    true_position : np.ndarray, shape (n_time_steps,)
        Animal's position at each time step
    place_fields : np.ndarray, shape (n_neurons, n_position_bins)
        Firing rate maps
    neuron_amplitudes : np.ndarray, shape (n_neurons,)
        Mean spike amplitude per neuron
    position_bin_edges : np.ndarray
        Bin edges for the position axis
    time_bin_size : float
        Duration of each time bin (seconds)
    amplitude_noise_std : float
        Standard deviation of amplitude noise (microvolts)

    Returns
    -------
    spike_counts : np.ndarray, shape (n_neurons, n_time_steps)
        Spike counts per neuron per time bin (for sorted decoding)
    spike_amplitudes_per_bin : list of np.ndarray
        For each time bin, the amplitudes of all spikes (neuron identity unknown)
    spike_neuron_ids_per_bin : list of np.ndarray
        Ground truth neuron identity for each spike (for analysis only)
    """
    n_neurons = len(neuron_amplitudes)
    n_time_steps = len(true_position)

    # Find which position bin the animal is in at each time step
    pos_bin_idx = np.clip(
        np.searchsorted(position_bin_edges, true_position) - 1,
        0,
        place_fields.shape[1] - 1,
    )

    spike_counts = np.zeros((n_neurons, n_time_steps), dtype=np.int32)
    spike_amplitudes_per_bin = []
    spike_neuron_ids_per_bin = []

    for t in range(n_time_steps):
        bin_amplitudes = []
        bin_neuron_ids = []

        for i in range(n_neurons):
            # Expected spike count from Poisson model
            rate = place_fields[i, pos_bin_idx[t]]
            expected_count = rate * time_bin_size
            n_spikes = np.random.poisson(expected_count)
            spike_counts[i, t] = n_spikes

            # Generate amplitude marks for each spike
            for _ in range(n_spikes):
                amp = neuron_amplitudes[i] + np.random.normal(0, amplitude_noise_std)
                bin_amplitudes.append(amp)
                bin_neuron_ids.append(i)

        spike_amplitudes_per_bin.append(np.array(bin_amplitudes))
        spike_neuron_ids_per_bin.append(np.array(bin_neuron_ids, dtype=np.int32))

    return spike_counts, spike_amplitudes_per_bin, spike_neuron_ids_per_bin


# %%
# Generate simulated data
place_fields, neuron_amplitudes, place_field_centers = simulate_place_cells_1d(
    N_NEURONS, TRACK_LENGTH, position_bin_centers,
)
true_position = simulate_trajectory_1d(N_TIME_STEPS, TRACK_LENGTH)
spike_counts, spike_amplitudes_per_bin, spike_neuron_ids_per_bin = simulate_spikes_and_marks(
    true_position, place_fields, neuron_amplitudes,
    position_bin_edges, TIME_BIN_SIZE, AMPLITUDE_NOISE_STD,
)

total_spikes = spike_counts.sum()
print(f"Simulated {N_NEURONS} neurons over {N_TIME_STEPS} time bins")
print(f"Total spikes: {total_spikes}")
print(f"Mean spikes per bin: {total_spikes / N_TIME_STEPS:.1f}")

# %%
# Visualize place fields and neuron amplitudes
fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")

# Place fields
ax = axes[0]
for i in range(N_NEURONS):
    ax.plot(position_bin_centers, place_fields[i], label=f"Neuron {i}")
ax.set(xlabel="Position (cm)", ylabel="Firing rate (Hz)", title="Simulated Place Fields")
ax.spines[["top", "right"]].set_visible(False)

# Trajectory
ax = axes[1]
time_axis = np.arange(N_TIME_STEPS) * TIME_BIN_SIZE
ax.plot(time_axis, true_position, "k-", linewidth=0.5)
ax.set(xlabel="Time (s)", ylabel="Position (cm)", title="Simulated Trajectory")
ax.spines[["top", "right"]].set_visible(False)

# Spike amplitudes colored by position (the key insight plot)
ax = axes[2]
all_amplitudes = []
all_positions = []
for t in range(N_TIME_STEPS):
    n_spikes_t = len(spike_amplitudes_per_bin[t])
    if n_spikes_t > 0:
        all_amplitudes.extend(spike_amplitudes_per_bin[t])
        all_positions.extend([true_position[t]] * n_spikes_t)

sc = ax.scatter(
    all_positions, all_amplitudes, c=all_positions,
    cmap="viridis", s=1, alpha=0.3,
)
ax.set(xlabel="Position (cm)", ylabel="Spike amplitude (μV)",
       title="Amplitude carries position info")
fig.colorbar(sc, ax=ax, label="Position (cm)")
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The rightmost plot shows the key insight: even without knowing which neuron
# produced each spike, the spike amplitude is correlated with position because
# different neurons (with different amplitudes) fire at different locations.

# %% [markdown]
# ## Part 3: Sorted Decoding (Baseline)
#
# First, let's decode using the traditional sorted approach. Since we simulated
# the data, we know each spike's true neuron identity — this is the best-case
# scenario for sorted decoding.
#
# We reuse the same Bayesian decoding logic from notebook 04a, simplified for
# 1D position.

# %%
def compute_log_likelihood_1d(
    spike_counts: NDArray[np.int32],
    place_fields: NDArray[np.floating],
    time_bin_size: float,
) -> NDArray[np.floating]:
    """Compute log-likelihood of each position for each time bin (1D).

    Same algorithm as notebook 04a, but for 1D place fields.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_neurons, n_time_bins)
        Spike counts per neuron per time bin
    place_fields : np.ndarray, shape (n_neurons, n_position_bins)
        Firing rate maps in Hz
    time_bin_size : float
        Duration of each time bin in seconds

    Returns
    -------
    np.ndarray, shape (n_time_bins, n_position_bins)
        Log-likelihood of each position at each time bin
    """
    n_neurons, n_time_bins = spike_counts.shape
    n_position_bins = place_fields.shape[1]

    # Expected spike counts: lambda = rate * dt
    expected_counts = place_fields * time_bin_size
    MIN_RATE = 1e-10
    expected_counts = np.maximum(expected_counts, MIN_RATE * time_bin_size)
    log_rates = np.log(expected_counts)

    log_likelihood = np.zeros((n_time_bins, n_position_bins))

    for t in range(n_time_bins):
        n = spike_counts[:, t]  # (n_neurons,)
        # n_i * log(lambda_i(x)) summed over neurons
        spike_term = np.sum(n[:, np.newaxis] * log_rates, axis=0)
        # -lambda_i(x) summed over neurons
        rate_term = -np.sum(expected_counts, axis=0)
        log_likelihood[t] = spike_term + rate_term

    return log_likelihood


# %%
def log_likelihood_to_posterior_1d(
    log_likelihood: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Convert log-likelihood to posterior probability (1D, uniform prior).

    Uses the log-sum-exp trick for numerical stability (see notebook 04a, Part 5).

    Parameters
    ----------
    log_likelihood : np.ndarray, shape (n_time_bins, n_position_bins)

    Returns
    -------
    np.ndarray, shape (n_time_bins, n_position_bins)
        Normalized posterior probability
    """
    log_max = np.max(log_likelihood, axis=1, keepdims=True)
    posterior = np.exp(log_likelihood - log_max)
    posterior_sum = np.sum(posterior, axis=1, keepdims=True)
    return posterior / (posterior_sum + 1e-10)


# %%
def decode_map_1d(
    posterior: NDArray[np.floating],
    position_bins: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Decode position using MAP estimate (1D).

    Parameters
    ----------
    posterior : np.ndarray, shape (n_time_bins, n_position_bins)
    position_bins : np.ndarray, shape (n_position_bins,)

    Returns
    -------
    np.ndarray, shape (n_time_bins,)
        Decoded position at each time bin
    """
    return position_bins[np.argmax(posterior, axis=1)]


# %%
# Run sorted decoding
log_ll_sorted = compute_log_likelihood_1d(spike_counts, place_fields, TIME_BIN_SIZE)
posterior_sorted = log_likelihood_to_posterior_1d(log_ll_sorted)
decoded_sorted = decode_map_1d(posterior_sorted, position_bin_centers)

error_sorted = np.abs(decoded_sorted - true_position)

print("Sorted decoding (perfect spike sorting):")
print(f"  Mean error:   {np.mean(error_sorted):.1f} cm")
print(f"  Median error: {np.median(error_sorted):.1f} cm")

# %%
# Visualize sorted decoding result
fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained",
                         height_ratios=[2, 1])

ax = axes[0]
ax.plot(time_axis, true_position, "b-", linewidth=1, label="True", alpha=0.7)
ax.plot(time_axis, decoded_sorted, "r.", markersize=1, alpha=0.3, label="Decoded (sorted)")
ax.set(ylabel="Position (cm)", title="Sorted Decoding — Perfect Spike Sorting")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(time_axis, error_sorted, "k-", linewidth=0.5, alpha=0.5)
ax.axhline(np.median(error_sorted), color="red", linestyle="--",
           label=f"Median: {np.median(error_sorted):.1f} cm")
ax.set(xlabel="Time (s)", ylabel="Error (cm)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# ## Part 4: The Marked Point Process Framework
#
# ### From Sorted to Clusterless
#
# In the sorted decoder (notebook 04a), the likelihood for a time bin is:
#
# $$\log P(\mathbf{n} \mid x) = \sum_{i=1}^{N} \left[ n_i \log \lambda_i(x) - \lambda_i(x) \cdot \Delta t \right]$$
#
# where $n_i$ is the spike count for neuron $i$ and $\lambda_i(x)$ is neuron
# $i$'s firing rate at position $x$. This requires knowing which neuron
# produced each spike.
#
# ### The Clusterless Likelihood
#
# In the clusterless approach, we don't know neuron identity. Instead, each
# spike carries a **mark** — a waveform feature (here, amplitude). We model
# the **joint intensity** $\Lambda(m, x)$: the rate of observing a spike with
# mark $m$ at position $x$.
#
# The clusterless log-likelihood for a time bin with spike marks
# $m_1, m_2, \ldots, m_K$ is:
#
# $$\log L(x) = \sum_{j=1}^{K} \log \Lambda(m_j, x) - \Lambda_{\text{ground}}(x) \cdot \Delta t$$
#
# where:
# - $\Lambda(m_j, x)$ is the joint intensity evaluated at spike $j$'s mark
#   and position $x$
# - $\Lambda_{\text{ground}}(x) = \int \Lambda(m, x) \, dm$ is the **ground
#   intensity** — the total spike rate at position $x$, integrated over all
#   possible marks
#
# This has the same structure as the sorted likelihood:
# - **Spike terms**: each spike "votes" for positions where spikes with its
#   mark are commonly observed
# - **Ground term**: penalizes positions with high overall spike rates (same
#   role as the $-\lambda_i(x)$ terms in the sorted decoder)
#
# ### Estimating the Joint Intensity
#
# We estimate $\Lambda(m, x)$ non-parametrically using a smoothed 2D histogram
# of (mark, position) pairs from training data, divided by occupancy. This is
# essentially kernel density estimation (KDE).

# %%
def estimate_joint_mark_intensity(
    spike_amplitudes: NDArray[np.floating],
    spike_positions: NDArray[np.floating],
    position_bin_edges: NDArray[np.floating],
    amplitude_bin_edges: NDArray[np.floating],
    occupancy_seconds: NDArray[np.floating],
    position_bandwidth: float,
    amplitude_bandwidth: float,
) -> NDArray[np.floating]:
    """Estimate the joint mark-position intensity using a smoothed 2D histogram.

    Parameters
    ----------
    spike_amplitudes : np.ndarray
        Amplitude of each spike
    spike_positions : np.ndarray
        Position at which each spike occurred
    position_bin_edges : np.ndarray
        Bin edges for the position axis
    amplitude_bin_edges : np.ndarray
        Bin edges for the amplitude axis
    occupancy_seconds : np.ndarray, shape (n_position_bins,)
        Time spent in each position bin (seconds)
    position_bandwidth : float
        Smoothing bandwidth for position axis (in bin units)
    amplitude_bandwidth : float
        Smoothing bandwidth for amplitude axis (in bin units)

    Returns
    -------
    np.ndarray, shape (n_amplitude_bins, n_position_bins)
        Joint intensity in units of spikes / (second * amplitude_bin)
    """
    # 2D histogram: counts of spikes in each (amplitude, position) bin
    joint_counts, _, _ = np.histogram2d(
        spike_amplitudes, spike_positions,
        bins=[amplitude_bin_edges, position_bin_edges],
    )

    # Smooth with Gaussian kernel (KDE-like)
    joint_counts_smooth = gaussian_filter(
        joint_counts, sigma=[amplitude_bandwidth, position_bandwidth],
    )

    # Divide by occupancy to convert counts to rate
    # occupancy_seconds has shape (n_position_bins,), broadcast over amplitude axis
    with np.errstate(divide="ignore", invalid="ignore"):
        joint_intensity = joint_counts_smooth / occupancy_seconds[np.newaxis, :]
        joint_intensity[~np.isfinite(joint_intensity)] = 0

    return joint_intensity


# %%
def compute_ground_intensity(
    joint_intensity: NDArray[np.floating],
    amplitude_bin_size: float,
) -> NDArray[np.floating]:
    """Integrate joint intensity over marks to get the ground intensity.

    The ground intensity gives the total spike rate at each position,
    regardless of mark value.

    Parameters
    ----------
    joint_intensity : np.ndarray, shape (n_amplitude_bins, n_position_bins)
    amplitude_bin_size : float
        Width of each amplitude bin

    Returns
    -------
    np.ndarray, shape (n_position_bins,)
        Total spike rate at each position
    """
    return joint_intensity.sum(axis=0) * amplitude_bin_size


# %%
# Collect all spike amplitudes and positions for estimating the joint intensity
all_spike_amps_train = []
all_spike_pos_train = []

for t in range(N_TIME_STEPS):
    n_spikes_t = len(spike_amplitudes_per_bin[t])
    if n_spikes_t > 0:
        all_spike_amps_train.extend(spike_amplitudes_per_bin[t])
        all_spike_pos_train.extend([true_position[t]] * n_spikes_t)

all_spike_amps_train = np.array(all_spike_amps_train)
all_spike_pos_train = np.array(all_spike_pos_train)

print(f"Total spikes for training: {len(all_spike_amps_train)}")

# %%
# Define amplitude bins
N_AMPLITUDE_BINS = 40
amp_min = all_spike_amps_train.min() - 20
amp_max = all_spike_amps_train.max() + 20
amplitude_bin_edges = np.linspace(amp_min, amp_max, N_AMPLITUDE_BINS + 1)
amplitude_bin_centers = (amplitude_bin_edges[:-1] + amplitude_bin_edges[1:]) / 2
amplitude_bin_size = amplitude_bin_edges[1] - amplitude_bin_edges[0]

# Compute occupancy (time in each position bin)
pos_bin_idx = np.clip(
    np.searchsorted(position_bin_edges, true_position) - 1,
    0, N_POSITION_BINS - 1,
)
occupancy_counts = np.bincount(pos_bin_idx, minlength=N_POSITION_BINS)
occupancy_seconds = occupancy_counts * TIME_BIN_SIZE

# Estimate joint intensity
POSITION_BANDWIDTH = 2.0   # Smoothing in position bins
AMPLITUDE_BANDWIDTH = 2.0  # Smoothing in amplitude bins

joint_intensity = estimate_joint_mark_intensity(
    all_spike_amps_train, all_spike_pos_train,
    position_bin_edges, amplitude_bin_edges,
    occupancy_seconds, POSITION_BANDWIDTH, AMPLITUDE_BANDWIDTH,
)

ground_intensity = compute_ground_intensity(joint_intensity, amplitude_bin_size)

print(f"Joint intensity shape: {joint_intensity.shape}")
print(f"Ground intensity range: {ground_intensity.min():.1f} – {ground_intensity.max():.1f} Hz")

# %%
# Visualize the joint intensity
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

ax = axes[0]
im = ax.imshow(
    joint_intensity,
    origin="lower",
    extent=[0, TRACK_LENGTH, amp_min, amp_max],
    aspect="auto",
    cmap="hot",
)
ax.set(xlabel="Position (cm)", ylabel="Spike amplitude (μV)",
       title="Joint mark-position intensity Λ(m, x)")
fig.colorbar(im, ax=ax, label="Rate (spikes/s/amp_bin)")

ax = axes[1]
ax.plot(position_bin_centers, ground_intensity, "k-", linewidth=2)
ax.set(xlabel="Position (cm)", ylabel="Total spike rate (Hz)",
       title="Ground intensity (integrated over marks)")
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The joint intensity heatmap shows "ridges" — each ridge corresponds to one
# neuron's characteristic amplitude at the positions covered by its place
# field. The clusterless decoder uses this structure directly, without ever
# identifying individual neurons.

# %% [markdown]
# ## Part 5: Clusterless Decoding

# %%
def compute_clusterless_log_likelihood_1d(
    spike_amplitudes_per_bin: list[NDArray[np.floating]],
    joint_intensity: NDArray[np.floating],
    ground_intensity: NDArray[np.floating],
    amplitude_bin_edges: NDArray[np.floating],
    time_bin_size: float,
) -> NDArray[np.floating]:
    """Compute clusterless log-likelihood for each position at each time bin.

    Parameters
    ----------
    spike_amplitudes_per_bin : list of np.ndarray
        For each time bin, array of spike amplitudes
    joint_intensity : np.ndarray, shape (n_amplitude_bins, n_position_bins)
    ground_intensity : np.ndarray, shape (n_position_bins,)
    amplitude_bin_edges : np.ndarray
        Bin edges for the amplitude axis
    time_bin_size : float
        Duration of each time bin in seconds

    Returns
    -------
    np.ndarray, shape (n_time_bins, n_position_bins)
        Log-likelihood of each position at each time bin
    """
    n_time_bins = len(spike_amplitudes_per_bin)
    n_position_bins = ground_intensity.shape[0]
    n_amplitude_bins = joint_intensity.shape[0]
    log_likelihood = np.zeros((n_time_bins, n_position_bins))

    for t in range(n_time_bins):
        # Ground intensity term (penalty for expected spikes)
        log_likelihood[t] = -ground_intensity * time_bin_size

        # Mark terms: each spike contributes log(joint_intensity at its mark)
        for amp in spike_amplitudes_per_bin[t]:
            amp_bin = np.clip(
                np.searchsorted(amplitude_bin_edges, amp) - 1,
                0, n_amplitude_bins - 1,
            )
            mark_intensity = joint_intensity[amp_bin, :]
            log_likelihood[t] += np.log(mark_intensity + 1e-10)

    return log_likelihood


# %%
# Run clusterless decoding
print("Computing clusterless log-likelihood...")
log_ll_clusterless = compute_clusterless_log_likelihood_1d(
    spike_amplitudes_per_bin, joint_intensity, ground_intensity,
    amplitude_bin_edges, TIME_BIN_SIZE,
)

posterior_clusterless = log_likelihood_to_posterior_1d(log_ll_clusterless)
decoded_clusterless = decode_map_1d(posterior_clusterless, position_bin_centers)
error_clusterless = np.abs(decoded_clusterless - true_position)

print(f"\nSorted decoding:      Mean = {np.mean(error_sorted):.1f} cm, "
      f"Median = {np.median(error_sorted):.1f} cm")
print(f"Clusterless decoding: Mean = {np.mean(error_clusterless):.1f} cm, "
      f"Median = {np.median(error_clusterless):.1f} cm")

# %%
# Compare sorted vs clusterless
fig, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained",
                         height_ratios=[2, 1])

ax = axes[0]
ax.plot(time_axis, true_position, "b-", linewidth=1, alpha=0.7, label="True")
ax.plot(time_axis, decoded_sorted, "r.", markersize=1, alpha=0.2, label="Sorted")
ax.plot(time_axis, decoded_clusterless, "g.", markersize=1, alpha=0.2, label="Clusterless")
ax.set(ylabel="Position (cm)",
       title="Sorted vs Clusterless Decoding (Well-Separated Neurons)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.plot(time_axis, error_sorted, "r-", linewidth=0.5, alpha=0.3, label="Sorted")
ax.plot(time_axis, error_clusterless, "g-", linewidth=0.5, alpha=0.3, label="Clusterless")
ax.axhline(np.median(error_sorted), color="r", linestyle="--", alpha=0.7,
           label=f"Sorted median: {np.median(error_sorted):.1f} cm")
ax.axhline(np.median(error_clusterless), color="g", linestyle="--", alpha=0.7,
           label=f"Clusterless median: {np.median(error_clusterless):.1f} cm")
ax.set(xlabel="Time (s)", ylabel="Error (cm)")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# With well-separated neurons (distinct amplitudes), sorted and clusterless
# decoding perform similarly. The sorted decoder has a slight advantage
# because it uses perfect neuron identity — the clusterless decoder must
# infer the amplitude-position relationship from the data.
#
# The real advantage of clusterless decoding appears when neurons are **hard
# to separate**.

# %% [markdown]
# ## Part 6: When Does Clusterless Help? — Overlapping Clusters
#
# Now let's simulate the scenario where spike sorting is difficult: neurons
# with **similar waveform features** but **different place fields**. This is
# common in real data — nearby neurons on the same electrode often have
# overlapping waveform shapes.

# %%
# Create a scenario with overlapping clusters
# Keep most neurons well-separated, but make 3 neurons have similar amplitudes
np.random.seed(123)

place_fields_overlap, neuron_amps_overlap, pf_centers_overlap = simulate_place_cells_1d(
    N_NEURONS, TRACK_LENGTH, position_bin_centers,
)

# Force 3 neurons to have nearly identical amplitudes (hard to sort)
OVERLAP_AMPLITUDE = 175.0  # All three neurons near this amplitude
neuron_amps_overlap[0] = OVERLAP_AMPLITUDE - 5
neuron_amps_overlap[1] = OVERLAP_AMPLITUDE
neuron_amps_overlap[2] = OVERLAP_AMPLITUDE + 5

print("Overlapping neurons:")
for i in range(3):
    print(f"  Neuron {i}: amplitude = {neuron_amps_overlap[i]:.0f} μV, "
          f"place field center = {pf_centers_overlap[i]:.0f} cm")

# %%
# Simulate spikes with overlapping clusters
true_position_overlap = simulate_trajectory_1d(N_TIME_STEPS, TRACK_LENGTH)
spike_counts_overlap, spike_amps_overlap, spike_ids_overlap = simulate_spikes_and_marks(
    true_position_overlap, place_fields_overlap, neuron_amps_overlap,
    position_bin_edges, TIME_BIN_SIZE, AMPLITUDE_NOISE_STD,
)

# %%
# Visualize the overlap problem
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

# Place fields for the 3 overlapping neurons
ax = axes[0]
colors = plt.cm.tab10(np.arange(N_NEURONS))
for i in range(N_NEURONS):
    lw = 2.5 if i < 3 else 0.8
    alpha = 1.0 if i < 3 else 0.3
    ax.plot(position_bin_centers, place_fields_overlap[i],
            color=colors[i], linewidth=lw, alpha=alpha,
            label=f"Neuron {i}" if i < 3 else None)
ax.set(xlabel="Position (cm)", ylabel="Firing rate (Hz)",
       title="Place Fields (bold = overlapping amplitudes)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# Amplitude distribution showing overlap
ax = axes[1]
for i in range(N_NEURONS):
    amp_range = np.linspace(neuron_amps_overlap[i] - 50, neuron_amps_overlap[i] + 50, 100)
    pdf = np.exp(-(amp_range - neuron_amps_overlap[i]) ** 2
                 / (2 * AMPLITUDE_NOISE_STD ** 2))
    lw = 2.5 if i < 3 else 0.8
    alpha = 1.0 if i < 3 else 0.3
    ax.plot(amp_range, pdf, color=colors[i], linewidth=lw, alpha=alpha,
            label=f"Neuron {i}" if i < 3 else None)
ax.set(xlabel="Spike amplitude (μV)", ylabel="Density",
       title="Amplitude Distributions (bold = overlapping)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# The three bold neurons have nearly identical amplitude distributions
# (heavily overlapping), but different place fields. A spike sorter would
# struggle to assign spikes to the correct neuron.

# %% [markdown]
# ### Simulating Imperfect Spike Sorting
#
# To model realistic spike sorting errors, we randomly reassign spikes from
# the overlapping neurons to each other with some probability.

# %%
def simulate_sorting_errors(
    spike_counts: NDArray[np.int32],
    spike_neuron_ids_per_bin: list[NDArray[np.int32]],
    overlap_neuron_ids: list[int],
    misassignment_rate: float,
) -> NDArray[np.int32]:
    """Simulate spike sorting errors for overlapping neurons.

    Spikes from overlapping neurons are randomly reassigned to other
    neurons in the overlap group.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_neurons, n_time_bins)
        True spike counts
    spike_neuron_ids_per_bin : list of np.ndarray
        Ground truth neuron IDs per spike per time bin
    overlap_neuron_ids : list of int
        Indices of neurons with overlapping features
    misassignment_rate : float
        Probability that a spike from an overlap neuron is assigned wrong

    Returns
    -------
    np.ndarray, shape (n_neurons, n_time_bins)
        Spike counts after sorting errors
    """
    n_neurons, n_time_bins = spike_counts.shape
    sorted_counts = np.zeros_like(spike_counts)
    overlap_set = set(overlap_neuron_ids)

    for t in range(n_time_bins):
        for idx, neuron_id in enumerate(spike_neuron_ids_per_bin[t]):
            if neuron_id in overlap_set and np.random.random() < misassignment_rate:
                # Randomly assign to another neuron in the overlap group
                wrong_neuron = np.random.choice(overlap_neuron_ids)
                sorted_counts[wrong_neuron, t] += 1
            else:
                sorted_counts[neuron_id, t] += 1

    return sorted_counts


# %%
# Compare three approaches
MISASSIGNMENT_RATE = 0.4  # 40% of overlapping spikes are misassigned

# 1. Perfect sorting (ground truth)
log_ll_perfect = compute_log_likelihood_1d(
    spike_counts_overlap, place_fields_overlap, TIME_BIN_SIZE,
)
posterior_perfect = log_likelihood_to_posterior_1d(log_ll_perfect)
decoded_perfect = decode_map_1d(posterior_perfect, position_bin_centers)
error_perfect = np.abs(decoded_perfect - true_position_overlap)

# 2. Imperfect sorting (with errors)
sorted_counts_bad = simulate_sorting_errors(
    spike_counts_overlap, spike_ids_overlap,
    overlap_neuron_ids=[0, 1, 2],
    misassignment_rate=MISASSIGNMENT_RATE,
)
log_ll_bad = compute_log_likelihood_1d(
    sorted_counts_bad, place_fields_overlap, TIME_BIN_SIZE,
)
posterior_bad = log_likelihood_to_posterior_1d(log_ll_bad)
decoded_bad = decode_map_1d(posterior_bad, position_bin_centers)
error_bad = np.abs(decoded_bad - true_position_overlap)

# 3. Clusterless decoding
# Collect all marks and positions
all_amps_ov = []
all_pos_ov = []
for t in range(N_TIME_STEPS):
    if len(spike_amps_overlap[t]) > 0:
        all_amps_ov.extend(spike_amps_overlap[t])
        all_pos_ov.extend([true_position_overlap[t]] * len(spike_amps_overlap[t]))

all_amps_ov = np.array(all_amps_ov)
all_pos_ov = np.array(all_pos_ov)

# Occupancy for overlap data
pos_bin_idx_ov = np.clip(
    np.searchsorted(position_bin_edges, true_position_overlap) - 1,
    0, N_POSITION_BINS - 1,
)
occupancy_ov = np.bincount(pos_bin_idx_ov, minlength=N_POSITION_BINS) * TIME_BIN_SIZE

joint_intensity_ov = estimate_joint_mark_intensity(
    all_amps_ov, all_pos_ov,
    position_bin_edges, amplitude_bin_edges,
    occupancy_ov, POSITION_BANDWIDTH, AMPLITUDE_BANDWIDTH,
)
ground_intensity_ov = compute_ground_intensity(joint_intensity_ov, amplitude_bin_size)

log_ll_cl = compute_clusterless_log_likelihood_1d(
    spike_amps_overlap, joint_intensity_ov, ground_intensity_ov,
    amplitude_bin_edges, TIME_BIN_SIZE,
)
posterior_cl = log_likelihood_to_posterior_1d(log_ll_cl)
decoded_cl = decode_map_1d(posterior_cl, position_bin_centers)
error_cl = np.abs(decoded_cl - true_position_overlap)

# %%
# Print comparison
print(f"Decoding with overlapping clusters (misassignment rate = {MISASSIGNMENT_RATE:.0%}):")
print(f"  Perfect sorting:   Mean = {np.mean(error_perfect):.1f} cm, "
      f"Median = {np.median(error_perfect):.1f} cm")
print(f"  Imperfect sorting: Mean = {np.mean(error_bad):.1f} cm, "
      f"Median = {np.median(error_bad):.1f} cm")
print(f"  Clusterless:       Mean = {np.mean(error_cl):.1f} cm, "
      f"Median = {np.median(error_cl):.1f} cm")

# %%
# Bar chart comparison
fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

methods = ["Perfect\nsorting", "Imperfect\nsorting", "Clusterless"]
median_errors = [np.median(error_perfect), np.median(error_bad), np.median(error_cl)]
colors = ["steelblue", "salmon", "mediumseagreen"]

bars = ax.bar(methods, median_errors, color=colors, edgecolor="black", linewidth=0.5)

# Add value labels on bars
for bar, val in zip(bars, median_errors):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f} cm", ha="center", va="bottom", fontsize=11)

ax.set(ylabel="Median decoding error (cm)",
       title="Clusterless is robust to overlapping clusters")
ax.spines[["top", "right"]].set_visible(False)

# %% [markdown]
# When neurons have overlapping waveform features:
#
# - **Perfect sorting** gives the best result (but is unrealistic)
# - **Imperfect sorting** degrades substantially — misassigned spikes corrupt
#   the encoding model
# - **Clusterless decoding** is robust because it never tries to assign spikes
#   to neurons. It uses the full amplitude-position relationship directly.
#
# In real data, we never have perfect spike sorting, so clusterless decoding
# provides a practical advantage.

# %% [markdown]
# ## Part 7: Discussion — Connecting the Pieces
#
# ### Three Levels of Decoding Sophistication
#
# Across notebooks 04a and 04b, we've built up a progression of decoders:
#
# 1. **Memoryless sorted decoder** (04a, Parts 1–8): Uses sorted spike counts
#    and Poisson likelihood, treats each time bin independently
#
# 2. **State space sorted decoder** (04a, Part 9): Adds a transition model
#    (random walk) to borrow strength across time — smoother, more accurate
#
# 3. **Clusterless decoder** (this notebook): Replaces sorted spike counts
#    with raw waveform features — robust to spike sorting errors
#
# These can be **combined**: a clusterless state space decoder uses waveform
# features as the data model and a transition model for temporal smoothing.
# This is the most powerful approach and is widely used in modern hippocampal
# decoding research.
#
# ### Applications
#
# - **Replay detection**: During rest and sleep, hippocampal neurons replay
#   spatial sequences at high speed. Clusterless decoding can detect these
#   replay events even when neurons overlap in feature space.
#
# - **Brain-computer interfaces (BCIs)**: Clusterless methods avoid the need
#   for time-consuming spike sorting, enabling real-time decoding.
#
# - **Theta sequences**: Within each theta cycle (~125 ms), place cells fire
#   in a compressed spatial sequence. Clusterless decoding at fine time scales
#   can resolve these sequences.
#
# ### Computational Considerations
#
# In this notebook we used a **1D mark** (amplitude) and **1D position**,
# making the joint intensity a simple 2D histogram. In real data, marks are
# high-dimensional (e.g., 3 PCs × 4 channels = 12 dimensions), and the joint
# intensity estimation becomes a high-dimensional KDE problem — computationally
# expensive and requiring careful bandwidth selection.
#
# Software packages like `replay_trajectory_classification`
# (Denovellis, Frank & Eden, 2021) implement efficient algorithms for this,
# including GPU-accelerated KDE and state space models with switching dynamics
# (for classifying local vs. non-local/replay trajectories).

# %% [markdown]
# ## Summary
#
# In this notebook, we learned how to:
#
# 1. **Understand the limitations of spike sorting** — information loss from
#    discarded spikes, subjective cluster boundaries, and misassignment
# 2. **Implement the marked point process framework** for clusterless decoding
# 3. **Estimate joint mark-position intensity** using smoothed 2D histograms
# 4. **Compare sorted and clusterless decoding** on simulated data
# 5. **See when clusterless decoding helps** — overlapping neurons degrade
#    sorted decoding but not clusterless
#
# ### Key Concepts
#
# | Concept | Description |
# |---------|-------------|
# | Spike sorting | Clustering spikes by waveform shape before decoding |
# | Clusterless decoding | Decoding directly from spike waveform features |
# | Mark | A waveform feature (e.g., amplitude) attached to each spike |
# | Marked point process | Point process where each event carries a mark |
# | Joint mark intensity | Rate of spikes with a particular mark at each position |
# | Ground intensity | Total spike rate at each position (integrated over marks) |
# | KDE | Kernel density estimation for non-parametric intensity estimation |
#
# ### The Clusterless Decoding Algorithm
#
# ```
# Training:
#     Estimate joint_intensity(mark, position) from training spikes
#     ground_intensity(position) = integral of joint_intensity over marks
#
# Decoding each time bin:
#     log_likelihood[x] = -ground_intensity[x] * dt
#     For each spike with mark m:
#         log_likelihood[x] += log(joint_intensity[m, x])
#     posterior = normalize(exp(log_likelihood - max(log_likelihood)))
#     decoded_position = argmax(posterior)
# ```
#
# ### Python Techniques Used
#
# - **`np.histogram2d`** for joint mark-position histograms
# - **`scipy.ndimage.gaussian_filter`** for KDE-style smoothing
# - **`np.searchsorted`** for efficient bin lookup
# - **Simulated data** for transparent pedagogical examples
#
# ### References
#
# - Denovellis, E. L., Frank, L. M., & Eden, U. T. (2021). *eLife*.
# - Kloosterman, F., Layton, S. P., Chen, Z., & Wilson, M. A. (2014).
#   *Journal of Neurophysiology*.
# - Deng, X., Liu, D. F., Karlsson, M. P., Frank, L. M., & Eden, U. T.
#   (2016). *Journal of Computational Neuroscience*.

# %% [markdown]
# ## Exercises
#
# Try these on your own to deepen your understanding:
#
# 1. **Bandwidth sensitivity**: Vary `POSITION_BANDWIDTH` and
#    `AMPLITUDE_BANDWIDTH` in the KDE (try 0.5, 1, 2, 4). How does decoding
#    accuracy change? What happens with too little or too much smoothing?
#
# 2. **Multiple mark dimensions**: Extend the simulation to use 2 waveform
#    features (e.g., amplitude and spike width). Estimate a 3D joint intensity
#    (amplitude × width × position). Does adding a second feature improve
#    clusterless decoding?
#
# 3. **State space + clusterless**: Combine the state space filter from
#    notebook 04a with the clusterless likelihood from this notebook. Apply it
#    to the overlapping-cluster scenario. How much does temporal smoothing help?
#
# 4. **Systematic overlap study**: Vary the amplitude separation between the
#    overlapping neurons (from 0 to 50 μV) and plot sorted vs clusterless
#    error as a function of separation. At what point does sorted decoding
#    match clusterless?
