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
# # Week 2c: Evaluating and Diagnosing Poisson Regression Models
#
# In Notebook 02b, we fit several Poisson regression models to spike train data.
# But fitting a model is only the beginning — we need to know whether the model
# actually captures the structure in the data.
#
# This notebook covers the **model evaluation and refinement cycle** from Lecture 5:
#
# 1. **AIC** — Compare models on predictive performance
# 2. **Likelihood Ratio Test** — Test whether added covariates improve fit
# 3. **Confidence Intervals** — Assess individual parameter significance
# 4. **KS Test (Time-Rescaling Theorem)** — Test overall goodness-of-fit
# 5. **Residual Analysis** — Diagnose *where* the model fails
#
# The goal is iterative: fit → diagnose → refine → repeat, until the model
# adequately captures the data or we understand its limitations.
#
# ## Learning Objectives
#
# By the end of this notebook, you will be able to:
#
# 1. Use AIC and likelihood ratio tests to compare nested models
# 2. Interpret confidence intervals on model parameters
# 3. Apply the time-rescaling theorem to assess goodness-of-fit
# 4. Interpret KS plots and autocorrelation of rescaled ISIs
# 5. Use residual analysis to identify model deficiencies
# 6. Follow the iterative model refinement cycle

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
from scipy.signal import correlate
from scipy.stats import chi2, kstest, uniform

# %% [markdown]
# ## Load Data from DANDI

# %%
# Define the dataset location on DANDI
DANDISET_ID = "000059"
DANDISET_VERSION = "0.230907.2101"
ASSET_PATH = (
    "sub-MS22/"
    "sub-MS22_ses-Peter-MS22-180629-110319-concat_desc-processed_behavior+ecephys.nwb"
)

# %%
# Connect to DANDI and get the streaming URL
with DandiAPIClient() as client:
    dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
    asset = dandiset.get_asset_by_path(ASSET_PATH)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

print(f"Streaming from: {s3_url[:80]}...")

# %%
# Open the NWB file for streaming
remote_file = RemoteFile(s3_url)
h5_file = h5py.File(remote_file, "r")
io = NWBHDF5IO(file=h5_file, load_namespaces=True)
nwbfile = io.read()

print(f"Session: {nwbfile.identifier}")

# %% [markdown]
# ## Extract Behavioral and Neural Data

# %%
# Get behavior data
behavior_module = nwbfile.processing["behavior"]

# Extract position
position_interface = next(
    interface
    for name, interface in behavior_module.data_interfaces.items()
    if "position" in name.lower()
)
spatial_series = next(iter(position_interface.spatial_series.values()))
position_data = spatial_series.data[:]
position_timestamps = spatial_series.timestamps[:]
x_position = position_data[:, 0]
y_position = position_data[:, 1]

# Extract speed
speed_interface = next(
    interface
    for name, interface in behavior_module.data_interfaces.items()
    if "speed" in name.lower()
)
speed_data = speed_interface.data[:]
speed_timestamps = speed_interface.timestamps[:]

print(f"Position time range: {position_timestamps[0]:.1f} - {position_timestamps[-1]:.1f} s")

# %%
# Select a good unit to analyze
units_df = nwbfile.units.to_dataframe()
good_unit_mask = units_df["quality"].isin(["good", "good2"])
good_unit_indices = np.where(good_unit_mask)[0]

# Choose a unit (same as 02b for consistency)
unit_idx = good_unit_indices[3]
spike_times = nwbfile.units["spike_times"][unit_idx]

# Filter spikes to position tracking epoch
epoch_mask = (spike_times >= position_timestamps.min()) & (
    spike_times <= position_timestamps.max()
)
spike_times_epoch = spike_times[epoch_mask]

print(f"Unit {unit_idx}: {len(spike_times_epoch)} spikes during position tracking")

# %% [markdown]
# ## Prepare Binned Data
#
# We bin time into small (2 ms) intervals and compute covariates for each bin,
# exactly as in Notebook 02b.

# %%
# Define time bins
BIN_SIZE = 0.002  # 2 ms bins - small enough that at most one spike per bin

bin_edges = np.arange(
    position_timestamps.min(), position_timestamps.max() + BIN_SIZE, BIN_SIZE
)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2

# Count spikes in each bin
spike_counts, _ = np.histogram(spike_times_epoch, bins=bin_edges)

# Interpolate position and speed to bin centers
x_binned = np.interp(bin_centers, position_timestamps, x_position)
y_binned = np.interp(bin_centers, position_timestamps, y_position)
speed_binned = np.interp(bin_centers, speed_timestamps, speed_data)

# Compute movement direction from position gradient
x_gradient = np.gradient(x_binned)
direction = np.where(x_gradient > 0, "rightward", "leftward")

# Create DataFrame
df = pd.DataFrame(
    {
        "spike_count": spike_counts,
        "time": bin_centers,
        "x_position": x_binned,
        "y_position": y_binned,
        "speed": speed_binned,
        "direction": pd.Categorical(direction),
    }
)
df = df.dropna()

print(f"Number of time bins: {len(df)}")
print(f"Total spikes: {df['spike_count'].sum()}")

# %% [markdown]
# ## Fit a Sequence of Models
#
# We'll fit models of increasing complexity, then evaluate each one.
# This follows the **iterative model refinement cycle** from Lecture 5:
#
# 1. Start simple (constant rate)
# 2. Add covariates one at a time
# 3. Diagnose each model's failures
# 4. Use failures to guide what to add next

# %%
# Model 0: Constant rate (null model)
model_0 = smf.glm("spike_count ~ 1", data=df, family=sm.families.Poisson())
results_0 = model_0.fit()

# Model 1: Position only (2D spline)
model_1 = smf.glm(
    "spike_count ~ bs(x_position, df=5) * bs(y_position, df=5)",
    data=df,
    family=sm.families.Poisson(),
)
results_1 = model_1.fit()

# Model 2: Position + speed
model_2 = smf.glm(
    "spike_count ~ bs(x_position, df=5) * bs(y_position, df=5) + speed",
    data=df,
    family=sm.families.Poisson(),
)
results_2 = model_2.fit()

# Model 3: Position + speed + direction
model_3 = smf.glm(
    "spike_count ~ bs(x_position, df=5) * bs(y_position, df=5) + speed + direction",
    data=df,
    family=sm.families.Poisson(),
)
results_3 = model_3.fit()

models = [
    ("Constant", results_0),
    ("Position", results_1),
    ("Position + Speed", results_2),
    ("Position + Speed + Dir", results_3),
]

print("Models fitted successfully")
for name, res in models:
    print(f"  {name}: {len(res.params)} params, AIC = {res.aic:.1f}")

# %% [markdown]
# ## Method 1: Comparing AIC Values
#
# The **Akaike Information Criterion (AIC)** balances model fit against complexity:
#
# $$\text{AIC} = -2 \log L + 2p$$
#
# where $L$ is the maximized likelihood and $p$ is the number of parameters.
# **Lower AIC is better.** AIC penalizes overfitting — adding parameters only
# helps if they improve the likelihood enough to offset the penalty.
#
# AIC differences (ΔAIC) between models are more interpretable than raw values:
# - ΔAIC < 2: Models are essentially equivalent
# - ΔAIC 4–7: Moderate evidence for the better model
# - ΔAIC > 10: Strong evidence

# %%
# Compare AIC values
print("Model Comparison: AIC")
print("=" * 65)
print(f"{'Model':<30} {'Params':>8} {'AIC':>12} {'ΔAIC':>10}")
print("-" * 65)

best_aic = min(res.aic for _, res in models)
for name, res in models:
    delta_aic = res.aic - best_aic
    marker = " ← best" if delta_aic == 0 else ""
    print(f"{name:<30} {len(res.params):>8} {res.aic:>12.1f} {delta_aic:>10.1f}{marker}")
print("=" * 65)

# %% [markdown]
# ## Method 2: Likelihood Ratio Test for Nested Models
#
# When one model is a **nested** special case of another (you can get the simpler
# model by setting some parameters to zero), we can use the **likelihood ratio test**
# (LRT) to ask: do the extra parameters significantly improve fit?
#
# The test statistic is:
#
# $$\Lambda = 2 (\log L_{\text{full}} - \log L_{\text{reduced}})$$
#
# Under the null hypothesis (extra parameters are zero), $\Lambda$ follows a
# chi-squared distribution with degrees of freedom equal to the number of
# extra parameters.
#
# **Note:** LRT only works for nested models. To compare non-nested models
# (e.g., position-only vs. speed-only), use AIC instead.

# %%
# Test: Does speed improve upon position alone?
lr_stat = 2 * (results_2.llf - results_1.llf)
df_diff = len(results_2.params) - len(results_1.params)
p_value = chi2.sf(lr_stat, df=df_diff)

print("Likelihood Ratio Test: Position vs Position + Speed")
print("-" * 50)
print(f"  Log-likelihood (Position):         {results_1.llf:.1f}")
print(f"  Log-likelihood (Position + Speed): {results_2.llf:.1f}")
print(f"  Test statistic: χ² = {lr_stat:.2f}")
print(f"  Extra parameters: {df_diff}")
print(f"  p-value: {p_value:.2e}")
print(f"  → {'Reject' if p_value < 0.05 else 'Fail to reject'} H₀ (speed has no effect)")

# %%
# Test: Does direction improve upon position + speed?
lr_stat = 2 * (results_3.llf - results_2.llf)
df_diff = len(results_3.params) - len(results_2.params)
p_value = chi2.sf(lr_stat, df=df_diff)

print("\nLikelihood Ratio Test: Position + Speed vs Position + Speed + Direction")
print("-" * 50)
print(f"  Log-likelihood (Pos + Spd):       {results_2.llf:.1f}")
print(f"  Log-likelihood (Pos + Spd + Dir): {results_3.llf:.1f}")
print(f"  Test statistic: χ² = {lr_stat:.2f}")
print(f"  Extra parameters: {df_diff}")
print(f"  p-value: {p_value:.2e}")
print(f"  → {'Reject' if p_value < 0.05 else 'Fail to reject'} H₀ (direction has no effect)")

# %% [markdown]
# ## Method 3: Confidence Intervals for Individual Parameters
#
# We can test whether individual parameters are significantly different from zero
# using the **Wald test**, which uses the estimated standard error at the MLE:
#
# $$z = \frac{\hat{\beta}}{SE(\hat{\beta})}$$
#
# The 95% confidence interval is $\hat{\beta} \pm 1.96 \cdot SE(\hat{\beta})$.
# If the interval contains zero, the parameter is not significantly different
# from zero.
#
# The Wald test is fast (computed automatically by `statsmodels`) but can be
# unreliable when parameters are near boundaries or data are sparse. The LRT
# is generally more robust for comparing groups of parameters.

# %%
# Examine individual parameters in the full model
print("Individual Parameter Significance (Full Model)")
print("=" * 70)

# Extract the parameters we can interpret directly (not spline basis functions)
interpretable = ["speed", "direction[T.rightward]"]
summary = results_3.summary2().tables[1]

for param_name in interpretable:
    if param_name in summary.index:
        row = summary.loc[param_name]
        coef = row["Coef."]
        ci_low = row["[0.025"]
        ci_high = row["0.975]"]
        p_val = row["P>|z|"]

        # Multiplicative interpretation via exp
        print(f"\n{param_name}:")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  95% CI: ({ci_low:.4f}, {ci_high:.4f})")
        print(f"  p-value: {p_val:.2e}")

        if "speed" in param_name:
            print(f"  Interpretation: {100 * (np.exp(coef) - 1):.2f}% change in rate per 1 cm/s")
        elif "direction" in param_name:
            print(f"  Interpretation: {100 * (np.exp(coef) - 1):.2f}% change for rightward vs leftward")

        contains_zero = ci_low <= 0 <= ci_high
        print(f"  Contains zero: {contains_zero} → {'NOT significant' if contains_zero else 'Significant'}")

# %% [markdown]
# ## Method 4: KS Test for Model Goodness-of-Fit
#
# The previous methods compare models to each other. The **KS test** asks a
# different question: does the model fit the data *at all*?
#
# ### The Time-Rescaling Theorem
#
# If we have the correct conditional intensity function $\lambda(t)$, we can
# transform any point process into a **homogeneous Poisson process** of rate 1.
#
# The transformation works by "rescaling time": we stretch time where the model
# predicts high rates and compress time where it predicts low rates.
#
# Specifically, for each inter-spike interval $(s_{k-1}, s_k)$, we compute
# the **rescaled waiting time**:
#
# $$u_k = \int_{s_{k-1}}^{s_k} \hat{\lambda}(t) \, dt \approx \sum_{t \in (s_{k-1}, s_k)} \hat{\lambda}(t) \cdot \Delta t$$
#
# If the model is correct, the $u_k$ should be independent draws from an
# **Exponential(1)** distribution. We can further transform to uniform:
#
# $$z_k = 1 - e^{-u_k}$$
#
# which should be Uniform(0, 1) if the model is correct.
#
# We then use a **Kolmogorov-Smirnov (KS) test** to compare the empirical
# distribution of $z_k$ to Uniform(0, 1).

# %%
def time_rescaling_ks_test(spike_times, fitted_intensity, bin_edges):
    """Apply the time-rescaling theorem and KS test.

    Parameters
    ----------
    spike_times : np.ndarray
        Observed spike times in seconds.
    fitted_intensity : np.ndarray
        Fitted conditional intensity (rate in Hz) for each time bin.
    bin_edges : np.ndarray
        Time bin edges in seconds.

    Returns
    -------
    rescaled_uniforms : np.ndarray
        Rescaled waiting times transformed to Uniform(0, 1).
    ks_stat : float
        KS test statistic.
    ks_pvalue : float
        KS test p-value.
    """
    bin_size = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_size / 2

    # Expected spike count per bin: lambda(t) * dt
    expected_counts = fitted_intensity * bin_size

    # Find bin index for each spike
    spike_bin_idx = np.searchsorted(bin_edges, spike_times) - 1
    spike_bin_idx = np.clip(spike_bin_idx, 0, len(bin_centers) - 1)

    # Compute rescaled waiting times between consecutive spikes
    # u_k = integral of lambda(t) dt from spike k-1 to spike k
    rescaled_times = []
    for k in range(1, len(spike_bin_idx)):
        # Sum expected counts between consecutive spikes
        start_bin = spike_bin_idx[k - 1]
        end_bin = spike_bin_idx[k]
        if end_bin > start_bin:
            u_k = np.sum(expected_counts[start_bin:end_bin])
            rescaled_times.append(u_k)

    rescaled_times = np.array(rescaled_times)

    # Transform to Uniform(0,1): z = 1 - exp(-u)
    rescaled_uniforms = 1 - np.exp(-rescaled_times)

    # KS test against Uniform(0, 1)
    ks_stat, ks_pvalue = kstest(rescaled_uniforms, "uniform")

    return rescaled_uniforms, ks_stat, ks_pvalue

# %% [markdown]
# ### Apply the KS Test to Each Model
#
# We'll compare how well each model passes the time-rescaling test.
# A well-fitting model should have rescaled ISIs that look uniform.

# %%
# Compute fitted intensities (firing rate in Hz) for each model
fitted_intensities = {}
for name, res in models:
    # fittedvalues are expected counts per bin; divide by bin_size to get Hz
    fitted_intensities[name] = res.fittedvalues.values / BIN_SIZE

# Apply KS test to each model
print("KS Test Results (Time-Rescaling Theorem)")
print("=" * 60)
print(f"{'Model':<30} {'KS stat':>10} {'p-value':>12}")
print("-" * 60)

ks_results = {}
for name, res in models:
    z, ks_stat, ks_p = time_rescaling_ks_test(
        spike_times_epoch, fitted_intensities[name], bin_edges[:len(df) + 1]
    )
    ks_results[name] = z
    print(f"{name:<30} {ks_stat:>10.4f} {ks_p:>12.2e}")
print("=" * 60)
print("\nNote: Small p-value → model does NOT fit well")

# %% [markdown]
# ### Visualizing the KS Plot
#
# The KS plot compares the empirical CDF of the rescaled waiting times to the
# theoretical Uniform(0,1) CDF (the 45° line). Deviations from the diagonal
# indicate model misfit:
#
# - **Below the diagonal at small values**: Model underestimates short ISIs
#   (too few predicted short intervals → possibly missing refractory period)
# - **Above the diagonal at large values**: Model overestimates rate
#   (predicts more spikes than observed in some intervals)
# - **Systematic S-curve**: General model mis-specification
#
# The dashed lines show approximate 95% confidence bounds from the KS test.

# %%
fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)

for ax, (name, _) in zip(axes, models):
    z = ks_results[name]
    n = len(z)

    # Empirical CDF
    z_sorted = np.sort(z)
    ecdf = np.arange(1, n + 1) / n

    # Plot
    ax.plot(z_sorted, ecdf, "k-", linewidth=1)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Uniform")

    # 95% KS confidence bounds
    ks_bound = 1.36 / np.sqrt(n)
    ax.plot([0, 1], [ks_bound, 1 + ks_bound], "r:", linewidth=0.5)
    ax.plot([0, 1], [-ks_bound, 1 - ks_bound], "r:", linewidth=0.5)

    ax.set(xlabel="Rescaled ISI (uniform)", title=name, xlim=(0, 1), ylim=(0, 1))
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Empirical CDF")
fig.suptitle(f"Unit {unit_idx}: KS Plots (Time-Rescaling Theorem)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Autocorrelation of Rescaled ISIs
#
# If the model is correct, the rescaled ISIs should be independent (no
# temporal structure). Residual autocorrelation indicates the model is missing
# some temporal dependency — perhaps spike-history effects or slow rate changes.

# %%
def autocorr(x, lags=50):
    """Compute autocorrelation of a time series.

    Parameters
    ----------
    x : np.ndarray
        Input time series.
    lags : int
        Number of lags to compute.

    Returns
    -------
    np.ndarray
        Autocorrelation coefficients for lags 0 to `lags`.
    """
    x_centered = x - np.mean(x)
    xcorr = correlate(x_centered, x_centered, mode="full")
    xcorr = xcorr[xcorr.size // 2 :] / xcorr[xcorr.size // 2]
    return xcorr[: lags + 1]


fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3), sharey=True)

for ax, (name, _) in zip(axes, models):
    z = ks_results[name]
    ac = autocorr(z, lags=50)

    ax.bar(np.arange(1, len(ac)), ac[1:], color="steelblue", width=0.8)

    # 95% bounds under independence
    n = len(z)
    sig = 2 / np.sqrt(n)
    ax.axhline(sig, color="red", linestyle=":", linewidth=0.5)
    ax.axhline(-sig, color="red", linestyle=":", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set(xlabel="Lag", title=name)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Autocorrelation")
fig.suptitle(f"Unit {unit_idx}: Autocorrelation of Rescaled ISIs", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Method 5: Residual Analysis
#
# Residuals measure the difference between observed and predicted spike counts:
#
# $$r_t = n_t - \hat{\lambda}_t$$
#
# where $n_t$ is the observed spike count and $\hat{\lambda}_t = \hat{f}(t) \cdot \Delta t$
# is the model's predicted expected count.
#
# If the model fits well, residuals should:
# - Have mean zero
# - Be uncorrelated with any covariate
# - Show no systematic temporal structure
#
# Plotting residuals against covariates reveals **what the model is missing**.
# This directly tells us what to add next in the refinement cycle.

# %%
# Use the best model so far for residual analysis
best_name = min(models, key=lambda m: m[1].aic)[0]
best_results = dict(models)[best_name]

# Compute residuals
residuals = df["spike_count"].values - best_results.fittedvalues.values

print(f"Analyzing residuals from: {best_name}")
print(f"  Mean residual: {residuals.mean():.6f} (should be ≈ 0)")
print(f"  Std residual: {residuals.std():.6f}")

# %% [markdown]
# ### Residuals vs. Covariates
#
# If the model captures the effect of a covariate, residuals should be flat
# (zero mean) across all values of that covariate. A systematic trend indicates
# the model is missing structure related to that covariate.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Residuals vs x position
ax = axes[0]
x_bin_edges_res = np.linspace(df["x_position"].min(), df["x_position"].max(), 25)
x_bin_centers_res = (x_bin_edges_res[:-1] + x_bin_edges_res[1:]) / 2
x_bin_idx = np.digitize(df["x_position"], x_bin_edges_res) - 1

mean_resid_x = []
se_resid_x = []
for i in range(len(x_bin_centers_res)):
    mask = x_bin_idx == i
    if mask.sum() > 10:
        mean_resid_x.append(residuals[mask].mean())
        se_resid_x.append(residuals[mask].std() / np.sqrt(mask.sum()))
    else:
        mean_resid_x.append(np.nan)
        se_resid_x.append(np.nan)

valid = ~np.isnan(mean_resid_x)
ax.errorbar(
    np.array(x_bin_centers_res)[valid],
    np.array(mean_resid_x)[valid],
    yerr=np.array(se_resid_x)[valid],
    fmt="o",
    color="black",
    markersize=4,
    capsize=2,
)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set(xlabel="X position (cm)", ylabel="Mean residual")
ax.set_title("Residuals vs X Position")
ax.spines[["top", "right"]].set_visible(False)

# Residuals vs speed
ax = axes[1]
speed_bin_edges_res = np.linspace(0, df["speed"].quantile(0.99), 20)
speed_bin_centers_res = (speed_bin_edges_res[:-1] + speed_bin_edges_res[1:]) / 2
speed_bin_idx = np.digitize(df["speed"], speed_bin_edges_res) - 1

mean_resid_speed = []
se_resid_speed = []
for i in range(len(speed_bin_centers_res)):
    mask = speed_bin_idx == i
    if mask.sum() > 10:
        mean_resid_speed.append(residuals[mask].mean())
        se_resid_speed.append(residuals[mask].std() / np.sqrt(mask.sum()))
    else:
        mean_resid_speed.append(np.nan)
        se_resid_speed.append(np.nan)

valid = ~np.isnan(mean_resid_speed)
ax.errorbar(
    np.array(speed_bin_centers_res)[valid],
    np.array(mean_resid_speed)[valid],
    yerr=np.array(se_resid_speed)[valid],
    fmt="o",
    color="black",
    markersize=4,
    capsize=2,
)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set(xlabel="Speed (cm/s)", ylabel="Mean residual")
ax.set_title("Residuals vs Speed")
ax.spines[["top", "right"]].set_visible(False)

# Residuals vs direction
ax = axes[2]
left_resid = residuals[df["direction"] == "leftward"]
right_resid = residuals[df["direction"] == "rightward"]

means = [left_resid.mean(), right_resid.mean()]
ses = [
    left_resid.std() / np.sqrt(len(left_resid)),
    right_resid.std() / np.sqrt(len(right_resid)),
]

ax.bar([0, 1], means, yerr=ses, capsize=5, color=["#0072B2", "#D55E00"], alpha=0.7)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Leftward", "Rightward"])
ax.set_ylabel("Mean residual")
ax.set_title("Residuals vs Direction")
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Unit {unit_idx}: Residual Analysis ({best_name})", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Temporal Autocorrelation of Residuals
#
# If the model captures all temporal structure, residuals should be uncorrelated
# over time. Significant autocorrelation suggests the model is missing temporal
# dynamics — perhaps spike-history effects (refractory period, bursting) or
# slow covariates (e.g., theta rhythm modulation).

# %%
# Compute autocorrelation of residuals
n_lags = 100
residual_ac = autocorr(residuals, lags=n_lags)

fig, ax = plt.subplots(figsize=(10, 4))

lags = np.arange(1, n_lags + 1) * BIN_SIZE * 1000  # Convert to ms
ax.bar(lags, residual_ac[1:], width=BIN_SIZE * 1000, color="steelblue")

# 95% bounds under independence
sig = 2 / np.sqrt(len(residuals))
ax.axhline(sig, color="red", linestyle=":", linewidth=0.5, label="95% bounds")
ax.axhline(-sig, color="red", linestyle=":", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.5)

ax.set(
    xlabel="Lag (ms)",
    ylabel="Autocorrelation",
    title=f"Unit {unit_idx}: Residual Autocorrelation ({best_name})",
)
ax.legend()
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## The Model Refinement Cycle
#
# Putting it all together, the model refinement cycle proceeds as follows:
#
# 1. **Fit** a simple baseline model (e.g., constant rate)
# 2. **Interpret** the parameters: what tuning does the model predict?
# 3. **Diagnose** misfit:
#    - Residuals vs. covariates: systematic errors?
#    - KS plot: does the model capture spike timing?
#    - Autocorrelation: unaccounted temporal structure?
# 4. **Propose** an improvement based on the diagnosis
# 5. **Fit** the refined model
# 6. **Compare** to the simpler model (LRT if nested, ΔAIC otherwise)
# 7. **Re-diagnose** — check if the problem was fixed
# 8. **Iterate** until diagnostics look acceptable or you understand the limitations
#
# Let's walk through one iteration of this cycle.

# %% [markdown]
# ### Step 1: Start with Position-Only Model

# %%
# Residuals for position-only model
residuals_pos = df["spike_count"].values - results_1.fittedvalues.values

# Check residuals vs speed to see if speed matters
speed_bin_idx_check = np.digitize(df["speed"], speed_bin_edges_res) - 1

mean_resid_by_speed = []
se_resid_by_speed = []
for i in range(len(speed_bin_centers_res)):
    mask = speed_bin_idx_check == i
    if mask.sum() > 10:
        mean_resid_by_speed.append(residuals_pos[mask].mean())
        se_resid_by_speed.append(residuals_pos[mask].std() / np.sqrt(mask.sum()))
    else:
        mean_resid_by_speed.append(np.nan)
        se_resid_by_speed.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 4))

valid = ~np.isnan(mean_resid_by_speed)
ax.errorbar(
    np.array(speed_bin_centers_res)[valid],
    np.array(mean_resid_by_speed)[valid],
    yerr=np.array(se_resid_by_speed)[valid],
    fmt="o-",
    color="black",
    markersize=5,
    capsize=3,
)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set(
    xlabel="Speed (cm/s)",
    ylabel="Mean residual",
    title=f"Unit {unit_idx}: Position Model → Residuals vs Speed",
)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

print("If residuals show a trend with speed, this suggests adding speed to the model.")

# %% [markdown]
# ### Step 2: Add Speed and Re-diagnose

# %%
# Now check residuals of Position + Speed model vs direction
residuals_pos_speed = df["spike_count"].values - results_2.fittedvalues.values

left_resid_2 = residuals_pos_speed[df["direction"] == "leftward"]
right_resid_2 = residuals_pos_speed[df["direction"] == "rightward"]

fig, ax = plt.subplots(figsize=(6, 4))

means_2 = [left_resid_2.mean(), right_resid_2.mean()]
ses_2 = [
    left_resid_2.std() / np.sqrt(len(left_resid_2)),
    right_resid_2.std() / np.sqrt(len(right_resid_2)),
]

ax.bar([0, 1], means_2, yerr=ses_2, capsize=5, color=["#0072B2", "#D55E00"], alpha=0.7)
ax.axhline(0, color="red", linestyle="--", linewidth=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Leftward", "Rightward"])
ax.set_ylabel("Mean residual")
ax.set_title(f"Unit {unit_idx}: Position + Speed Model → Residuals vs Direction")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

print("If residuals differ by direction, this suggests adding direction to the model.")

# %% [markdown]
# ### Step 3: Compare All Models

# %%
# Final model comparison summary
print("Model Refinement Summary")
print("=" * 70)
print(f"{'Model':<30} {'Params':>8} {'AIC':>12} {'Log-Lik':>12}")
print("-" * 70)
for name, res in models:
    print(f"{name:<30} {len(res.params):>8} {res.aic:>12.1f} {res.llf:>12.1f}")
print("=" * 70)

# Nested model comparisons
print("\nNested Model Comparisons (LRT):")
for i in range(1, len(models)):
    name_reduced, res_reduced = models[i - 1]
    name_full, res_full = models[i]
    lr = 2 * (res_full.llf - res_reduced.llf)
    df_diff = len(res_full.params) - len(res_reduced.params)
    p = chi2.sf(lr, df=df_diff)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name_reduced} → {name_full}: χ²={lr:.1f}, Δdf={df_diff}, p={p:.2e} {sig}")

# %% [markdown]
# ## Exercises
#
# Try these on your own to deepen your understanding:
#
# 1. **Different neuron**: Repeat this analysis for a different unit. Do all
#    neurons benefit from the same covariates?
#
# 2. **History dependence**: The KS test and residual autocorrelation often
#    reveal temporal structure. Try adding spike-history terms to the model.
#    Create a column for "spike in previous bin" and add it as a covariate.
#    Does this improve the KS plot?
#
# 3. **Spline complexity**: Try different spline degrees of freedom (df=3, 5,
#    8, 12) for the position model. Use AIC to find the optimal complexity.
#    Does the KS test agree with AIC about which is best?
#
# 4. **Speed nonlinearity**: The current model assumes a linear relationship
#    between log-rate and speed. Try `bs(speed, df=4)` for a nonlinear speed
#    effect. Does this improve fit?
#
# 5. **Cross-validated prediction**: Split data into first and second halves.
#    Fit on the first half, compute log-likelihood on the second half. Does
#    the model that wins on AIC also win on held-out data?

# %% [markdown]
# ## Summary
#
# In this notebook, we learned how to evaluate Poisson regression models using
# five complementary methods:
#
# | Method | What it tests | Compares |
# |--------|---------------|----------|
# | AIC | Predictive performance | Any two models |
# | Likelihood Ratio Test | Statistical improvement | Nested models only |
# | Confidence Intervals | Individual parameters | Parameter vs. zero |
# | KS Test (Time-Rescaling) | Overall goodness-of-fit | Model vs. data |
# | Residual Analysis | Where the model fails | Model vs. covariates |
#
# ### The Model Refinement Cycle
#
# ```
# Fit simple model
#     ↓
# Diagnose (residuals, KS, autocorrelation)
#     ↓
# Identify what's missing
#     ↓
# Add covariate / change functional form
#     ↓
# Re-fit and compare (LRT, ΔAIC)
#     ↓
# Re-diagnose
#     ↓
# Conclude or iterate
# ```
#
# This cycle embodies the scientific method applied to statistical modeling:
# propose a hypothesis (model), test it against data, identify failures, and
# refine. Poisson regression and point process theory provide a structured,
# principled framework for this process.
#
# ### Next Steps
#
# In Week 3, we'll analyze the **local field potential (LFP)** and its spectral
# properties. We'll examine theta oscillations, which coordinate the timing of
# place cell spikes and are critical for navigation.

# %% [markdown]
# ## Cleanup

# %%
io.close()
