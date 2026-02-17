# Future Slide Improvements

## General

- More examples
- Slides more uniformly formatted
- Equations more uniformly formatted
- Generate PDFs with beamer

## Kernel Density Estimation

- Edge effects and how to deal with them
- Binning then KDE vs. KDE on raw data tradeoffs

## Point Process Time rescaling

- Misfit models of point processes: ISIs too long/short, missing covaraites

## Model Comparison

- Better explanation of AIC vs. Likelihood ratio test uses

## Spectral Analysis

- Fix expalantion of Fourier frequency bins vs. frequency resolution

## Decoding

### Lecture 7

- What is decoding?
- How is decoding used in spatial neuroscience?
  - Theta
  - Replay
- Review of point process encoding models
- Review of Poisson likelihood
- Review of Bayes Theorem
- Bayesian decoding of position from place cell activity
- Examples
- Practical issues
  - numerical stability of likelihoods (product and small numbers)
  - choice of time bin size
  - choice of position bin size
  - choice of smoothing in the encoding model
  - assumptions of this model (correct encoding, conditional independence of neurons, etc.)
  - we use the expected rate from the encoding model, but this does not take into account uncertainty in the encoding model parameters (e.g. place field locations and shapes)
- Examples with different priors
- Problems with Bayesian decoding of position from place cell activity
  - No spike likelihood without full place field coverage (biases in cell sampling)
  - continuity
  - spike sorting errors
  - Non-stationary place fields
  - Biases in behavior (e.g. more time spent in some locations than others)
    - occupancy estimation issues
  - Assumes you know the encoding model (e.g. place fields) and that it is correct
  - Computation with large numbers of position bins and small time bins
- How to determine trajectory from decoded position estimates
  - MAP estimate of trajectory (theta vs. replay)
  - Radon transform
  - Linear regression
  - Weighted Correlation
- Shuffling for significance testing
  - Shuffle spike times
  - Shuffle cell ID
  - Shuffle place field locations
  - Pseudo-event shuffles
  - Permutation tests
  - Problems with Shuffles
- Problems with fitting a line to decoding
  - impact of bin size on decoding error
  - linear assumption
  - Arbitrary criteria for selecting which time bins to decode
    - e.g. only decode time bins with at least 5 spikes, or only decode time bins with a significant replay event
    - SWR detection
    - MUA detection
    - MAP based criteria
- Decoding other variables from neural activity
  - Discrete variables (e.g. left vs. right turn)
  - Likelihood ratio
  - Circular variables (e.g. head direction)
- Alternative approaches to decoding position from place cell activity
  - PCA/ICA
- Summary

### Lecture 8

- Recap of Lecture 7
- State space models
  - Adding a transition matrix as a prior
  - causal vs. acausal decoding
- Issues with continuous transition models
  - Can't capture jumps in position (e.g. replay)
  - Can't capture different dynamics during theta vs. replay
- Switching state space models
  - Uniform
  - Multiple tracks
  - direction of running
  - Local and non-local states
  - Modeling no-spike states
- Uncertainty
  - Highest posterior density intervals
- Issues
  - Computation with large numbers of position bins and small time bins
- Estimating parameters of state space models
  - EM algorithm
  - Variational inference
- Model checking for state space models
- Model comparison for state space models
- Clusterless decoding
  - Using spike features instead of spike sorting
  - Advantages and disadvantages of clusterless decoding
  - Examples of clusterless decoding
  - State space models for clusterless decoding
  - Model checking for state space models for clusterless decoding
