"""Confidence intervals used by the evaluation suite (numpy only)."""
import numpy as np

Z_95 = 1.959963984540054


def wilson_interval(successes, trials, z=Z_95):
    """95% Wilson score interval for a proportion. Unlike the normal approximation it
    stays inside [0, 1] and gives a useful interval when every trial succeeds or fails."""
    if trials == 0:
        return None, None
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_interval(values, statistic=np.median, resamples=10_000, seed=0, level=0.95):
    """Percentile bootstrap interval for statistic(values), resampling values with
    replacement. Values may be inf (a run that never reached the event); the interval
    end is then inf too, meaning "not reached within the budget"."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(resamples, values.size))]
    estimates = statistic(samples, axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(estimates, [tail, 100 - tail], method="inverted_cdf")
    return float(low), float(high)


def paired_bootstrap_difference(a, b, resamples=10_000, seed=0, level=0.95):
    """Mean of b minus mean of a, with a percentile bootstrap interval that resamples
    the pairs (a[i], b[i]) together. Used to compare two settings run on the same mazes
    with the same exploration seeds."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired samples must be non-empty and the same length")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, a.size, size=(resamples, a.size))
    estimates = b[index].mean(axis=1) - a[index].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(estimates, [tail, 100 - tail], method="inverted_cdf")
    return float(b.mean() - a.mean()), float(low), float(high)
