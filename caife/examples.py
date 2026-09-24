"""Module providing synthetic toy data for use in examples and tests."""

import numpy as np
from scipy.stats import norm

# the fixed ground-truth mixture from which `y` is drawn; shared with
# `create_effective_areas`, which needs to know this same shape
_MIXTURE_WEIGHTS = np.array([0.6, 0.4])
_MIXTURE_MEANS = np.array([-2.0, 3.0])
_MIXTURE_STDS = np.array([0.8, 1.2])


def create_data(n_samples=200_000, s_min=-1.0, s_max=1.0, resample=False, rng=None):
    """Create synthetic proxy, target, weight, and systematics data.

    `y` is drawn from a fixed two-component Gaussian mixture. `X` is a univariate proxy of `y`, obtained by smearing each `y` through a Gaussian kernel whose mean and standard deviation depend on a systematic parameter `S`. `S` is drawn independently of `y`, uniformly over a fixed range. `w` are sample weights that are mostly random but slightly increase with `y`.

    Args:
        n_samples (optional): The number of samples to draw. Defaults to `200_000`.
        s_min (optional): The minimum value of `S`. Defaults to `-1`.
        s_max (optional): The maximum value of `S`. Defaults to `1`.
        resample (optional): Whether to resample the data according to its weights `w`. Only set this to `True` when you use this function to generate a separate test set; if you first generate the full data and then apply `split_data`, keep the default value. Defaults to `False`.
        rng (optional): A NumPy random number generator, an integer seed, or `None` for an unseeded generator. Defaults to `None`.

    Returns:
        A tuple `(X, y, w, S)` of NumPy arrays, where `X` and `S` have shape `(n_samples, 1)` and `y` and `w` have shape `(n_samples,)`.
    """
    rng = np.random.default_rng(rng)

    # y: a fixed two-component Gaussian mixture
    component = rng.choice(len(_MIXTURE_WEIGHTS), size=n_samples, p=_MIXTURE_WEIGHTS)
    y = rng.normal(_MIXTURE_MEANS[component], _MIXTURE_STDS[component])

    # S: a systematic parameter, independent of y, uniform over a fixed range
    S = rng.uniform(s_min, s_max, size=(n_samples, 1))

    # how S affects the smearing kernel that turns y into X
    smearing_std = 1.0
    systematic_shift = 0.5 # shifts the smearing mean by up to +-0.5
    systematic_scale = 0.3 # scales the smearing std within [0.7, 1.3]

    # X: y smeared through a Gaussian kernel whose mean and std depend on S
    s = S[:, 0]
    shift = systematic_shift * s
    scale = smearing_std * (1 + systematic_scale * s) # > 0 since |s| <= 1
    X = (y + shift + rng.normal(size=n_samples) * scale).reshape(-1, 1)

    # w: random weights with a slight, monotonic dependence on y
    w_y_dependence = 0.05
    y_ref = _MIXTURE_WEIGHTS @ _MIXTURE_MEANS # the mixture's true mean
    w = rng.uniform(0.5, 1.5, size=n_samples) * (1 + w_y_dependence * (y - y_ref))
    w = np.clip(w, 1e-3, None) # keep weights strictly positive, even for extreme outliers

    # optionally resample according to the weights
    if resample:
        i_bootstrap = rng.choice(n_samples, size=n_samples, replace=True, p=w / w.sum())
        X, y, w, S = X[i_bootstrap], y[i_bootstrap], w[i_bootstrap], S[i_bootstrap]

    return X, y, w, S


def split_data(X, y, w, S, rng=None):
    """Split data into a training and a test set.

    `X`, `y`, `w`, and `S` are first split into two equally sized halves. The test half is then resampled according to its own weights `w`.

    Args:
        X: Proxy samples, shape `(n_samples, n_proxy_features)`.
        y: Target samples, shape `(n_samples,)`.
        w: Sample weights, shape `(n_samples,)`.
        S: Systematic parameter values, shape `(n_samples, n_systematic_parameters)`.
        rng (optional): A NumPy random number generator, an integer seed, or `None` for an unseeded generator. Defaults to `None`.

    Returns:
        A pair of tuples `(X_trn, y_trn, w_trn, S_trn)` and `(X_tst, y_tst)`.
    """
    rng = np.random.default_rng(rng)
    n_half = len(y) // 2
    i_trn, i_tst = np.split(rng.permutation(2 * n_half), 2) # equally sized halves

    X_trn, y_trn, w_trn, S_trn = X[i_trn], y[i_trn], w[i_trn], S[i_trn]
    X_tst, y_tst, w_tst = X[i_tst], y[i_tst], w[i_tst]

    # resample the test half according to its own weights, keeping its size fixed
    i_bootstrap = rng.choice(n_half, size=n_half, replace=True, p=w_tst / w_tst.sum())
    X_tst, y_tst = X_tst[i_bootstrap], y_tst[i_bootstrap]

    return (X_trn, y_trn, w_trn, S_trn), (X_tst, y_tst)


def create_effective_areas(target_bins, slope=-0.5):
    """Create effective areas that turn the mixture's shape log-linear.

    `create_data`'s target `y` follows a two-component Gaussian mixture, whose double-peak shape is not log-linear in its own right. This function computes, for each inner bin of `target_bins`, an effective area `a_eff` such that `log(f_true / a_eff)` is an exact linear function of the bin index, where `f_true` is the mixture's true probability mass in that bin. Dividing an empirical spectrum by `a_eff` before regularizing its logarithm, as `caife`'s example notebooks do, then matches the log-linear prior that Tikhonov regularization implicitly imposes; for a spectrum drawn from `create_data`, `log(f / a_eff)` is only approximately linear, since sampling introduces some noise around the true mixture shape.

    Args:
        target_bins: The target bin boundaries, ranging from `-inf` to `inf`, as used elsewhere in `caife`. Only the inner boundaries (i.e., excluding the two overflow bins) are used here.
        slope (optional): The slope of `log(f_true / a_eff)` over the bin index. Defaults to `-0.5`.

    Returns:
        An array of effective areas, one for each inner bin of `target_bins` (i.e., of length `len(target_bins) - 3`).
    """
    edges = np.asarray(target_bins)[1:-1] # inner boundaries, excluding -inf and inf
    cdf = sum(
        weight_k * norm.cdf(edges, mean_k, std_k)
        for weight_k, mean_k, std_k in zip(_MIXTURE_WEIGHTS, _MIXTURE_MEANS, _MIXTURE_STDS)
    )
    f_true = np.diff(cdf) # the mixture's true probability mass per inner bin

    i = np.arange(len(f_true))
    return f_true / np.exp(slope * i)
