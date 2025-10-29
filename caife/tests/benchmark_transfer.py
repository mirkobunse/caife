"""Benchmark possibilities for computing the transfer matrix `A`."""

import numpy as np
import timeit

def transfer_histogram2d(X, y, n_bins_proxy, n_bins_target, sample_weight=None):
    """Compute the transfer matrix `A` through `np.histogram2d`, like `funfolding` does."""
    A = np.histogram2d(
        x=X,
        y=y,
        bins=(np.arange(n_bins_proxy+1)-.5, np.arange(n_bins_target+1)-.5),
        weights=sample_weight,
    )[0]
    M_norm = np.diag(1 / np.sum(A, axis=0))
    return A @ M_norm # normalize

def transfer_classwise(X, y, n_bins_proxy, n_bins_target, sample_weight=None):
    """Compute the transfer matrix `A` through class-wise means, like `qunfold` does."""
    X = np.eye(n_bins_proxy)[X] # one-hot encoding
    A = np.zeros((n_bins_proxy, n_bins_target))
    for y_i in range(n_bins_target):
        if np.sum(y==y_i) > 0:
            w = None if sample_weight is None else sample_weight[y==y_i]
            A[:, y_i] = np.average(X[y==y_i], weights=w, axis=0)
    return A

def transfer_bincount(X, y, n_bins_proxy, n_bins_target, sample_weight=None):
    """Compute the transfer matrix `A` through `np.bincount`, like `caife` does."""
    A = np.bincount(
        n_bins_target * X + y, # combined X*y bins
        weights=sample_weight,
        minlength=n_bins_proxy * n_bins_target,
    ).reshape((n_bins_proxy, n_bins_target))
    return A / A.sum(axis=0, keepdims=True) # normalize

def main(seed=1491):
    """Benchmark possibilities for computing the transfer matrix `A`."""
    rng = np.random.default_rng(seed)

    # create random, binned, and univariate data
    n_samples = 1_000
    n_bins_target = 5
    n_bins_proxy = 8
    y = rng.integers(0, n_bins_target, size=n_samples)
    X = y * (n_bins_proxy / n_bins_target) + rng.normal(size=n_samples)
    X = np.clip(np.rint(X).astype(int), min=0, max=n_bins_proxy-1)
    w = rng.uniform(size=n_samples)

    # test equivalence of implementations
    transfer_args = (X, y, n_bins_proxy, n_bins_target)
    A_histogram2d = transfer_histogram2d(*transfer_args)
    A_classwise = transfer_classwise(*transfer_args)
    A_bincount = transfer_bincount(*transfer_args)
    np.testing.assert_allclose(
        A_classwise, # actual
        A_histogram2d, # desired
        err_msg="A_classwise != A_histogram2d",
    )
    np.testing.assert_allclose(
        A_bincount, # actual
        A_histogram2d, # desired
        err_msg="A_bincount != A_histogram2d",
    )
    print("Successfully tested all implementations for equivalence")

    # benchmark implementations
    print(
        "t_histogram2d =",
        timeit.timeit(lambda: transfer_histogram2d(*transfer_args), number=100),
    )
    print(
        "t_classwise =",
        timeit.timeit(lambda: transfer_classwise(*transfer_args), number=100),
    )
    print(
        "t_bincount =",
        timeit.timeit(lambda: transfer_bincount(*transfer_args), number=100),
    )

if __name__ == "__main__":
    main()
