import unittest
from unittest import TestCase

import numpy as np
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

import caife
from caife._qunfold import TikhonovRegularization


class TestTikhonovRegularization(TestCase):
    def test_tikhonov(self):
        n_classes = 12
        n_samples = 5

        X, y = make_classification(n_samples=1_000, n_features=20, n_informative=13, n_classes=n_classes)

        target_bins = np.arange(n_classes + 1, dtype=float) # n_classes+1 bin boundaries required
        target_bins[0] = -np.inf # bins have to range from -inf to inf
        target_bins[-1] = np.inf

        background = np.random.randn(*X.shape)

        # configure and fit a model with a TreeBinning
        max_n_bins_proxy = 8
        binning = caife.TreeBinning(
            tree=DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy),
        )
        model = caife.LinearCountModel(target_bins, binning)
        model.fit(X, y, background=background)

        A = model.A_

        tikhonov_qunfold = TikhonovRegularization().instantiate(None, A, None)

        # single sample (1D)
        f = np.random.randn(n_classes)

        # Test that implementation is equivalent to qunfold
        self.assertAlmostEqual(
            caife.tikhonov_regularization(f),
            tikhonov_qunfold(f),
            delta=6
        )

        # multi sample (2D)
        f = np.random.randn(n_samples, n_classes)

        single_sample_reg = np.array([caife.tikhonov_regularization(f[i, :]) for i in range(n_samples)])
        qunfold_sample_reg = np.array([tikhonov_qunfold(f[i, :]) for i in range(n_samples)])
        multi_sample_reg = caife.tikhonov_regularization(f)

        # Test that multi sample array equals single sample reg_terms
        np.testing.assert_almost_equal(
            multi_sample_reg,
            single_sample_reg,
        )

        # Test that multi sample array equals qunfold implementation
        np.testing.assert_almost_equal(
            multi_sample_reg,
            qunfold_sample_reg,
            decimal=4 # qunfold rounds values quite often, sometimes to the 4th decimal place
        )

    def test_linear_spectra(self):
        rng = np.random.default_rng(0)
        for n_classes in [3, 4, 7, 12, 50]:
            for _ in range(5):
                # any f that is linear in its indices has a zero regularization value, both
                # with the default arguments and with the corresponding explicit arguments
                a, b = rng.uniform(-100, 100), rng.uniform(0, 1000)
                f = a * np.arange(n_classes) + b
                for value in [
                        caife.tikhonov_regularization(f),
                        caife.tikhonov_regularization(f, scaling_factors=None, order=1),
                    ]:
                    self.assertAlmostEqual(float(np.squeeze(value)), 0, delta=1e-8)

            # the same holds for a matrix of linear candidate solutions, one per row
            a = rng.uniform(-100, 100, size=(4, 1))
            b = rng.uniform(0, 1000, size=(4, 1))
            f = a * np.arange(n_classes) + b # shape (n_solutions, n_classes)
            values = caife.tikhonov_regularization(f)
            self.assertEqual(np.shape(values), (4,))
            np.testing.assert_allclose(values, 0, atol=1e-8)

            # in contrast, a curved f is penalized
            value = caife.tikhonov_regularization(np.arange(n_classes)**2.)
            self.assertGreater(float(np.squeeze(value)), 0)

    def test_quadratic_spectra(self):
        rng = np.random.default_rng(0)
        for n_classes in [4, 7, 12, 50]: # order 2 leaves n_classes-3 bins to regularize
            for _ in range(5):
                # any f that is quadratic in its indices has a zero regularization value of order 2
                a, b, c = rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(0, 1000)
                f = a * np.arange(n_classes)**2 + b * np.arange(n_classes) + c
                value = caife.tikhonov_regularization(f, scaling_factors=None, order=2)
                self.assertAlmostEqual(float(np.squeeze(value)), 0, delta=1e-8)

            # the same holds for a matrix of quadratic candidate solutions, one per row
            a = rng.uniform(-100, 100, size=(4, 1))
            b = rng.uniform(-100, 100, size=(4, 1))
            c = rng.uniform(0, 1000, size=(4, 1))
            f = a * np.arange(n_classes)**2 + b * np.arange(n_classes) + c # shape (n_solutions, n_classes)
            values = caife.tikhonov_regularization(f, order=2)
            self.assertEqual(np.shape(values), (4,))
            np.testing.assert_allclose(values, 0, atol=1e-8)

            # in contrast, a cubic f is penalized
            value = caife.tikhonov_regularization(np.arange(n_classes)**3., order=2)
            self.assertGreater(float(np.squeeze(value)), 0)


if __name__ == '__main__':
    unittest.main()
