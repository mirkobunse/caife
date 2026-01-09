import caife
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from unittest import TestCase
import unittest
from qunfold import TikhonovRegularization

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


if __name__ == '__main__':
    unittest.main()