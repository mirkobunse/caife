import caife
import numpy as np
from sklearn import DecisionTreeClassifier
from unittest import TestCase

class TestTreeBinning(TestCase):
    def test_with_linear_model(self):
        # TODO create synthetic data
        X, y = None, None
        target_bins = None

        # configure and fit a model with a TreeBinning
        max_n_bins_proxy = 5
        binning = caife.TreeBinning(
            target_bins=target_bins,
            tree=DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy),
        )
        model = caife.LinearModel(binning)
        model.fit(X, y)

        # check that max_leaf_nodes is respected
        f_random = np.random.dirichlet(np.ones(len(target_bins)-1))
        n_bins_proxy = model.binning.n_bins_proxy
        self.assertTrue(n_bins_proxy <= max_n_bins_proxy)
        self.assertTrue(model.A_.shape[1] == n_bins_proxy) # TODO or shape[0]?
        self.assertTrue(len(model(f_random)) == n_bins_proxy) # apply model

        # check that each column (TODO or row?) sums to one
        np.testing.assert_almost_equal(
            actual=np.sum(model.A_, axis=1), # shape (n_target_bins,)
            desired=1.,
        )
