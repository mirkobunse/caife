import caife
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from unittest import TestCase
import unittest

class TestTreeBinning(TestCase):
    def test_dev(self):
        X, y = make_classification(n_samples=1_000, n_features=10, n_informative=7, n_classes=4)
        max_n_bins_proxy = 5
        binning = caife.TreeBinning(
            tree=DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy),
        )

        A = binning.fit_transform(X, y, average=False)
        X_tree = binning.transform(X, average=False)
        self.assertFalse(False)
    
    def test_with_linear_model(self):
        X, y = make_classification(n_samples=1_000, n_features=10, n_informative=7, n_classes=4)

        # TODO: changed transform_tagets such that np.digitize includes right side of intervall instead of left.
        # so value 0 falls in bin 0, val 1 in 1 and so on.
        target_bins = [0, 1, 2, 3, 4]

        background = np.random.randn(*X.shape)

        # configure and fit a model with a TreeBinning
        max_n_bins_proxy = 5
        binning = caife.TreeBinning(
            tree=DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy),
        )
        model = caife.LinearModel(target_bins, binning)
        model.fit(X, y, background=background)

        # check that max_leaf_nodes is respected
        f_random = np.random.dirichlet(np.ones(len(target_bins)-1))
        n_bins_proxy = model.A_.shape[0]
        self.assertTrue(n_bins_proxy <= max_n_bins_proxy)
        self.assertTrue(model.A_.shape[0] == n_bins_proxy)
        self.assertTrue(len(model(f_random)) == n_bins_proxy) # apply model

        # check that each column sums to one
        np.testing.assert_almost_equal(
            actual=np.sum(model.A_, axis=0), # shape (n_target_bins,)
            desired=np.ones(model.n_bins_target),
        )

    def test_correct_bins(self):
        n_classes = 6
        X, y = make_classification(n_samples=1_000, n_features=10, n_informative=7, n_classes=n_classes)

        max_n_bins_proxy = 7
        tree = DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy).fit(X[:900], y[:900])
        binning = caife.TreeBinning(
            tree=tree,
            fit_tree=False
        )
        A = binning.fit_transform(X[:900], y[:900], average=False)

        binned_proxy = binning.transform(X[900:], average=False)
        tree_apply = tree.apply(X[900:])
        
        used_idx = []
        for val in range(binning.n_bins_):
            indices = np.unique(tree_apply[binned_proxy == val])
            self.assertEqual(indices.shape[0], 1) # all vals are in same leaf if they are in the same bin
            self.assertFalse(indices[0] in used_idx) # leaf index was not used in other bin
            used_idx.append(indices[0])


class TestGridSearchRepresentation(TestCase):
    def test():
        pass


if __name__ == '__main__':
    unittest.main()
