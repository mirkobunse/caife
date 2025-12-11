import caife
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from unittest import TestCase
import unittest



class TestLinearCountModel(TestCase):
    def test(self):
        rng = np.random.default_rng(1491)
        y = rng.uniform(size=1_000) * 4
        X = (y + rng.normal(size=1_000)).reshape((-1, 1))

        # assert that target_bins are required to range from -inf to inf
        erroring_model = caife.LinearCountModel(
            target_bins=np.arange(5), # not ranging from -inf to inf
            representation=caife.UnivariateBinning(
                proxy_bins=np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            )
        )
        self.assertRaises( # erroring_model.fit(X, y) raises a ValueError
            ValueError,
            erroring_model.fit,
            X, # *args
            y,
        )

        # assert correct target representation
        target_bins = np.arange(5, dtype=float)
        target_bins[0] = -np.inf
        target_bins[-1] = np.inf
        model = caife.LinearCountModel(
            target_bins=target_bins,
            representation=caife.UnivariateBinning(
                proxy_bins=np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            )
        )
        # y ∈ [ 0, 1, 2, 3 ], X ∈ [ 0, 1, 2, 3, 4, 5 ]
        self.assertEqual(model.represent_target(y).min(), 0)
        self.assertEqual(model.represent_target(y).max(), 3)
        for value in [ np.nan, np.inf, -np.inf ]:
            self.assertRaises( # model.represent_target(inf | nan) raises a ValueError
                ValueError,
                model.represent_target,
                [ value ], # *args
            )
        model.fit(X, y)
        self.assertEqual(model.A_.shape, (6, 4))


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

        target_bins = np.arange(5, dtype=float) # n_classes+1 bin boundaries required
        target_bins[0] = -np.inf # bins have to range from -inf to inf
        target_bins[-1] = np.inf

        background = np.random.randn(*X.shape)

        # configure and fit a model with a TreeBinning
        max_n_bins_proxy = 5
        binning = caife.TreeBinning(
            tree=DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy),
        )
        model = caife.LinearCountModel(target_bins, binning)
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

    def test_correct_binning(self):
        n_classes = 6
        X, y = make_classification(n_samples=1_000, n_features=10, n_informative=7, n_classes=n_classes)

        max_n_bins_proxy = 7
        tree = DecisionTreeClassifier(max_leaf_nodes=max_n_bins_proxy).fit(X[:900], y[:900])
        binning = caife.TreeBinning(tree=tree, fit_tree=False)
        A = binning.fit_transform(X[:900], y[:900], average=True)

        np.testing.assert_almost_equal(
            actual=np.sum(A, axis=0), # shape (n_classes,)
            desired=np.ones(n_classes),
        )

        binned_proxy_one_hot = binning.transform(X[900:], average=False)
        tree_apply = tree.apply(X[900:])
        binned_proxy_idx = np.argmax(binned_proxy_one_hot, axis=1)

        # test that f(x) only consists of one hot vectors 
        # all rows sum to 1
        np.testing.assert_equal(
            actual=np.sum(binned_proxy_one_hot, axis=1), 
            desired=np.ones(binned_proxy_one_hot.shape[0]),
        )
        # all rows contain a 1
        self.assertTrue(np.all(np.any(binned_proxy_one_hot == 1, axis=1)))

        # test that all values get binned correctly
        used_idx = []
        for val in range(binning.n_output_features):
            indices = np.unique(tree_apply[binned_proxy_idx == val])
            self.assertEqual(indices.shape[0], 1) # all vals are in same leaf if they are in the same bin
            self.assertFalse(indices[0] in used_idx) # leaf index was not used in other bin
            used_idx.append(indices[0])

        binned_proxy = binning.transform(X[900:], average=True)

        # relative counts sum to one
        self.assertAlmostEqual(np.sum(binned_proxy), 1.)

        # total_values == bincount(tree_apply)
        counts = np.bincount(tree_apply)
        np.testing.assert_equal(
            (binned_proxy * counts.sum()).astype(int),
            counts[counts.nonzero()]
        )


class TestGridSearchRepresentation(TestCase):
    def test(self):
        seed = 1491
        n_bins_target = 3

        # create synthetic data
        X, y = make_classification(
            n_samples=1_000,
            n_informative=n_bins_target,
            n_classes=n_bins_target,
            random_state=seed,
        )

        # configure the GridSearchRepresentation
        base_binning = caife.TreeBinning(DecisionTreeClassifier(random_state=seed))
        param_grid = {
            "tree__max_leaf_nodes": [ 5, 10, 15 ],
            "tree__criterion": [ "gini", "entropy" ],
        }
        binning = caife.GridSearchRepresentation(
            base_binning,
            param_grid,
            is_verbose=True,
        )

        # fit: take out the grid search
        A_best = binning.fit_transform(X, y)
        X_best = binning.transform(X)
        params_best = binning.results_[0]["params"]

        # check the outcome
        losses = np.array([ x["losses"][binning.criterion] for x in binning.results_ ])
        self.assertTrue(np.all(losses[1:] - losses[:-1] >= 0)) # check if sorted
        self.assertTrue(np.all(A_best == binning.results_[0]["A"]))
        self.assertTrue(np.all(X_best == binning.results_[0]["representation"].transform(X)))
        self.assertTrue(A_best.shape[0] <= np.max(param_grid["tree__max_leaf_nodes"]))
        self.assertTrue(A_best.shape[1] == n_bins_target)

        # re-run with the best parameters
        single_cell_grid = { k: [v] for k, v in params_best.items() }
        binning = caife.GridSearchRepresentation(base_binning, single_cell_grid)
        A_rep = binning.fit_transform(X, y)
        self.assertTrue(np.all(A_rep == A_best))

        # re-run with the best parameters and an empty grid
        empty_grid = {}
        base_binning.set_params(**params_best)
        binning = caife.GridSearchRepresentation(base_binning, empty_grid)
        A_rep = binning.fit_transform(X, y)
        self.assertTrue(np.all(A_rep == A_best))


if __name__ == '__main__':
    unittest.main()
