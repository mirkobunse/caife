import unittest
from unittest import TestCase

import numpy as np

import caife


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


class TestMultiTargetModel(TestCase):
    def test(self):
        rng = np.random.default_rng(1491)
        Y = rng.uniform(size=(1_000, 2)) * 4
        X = np.stack((
            (Y[:,0] + rng.normal(size=1_000)),
            (Y[:,1] + rng.normal(size=1_000)),
        )).T

        # assert that target_bins are required to range from -inf to inf
        representation = caife.GridBinning([
            np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            np.concatenate(([-np.inf], np.arange(3), [np.inf])),
        ])
        erroring_model = caife.MultiTargetModel(
            target_bins=[np.arange(5), np.arange(3)], # not ranging from -inf to inf
            representation=representation,
        )
        self.assertRaises( # erroring_model.fit(X, y) raises a ValueError
            ValueError,
            erroring_model.fit,
            X, # *args
            Y,
        )

        # assert correct target representation
        target_bins = [
            np.arange(5, dtype=float), # 4 classes
            np.arange(3, dtype=float), # 2 classes
        ]
        for b_i in target_bins:
            b_i[0] = -np.inf
            b_i[-1] = np.inf
        model = caife.MultiTargetModel(
            target_bins=target_bins,
            representation=representation,
        )
        self.assertEqual(model.represent_targets(Y).min(), 0)
        self.assertEqual(model.represent_targets(Y).max(), 7)
        for value in [ np.nan, np.inf, -np.inf ]:
            self.assertRaises( # model.represent_target(inf | nan) raises a ValueError
                ValueError,
                model.represent_targets,
                [ value ], # *args
            )
        model.fit(X, Y)

        # instantiate standard models as a reference
        model_a = caife.LinearCountModel(
            target_bins=target_bins[0],
            representation=caife.UnivariateBinning(representation.proxy_bins[0]),
        )
        model_b = caife.LinearCountModel(
            target_bins=target_bins[1],
            representation=caife.UnivariateBinning(representation.proxy_bins[1]),
        )

        # check for equivalence in target and proxy views
        np.testing.assert_equal(
            model.represent_targets(Y, separate=True)[:,0],
            model_a.represent_target(Y[:,0]),
        )
        np.testing.assert_equal(
            model.represent_targets(Y, separate=True)[:,1],
            model_b.represent_target(Y[:,1]),
        )
        np.testing.assert_equal(
            caife.MultiTargetModel(
                target_bins=target_bins,
                representation=caife.GridBinning([representation.proxy_bins[0]]),
            ).fit(X[:,[0]], Y).proxy_view(X[:,[0]]),
            model_a.proxy_view(X[:,[0]]),
        )
        np.testing.assert_equal(
            caife.MultiTargetModel(
                target_bins=target_bins,
                representation=caife.GridBinning([representation.proxy_bins[1]]),
            ).fit(X[:,[1]], Y).proxy_view(X[:,[1]]),
            model_b.proxy_view(X[:,[1]]),
        )
        y_a = model_a.represent_target(Y[:,0])
        y_b = model_b.represent_target(Y[:,1])
        reference = np.zeros(model.n_bins_per_target)
        for i_a in range(model_a.n_bins_target):
            for i_b in range(model_b.n_bins_target):
                reference[i_a, i_b] = np.sum((y_a == i_a) & (y_b == i_b))
        np.testing.assert_equal(model.target_view(Y), reference)


if __name__ == '__main__':
    unittest.main()
