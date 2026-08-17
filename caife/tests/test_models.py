import unittest
from unittest import TestCase

import numpy as np

import caife
from caife.models.collections import SystematicBinCollection


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


class TestSystematicBinCollection(TestCase):
    def test_create_bin_indices(self):
        rng = np.random.default_rng(1491)
        for n_systematics in range(1, 6):
            systematics = rng.uniform(size=(10_000, n_systematics))
            for n_bins_per_systematic in range(2, 6):

                # test create_systematic_bins
                systematic_bins = SystematicBinCollection.create_systematic_bins(
                    systematics,
                    n_bins_per_systematic,
                )
                self.assertEqual(
                    systematic_bins.shape,
                    (n_systematics, n_bins_per_systematic+1),
                )
                np.testing.assert_equal(systematic_bins[:,0], -np.inf)
                np.testing.assert_equal(systematic_bins[:,-1], np.inf)

                # test create_bin_indices
                bin_indices = SystematicBinCollection.create_bin_indices(
                    systematics,
                    systematic_bins
                )
                self.assertEqual(
                    len(bin_indices),
                    n_bins_per_systematic ** n_systematics,
                )

                # check that every sample occurs exactly once
                n_sampled = np.zeros(len(systematics), dtype=int)
                for is_in_bin in bin_indices.values():
                    n_sampled[is_in_bin] += 1
                self.assertEqual(n_sampled.min(), 1)
                self.assertEqual(n_sampled.max(), 1)


if __name__ == '__main__':
    unittest.main()
