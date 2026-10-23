import unittest
from unittest import TestCase

import numpy as np

import caife


class TestLatentSpectrum(TestCase):
    def assert_group_mapping(self, latent, ell, bin_groups):
        """Assert that each bin is driven by the latent parameter of its group, and by no other one."""
        f = latent(ell)
        self.assertEqual(f.shape, (latent.n_bins_target,)) # one count per bin
        _, ranks = np.unique(bin_groups, return_inverse=True) # rank 0 = smallest group = fixed
        self.assertEqual(len(ell), ranks.max()) # one parameter per group, except for the fixed one
        f_fixed = f[ranks == 0][0]
        for i in range(len(f)):
            ell_i = np.log(f[i] / f_fixed)
            if ranks[i] == 0: # the fixed group has a latent value of zero
                self.assertAlmostEqual(ell_i, 0)
            else: # each other group has the latent value of its own parameter
                self.assertAlmostEqual(ell_i, ell[ranks[i] - 1])
        self.assertAlmostEqual(f.sum(), latent.n_samples)

    def test_bin_groups(self):
        rng = np.random.default_rng(1491)
        n_samples = 12_345
        n_bins_target = 10
        latent = caife.LatentSpectrum(n_samples, n_bins_target)

        # test parameters without sharing
        ell = latent.create_starting_point(rng)
        self.assertEqual(ell.shape, (n_bins_target - 1,))

        # create a template for parameter sharing
        bin_groups = caife.LatentSpectrum.create_bin_groups(n_bins_target)
        self.assertEqual(bin_groups[0], -1)

        # make two bins with free parameters equal
        bin_groups[2] = bin_groups[1]
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, bin_groups)
        sharing_ell = sharing_latent.create_starting_point(rng)
        for _ in range(5):
            self.assert_group_mapping(sharing_latent, sharing_ell, bin_groups)
            sharing_ell = sharing_latent.create_starting_point(rng) # new draw

        # do the same with a fixed parameter
        bin_groups = caife.LatentSpectrum.create_bin_groups(n_bins_target)
        bin_groups[1] = bin_groups[0] # both fixed
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, bin_groups)
        sharing_ell = sharing_latent.create_starting_point(rng)
        for _ in range(5):
            self.assert_group_mapping(sharing_latent, sharing_ell, bin_groups)
            sharing_ell = sharing_latent.create_starting_point(rng) # new draw

        # arbitrary group numbers with gaps, where the smallest group number is the fixed one
        bin_groups = np.array([7, 3, 3, 9, 9, 9, 12, 5, 5, 20])
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, bin_groups)
        for _ in range(5):
            sharing_ell = sharing_latent.create_starting_point(rng)
            self.assert_group_mapping(sharing_latent, sharing_ell, bin_groups)

        # each bin needs exactly one group number
        with self.assertRaises(ValueError):
            caife.LatentSpectrum(n_samples, n_bins_target, bin_groups[:-1])


if __name__ == '__main__':
    unittest.main()
