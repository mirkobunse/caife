import unittest
from unittest import TestCase

import numpy as np

import caife


class TestLatentSpectrum(TestCase):
    def test_bin_groups(self):
        rng = np.random.default_rng(1491)
        n_samples = 12_345
        n_bins_target = 10
        latent = caife.LatentSpectrum(n_samples, n_bins_target)

        # test parameters without sharing
        ell = latent.create_starting_point(rng)
        self.assertEqual(ell.shape, (n_bins_target - 1,))

        # create a template for parameter sharing
        bin_groups = np.array(
            caife.LatentSpectrum.create_bin_groups(n_bins_target))
        self.assertEqual(bin_groups[0], -1)

        # make two bins with free parameters equal
        bin_groups[2] = bin_groups[1]
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, bin_groups)
        sharing_ell = sharing_latent.create_starting_point(rng)
        self.assertEqual(sharing_ell.shape, (n_bins_target - 2,)) # one less
        self.assertEqual( # output shapes match
            sharing_latent(sharing_ell).shape,
            latent(ell).shape,
        )
        for _ in range(5):
            self.assertEqual(
                sharing_latent(sharing_ell)[1],
                sharing_latent(sharing_ell)[2],
            )
            sharing_ell = sharing_latent.create_starting_point(rng) # new draw

        # do the same with a fixed parameter
        bin_groups = np.array(
            caife.LatentSpectrum.create_bin_groups(n_bins_target))
        bin_groups[1] = bin_groups[0] # both fixed
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, bin_groups)
        sharing_ell = sharing_latent.create_starting_point(rng)
        self.assertEqual(sharing_ell.shape, (n_bins_target - 2,)) # one less
        self.assertEqual( # output shapes match
            sharing_latent(sharing_ell).shape,
            latent(ell).shape,
        )
        for _ in range(5):
            self.assertEqual(
                sharing_latent(sharing_ell)[0],
                sharing_latent(sharing_ell)[1],
            )
            sharing_ell = sharing_latent.create_starting_point(rng) # new draw


if __name__ == '__main__':
    unittest.main()
