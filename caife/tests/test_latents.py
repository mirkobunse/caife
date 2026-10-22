import unittest
from unittest import TestCase

import numpy as np

import caife


class TestLatentSpectrum(TestCase):
    def test_sharing(self):
        rng = np.random.default_rng(1491)
        n_samples = 12_345
        n_bins_target = 10
        latent = caife.LatentSpectrum(n_samples, n_bins_target)

        # test parameters without sharing
        ell = latent.create_starting_point(rng)
        self.assertEqual(ell.shape, (n_bins_target - 1,))

        # create a template for parameter sharing
        sharing_indices = np.array(
            caife.LatentSpectrum.parameter_indices(n_bins_target))
        self.assertEqual(sharing_indices[0], -1)

        # make two bins with free parameters equal
        sharing_indices[2] = sharing_indices[1]
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, sharing_indices)
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
        sharing_indices = np.array(
            caife.LatentSpectrum.parameter_indices(n_bins_target))
        sharing_indices[1] = sharing_indices[0] # both fixed
        sharing_latent = caife.LatentSpectrum(
            n_samples, n_bins_target, sharing_indices)
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
