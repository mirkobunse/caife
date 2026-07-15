import caife
import numpy as np
import unittest
from dataclasses import dataclass
from unittest import TestCase


@dataclass
class _LatentVector(caife.models.latents.AbstractLatentVector):
    starting_point: list
    def __call__(self, ell):
        return ell + 1
    def create_starting_point(self, rng=None):
        return self.starting_point


class _Solver(caife.solvers.AbstractSolver):
    def solve_latent(self, latent_nll, x0):
        value = latent_nll(x0) # call the function
        return x0, {"value": value} # = ell*, aux


class TestAbstractSolver(TestCase):
    def test_solve(self):
        rng = np.random.default_rng(0)
        for i_trial in range(5):
            if i_trial == 0:
                n_components = 1 # ensure that a single component is tested
            else:
                n_components = rng.choice(10) + 2
            component_lengths = rng.choice(100, size=n_components) + 1

            # test that solve_latent receives the right inputs
            result_components = tuple(rng.random(l) for l in component_lengths)
            latent_vectors = [_LatentVector(x) for x in result_components]
            def nll(projected_components):
                np.testing.assert_equal(
                    projected_components,
                    tuple(x + 1 for x in result_components),
                )
                return -1
            result, aux = _Solver().solve(
                nll,
                latent_vectors,
                return_aux=True,
            )
            self.assertEqual(aux, {"value": -1})


if __name__ == '__main__':
    unittest.main()
