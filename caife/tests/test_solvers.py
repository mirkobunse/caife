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
    def solve_latent(self, args):
        value = self.latent_nll_(self.create_starting_vector(), *args) # call the function
        return caife.solvers.LatentResult(
            ell=self.create_starting_vector(),
            is_valid=True,
            value=value,
            aux={"value": value},
        )


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
                return np.array(-1)
            _, aux = _Solver(nll, latent_vectors).solve(return_aux=True)
            for latent_result in aux["latent_results"]:
                self.assertEqual(latent_result.value, -1)
                self.assertEqual(latent_result.aux, {"value": -1})


if __name__ == '__main__':
    unittest.main()
