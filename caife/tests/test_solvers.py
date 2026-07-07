import caife
import numpy as np
import unittest
from dataclasses import dataclass
from unittest import TestCase


@dataclass
class _LatentVector(caife.solvers.AbstractLatentVector):
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
    def test_create_split_fn(self):
        rng = np.random.default_rng(0)
        for i_trial in range(5):
            if i_trial == 0:
                n_components = 1 # ensure that a single component is tested
            else:
                n_components = rng.choice(10) + 2
            component_lengths = rng.choice(100, size=n_components) + 1

            # test the plain splitting of a vector
            result_components = tuple(rng.random(l) for l in component_lengths)
            split_fn = caife.solvers.create_split_fn(result_components)
            result_vector = np.concatenate(result_components)
            split_components = split_fn(result_vector)
            np.testing.assert_equal(
                tuple(split_components), # actual; requires conversion to tuple
                result_components, # desired
            )

            # test the projected splitting of a vector
            split_fn = caife.solvers.create_split_fn(
                result_components,
                projections=lambda x: x+1,
            )
            result_vector = np.concatenate(result_components)
            split_components = split_fn(result_vector)
            np.testing.assert_equal(
                tuple(split_components), # actual
                tuple(x+1 for x in result_components), # desired
            )

            # test that a plain splitting is the identity for a single component
            split_fn = caife.solvers.create_split_fn(result_components[0])
            split_components = split_fn(result_components[0])
            np.testing.assert_equal(
                tuple(split_fn(result_components[0]))[0], # requires conversion
                result_components[0],
            )

            # test the plain splitting of matrices
            result_components = tuple(rng.random((l, l)) for l in component_lengths)
            split_fn = caife.solvers.create_split_fn(
                result_components,
                two_dimensional=True,
            )
            result_matrix = np.zeros((
                np.sum(component_lengths),
                np.sum(component_lengths),
            ))
            i = 0
            for x in result_components:
                result_matrix[i:(i+len(x)),i:(i+len(x))] = x
                i += len(x)
            split_components = split_fn(result_matrix)
            np.testing.assert_equal(
                tuple(split_components), # actual; requires conversion to tuple
                result_components, # desired
            )

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
            def nll(*projected_components):
                np.testing.assert_equal(
                    projected_components,
                    tuple(x + 1 for x in result_components),
                )
                return -1
            result, aux = _Solver().solve(
                nll,
                *latent_vectors,
                return_aux=True,
            )
            self.assertEqual(aux, {"value": -1})


if __name__ == '__main__':
    unittest.main()
