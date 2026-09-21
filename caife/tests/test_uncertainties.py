import unittest
from unittest import TestCase

import jax
import numpy as np
from jax import numpy as jnp

import caife


class TestEvaluation(TestCase):
    def test_uncertainty_from_hessian(self):
        rng = np.random.default_rng(0)
        for i_trial in range(5):
            if i_trial == 0:
                n_components = 1 # ensure that a single component is tested
            else:
                n_components = rng.choice(10) + 2
            component_lengths = rng.choice(100, size=n_components) + 3
            result = tuple(rng.random(l) for l in component_lengths)

            # test with a single result component
            def nll(x):
                return jnp.sum(x**2)
            errors = caife.uncertainty_from_hessian(result[0], nll)
            self.assertEqual(len(errors), len(result[0]))

            # test with a tuple
            def nll(x_tuple):
                value = 0
                for x in x_tuple:
                    value += jnp.sum(x**2)
                return value
            errors = caife.uncertainty_from_hessian(result, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(
                lambda x, y: len(x)==len(y), errors, result)))

            # test with a list
            result_list = list(result)
            errors = caife.uncertainty_from_hessian(result_list, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(
                lambda x, y: len(x)==len(y), errors, result_list)))

            # test with a dict
            result_dict = {f"x{i}": x for i, x in enumerate(result)}
            def nll(x_dict):
                value = 0
                for x in x_dict.values():
                    value += jnp.sum(x**2)
                return value
            errors = caife.uncertainty_from_hessian(result_dict, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(
                lambda x, y: len(x)==len(y), errors, result_dict)))


if __name__ == '__main__':
    unittest.main()
