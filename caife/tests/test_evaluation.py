import caife
import jax
import numpy as np
import unittest
from jax import numpy as jnp
from unittest import TestCase


class TestEvaluation(TestCase):
    def test_global_correlation_coefficients(self):
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
            gcc = caife.global_correlation_coefficients(result[0], nll)
            self.assertEqual(gcc, 0)

            # test with a tuple
            def nll(x_tuple):
                value = 0
                for x in x_tuple:
                    value += jnp.sum(x**2)
                return value
            gccs = caife.global_correlation_coefficients(result, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result))

            # test with a list
            result_list = list(result)
            gccs = caife.global_correlation_coefficients(result_list, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result_list))

            # test with a dict
            result_dict = {f"x{i}": x for i, x in enumerate(result)}
            def nll(x_dict):
                value = 0
                for x in x_dict.values():
                    value += jnp.sum(x**2)
                return value
            gccs = caife.global_correlation_coefficients(result_dict, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result_dict))

    def test_pairwise_correlation_scores(self):
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
            pcs = caife.pairwise_correlation_scores(result[0], nll)
            self.assertEqual(pcs, 0)

            # test with a tuple
            def nll(x_tuple):
                value = 0
                for x in x_tuple:
                    value += jnp.sum(x**2)
                return value
            pcss = caife.pairwise_correlation_scores(result, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result))

            # test with a list
            result_list = list(result)
            pcss = caife.pairwise_correlation_scores(result_list, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result_list))

            # test with a dict
            result_dict = {f"x{i}": x for i, x in enumerate(result)}
            def nll(x_dict):
                value = 0
                for x in x_dict.values():
                    value += jnp.sum(x**2)
                return value
            pcss = caife.pairwise_correlation_scores(result_dict, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result_dict))


if __name__ == '__main__':
    unittest.main()
