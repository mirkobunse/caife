import unittest
import warnings
from unittest import TestCase

import jax
import numpy as np
from jax import numpy as jnp

import caife


class TestEvaluation(TestCase):
    def test_gcc_from_nll(self):
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
            gcc = caife.gcc_from_nll(result[0], nll)
            self.assertEqual(gcc, 0)

            # test with a tuple
            def nll(x_tuple):
                value = 0
                for x in x_tuple:
                    value += jnp.sum(x**2)
                return value
            gccs = caife.gcc_from_nll(result, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result))

            # test with a list
            result_list = list(result)
            gccs = caife.gcc_from_nll(result_list, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result_list))

            # test with a dict
            result_dict = {f"x{i}": x for i, x in enumerate(result)}
            def nll(x_dict):
                value = 0
                for x in x_dict.values():
                    value += jnp.sum(x**2)
                return value
            gccs = caife.gcc_from_nll(result_dict, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, gccs)))
            self.assertEqual(jax.tree.structure(gccs), jax.tree.structure(result_dict))

    def test_pcs_from_nll(self):
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
            pcs = caife.pcs_from_nll(result[0], nll)
            self.assertEqual(pcs, 0)

            # test with a tuple
            def nll(x_tuple):
                value = 0
                for x in x_tuple:
                    value += jnp.sum(x**2)
                return value
            pcss = caife.pcs_from_nll(result, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result))

            # test with a list
            result_list = list(result)
            pcss = caife.pcs_from_nll(result_list, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result_list))

            # test with a dict
            result_dict = {f"x{i}": x for i, x in enumerate(result)}
            def nll(x_dict):
                value = 0
                for x in x_dict.values():
                    value += jnp.sum(x**2)
                return value
            pcss = caife.pcs_from_nll(result_dict, nll)
            self.assertTrue(jax.tree.all(jax.tree.map(lambda x: x==0, pcss)))
            self.assertEqual(jax.tree.structure(pcss), jax.tree.structure(result_dict))

        # regression test: sub-correlation blocks must be distinguished; a bug
        # computed each score from the fill joint correlation matrix
        r_a, r_b = 0.0, 0.6 # a has independent bins, b has a correlated pair
        def nll(x_tuple):
            a, b = x_tuple
            return (
                .5 * jnp.sum(a**2) + r_a * a[1] * a[2]
                + .5 * jnp.sum(b**2) + r_b * b[1] * b[2]
            )
        result = (jnp.array([.1, .2, .3, .4]), jnp.array([.4, .3, .2, .1]))

        # no overflow: 2*2 bins = one off-diagonal, so the score equals |r|
        pcs_a, pcs_b = caife.pcs_from_nll(result, nll)
        self.assertAlmostEqual(pcs_a, abs(r_a))
        self.assertAlmostEqual(pcs_b, abs(r_b))
        self.assertNotAlmostEqual(pcs_a, pcs_b) # the two components must not collapse

        # with overflow: 4*4 bins = 6 off-diagonals, so the score is |r| / 6
        pcs_a, pcs_b = caife.pcs_from_nll(result, nll, ignore_overflow_bins=False)
        self.assertAlmostEqual(pcs_a, abs(r_a) / 6)
        self.assertAlmostEqual(pcs_b, abs(r_b) / 6)
        self.assertNotAlmostEqual(pcs_a, pcs_b)


def _gaussian_aux(cov, unravel_fn):
    """Create the auxiliary information of a solver whose latent vectors are the identity."""
    return {
        "ell": np.zeros(cov.shape[0]),
        "hess": np.linalg.inv(cov),
        "unravel_fn": unravel_fn,
    }


def _gcc_and_pcs(cov):
    """Directly compute the mean global correlation coefficient and the pair-wise correlation score."""
    gcc = np.sqrt(1 - 1 / (np.diagonal(cov) * np.diagonal(np.linalg.inv(cov))))
    corr = cov / np.sqrt(np.outer(np.diagonal(cov), np.diagonal(cov)))
    return gcc.mean(), np.mean(np.abs(corr[np.triu_indices(len(cov), k=1)]))


class TestFromAux(TestCase):
    def test_marginalization(self):
        # a random covariance of two components, a with 5 entries and b with 3 entries
        rng = np.random.default_rng(0)
        X = rng.normal(size=(8, 8))
        cov = X @ X.T + np.eye(8)
        aux = _gaussian_aux(cov, lambda ell: (ell[:5], ell[5:]))

        # the scores of a are computed from the marginal covariance of its (inner) entries
        for ignore_overflow_bins, idx in [(False, np.arange(5)), (True, np.arange(1, 4))]:
            gcc_a, _ = caife.gcc_from_aux(aux, ignore_overflow_bins=ignore_overflow_bins)
            pcs_a, _ = caife.pcs_from_aux(aux, ignore_overflow_bins=ignore_overflow_bins)
            expected_gcc_a, expected_pcs_a = _gcc_and_pcs(cov[np.ix_(idx, idx)])
            self.assertAlmostEqual(gcc_a, expected_gcc_a)
            self.assertAlmostEqual(pcs_a, expected_pcs_a)

    def test_latent_spectrum(self):
        # fitting Poisson counts g with a LatentSpectrum, which fixes the total number of
        # counts, yields the multinomial covariance diag(g) - g g^T / N; its correlations
        # are therefore known in closed form
        g = jnp.array([50., 120., 300., 200., 80., 30.])
        N = g.sum()
        solver = caife.ScipySolver(
            lambda f: caife.poisson_nll(f, g), # the response matrix is the identity
            caife.LatentSpectrum(n_samples=N, n_bins_target=len(g)),
            seed=0,
        )
        f_est, aux = solver.solve(return_aux=True)
        np.testing.assert_allclose(f_est, g, rtol=1e-6)
        multinomial_cov = np.diag(g) - np.outer(g, g) / N
        expected_gcc, expected_pcs = _gcc_and_pcs(multinomial_cov[1:-1, 1:-1])
        self.assertAlmostEqual(caife.gcc_from_aux(aux), expected_gcc, places=6)
        self.assertAlmostEqual(caife.pcs_from_aux(aux), expected_pcs, places=6)

        # without ignoring any bin, the covariance is singular, because each
        # bin is an exact linear combination of the other bins
        self.assertAlmostEqual(
            caife.gcc_from_aux(aux, ignore_overflow_bins=False), 1, places=6)
        corr = multinomial_cov / np.sqrt(np.outer(np.diag(multinomial_cov), np.diag(multinomial_cov)))
        expected_pcs = np.mean(np.abs(corr[np.triu_indices(len(g), k=1)]))
        self.assertAlmostEqual(
            caife.pcs_from_aux(aux, ignore_overflow_bins=False), expected_pcs, places=6)

    def test_pytrees(self):
        # a dict of a 2D component f, with shape (3, 4), and of a single-entry component s
        rng = np.random.default_rng(1)
        X = rng.normal(size=(13, 13))
        cov = X @ X.T + np.eye(13)
        aux = _gaussian_aux(cov, lambda ell: {"f": ell[:12].reshape(3, 4), "s": ell[12:]})
        with warnings.catch_warnings():
            warnings.simplefilter("error") # turn warnings into errors
            for fn in [caife.gcc_from_aux, caife.pcs_from_aux]:
                scores = fn(aux)
                self.assertEqual(jax.tree.structure(scores), jax.tree.structure({"f": 0, "s": 0}))

                # by default, the overflow bins are excluded along every axis of f, and no
                # entry of s remains
                idx = np.arange(12).reshape(3, 4)[1:-1, 1:-1].reshape(-1)
                expected = _gcc_and_pcs(cov[np.ix_(idx, idx)])[fn == caife.pcs_from_aux]
                self.assertAlmostEqual(scores["f"], expected)
                self.assertTrue(np.isnan(scores["s"]))

                # a tuple of bools controls each axis of f separately
                scores = fn(aux, ignore_overflow_bins={"f": (False, True), "s": False})
                idx = np.arange(12).reshape(3, 4)[:, 1:-1].reshape(-1)
                expected = _gcc_and_pcs(cov[np.ix_(idx, idx)])[fn == caife.pcs_from_aux]
                self.assertAlmostEqual(scores["f"], expected)
                if fn == caife.gcc_from_aux:
                    self.assertEqual(scores["s"], 0) # no other entries to correlate with
                else:
                    self.assertTrue(np.isnan(scores["s"])) # no pairs of entries

                # the tuple must have one bool per axis
                with self.assertRaises(ValueError):
                    fn(aux, ignore_overflow_bins={"f": (True,), "s": False})


if __name__ == '__main__':
    unittest.main()
