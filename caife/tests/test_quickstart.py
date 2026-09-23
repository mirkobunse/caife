"""Tests that mirror the unfolding pipeline of `examples/quickstart.ipynb`."""

import unittest
from unittest import TestCase

import numpy as np
from jax import numpy as jnp

import caife
from caife import examples


def create_quickstart_setup(seed=1491):
    """Set up the model, the data, and the likelihood of the quickstart notebook."""
    (X_trn, y_trn, w_trn, S_trn), (X_tst, y_tst) = examples.split_data(
        *examples.create_data(n_samples=20_000, rng=seed), rng=seed)
    target_bins = np.concatenate(([-np.inf], np.linspace(-4, 5, 10), [np.inf]))
    proxy_bins = np.concatenate(([-np.inf], np.linspace(-6, 8, 15), [np.inf]))

    model = caife.LinearCountModel(target_bins, caife.UnivariateBinning(proxy_bins))
    model.fit(X_trn, y_trn, sample_weight=w_trn)

    g_tst = model.proxy_view(X_tst)
    a_eff = examples.create_effective_areas(target_bins)
    def nll(params, tau):
        value = caife.poisson_nll(model(params), g_tst)
        value += caife.tikhonov_regularization(
            jnp.log(params[1:-1] / a_eff + 1e-10)) / tau
        return value
    solver = caife.ScipySolver(
        nll, model.create_latents(X_tst), seed=seed, n_trials=3)

    return {
        "model": model,
        "nll": nll,
        "solver": solver,
        "target_bins": target_bins,
        "proxy_bins": proxy_bins,
        "X_trn": X_trn,
        "y_trn": y_trn,
        "w_trn": w_trn,
        "S_trn": S_trn,
        "X_tst": X_tst,
        "y_tst": y_tst,
        "a_eff": a_eff,
    }


class TestQuickstart(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = create_quickstart_setup() # JIT the loss function only once

    def test_views(self):
        model, target_bins, proxy_bins = (
            self.setup["model"], self.setup["target_bins"], self.setup["proxy_bins"])

        # the views are count spectra over the respective bins
        for f in [
                model.target_view(self.setup["y_trn"]),
                model.target_view(self.setup["y_trn"], sample_weight=self.setup["w_trn"]),
                model.target_view(self.setup["y_tst"]),
            ]:
            self.assertEqual(f.shape, (len(target_bins) - 1,))
            self.assertTrue(np.all(f >= 0))
            self.assertAlmostEqual(f.sum(), len(self.setup["y_trn"]), delta=1)

        for g in [model.proxy_view(self.setup["X_trn"]), model.proxy_view(self.setup["X_tst"])]:
            self.assertEqual(g.shape, (len(proxy_bins) - 1,))
            self.assertTrue(np.all(g >= 0))
            self.assertAlmostEqual(g.sum(), len(self.setup["X_trn"]), delta=1)

    def test_response_matrix(self):
        model, target_bins, proxy_bins = (
            self.setup["model"], self.setup["target_bins"], self.setup["proxy_bins"])

        # A maps the target bins to the proxy bins; each of its columns is a
        # proxy distribution and therefore sums to one
        self.assertEqual(model.A_.shape, (len(proxy_bins) - 1, len(target_bins) - 1))
        np.testing.assert_allclose(
            np.asarray(model.A_).sum(axis=0),
            np.ones(len(target_bins) - 1),
            atol=1e-6,
        )

    def test_unfolding(self, selected_tau=1e-3):
        f_tst = self.setup["model"].target_view(self.setup["y_tst"])
        f_est, aux = self.setup["solver"].solve(args=(selected_tau,), return_aux=True)

        # a valid solution is found
        self.assertIsNotNone(f_est)
        self.assertEqual(f_est.shape, f_tst.shape)

        # the LatentSpectrum guarantees a positive spectrum that sums to the
        # number of observations that are being unfolded
        self.assertTrue(np.all(f_est > 0))
        self.assertAlmostEqual(f_est.sum(), len(self.setup["X_tst"]), delta=1)

        # the solution resembles the ground truth
        relative_errors = np.abs(f_est[1:-1] - f_tst[1:-1]) / f_tst[1:-1]
        self.assertLess(relative_errors.mean(), .1)

        # the uncertainties bracket the solution
        f_lower, f_upper = caife.uncertainty_from_aux(
            aux, n_samples=1_000, rng=np.random.default_rng(1491))
        self.assertTrue(np.all(f_lower <= f_est))
        self.assertTrue(np.all(f_est <= f_upper))

        # the auxiliary information describes a valid, well-conditioned fit
        self.assertTrue(np.isfinite(aux["cond"]))
        self.assertGreater(aux["cond"], 0)
        self.assertTrue(any(result.is_valid for result in aux["latent_results"]))

    def test_criteria_of_the_tau_scan(self):
        n_valid = 0
        for tau in [1e0, 1e-1, 1e-2]:
            f_est, aux = self.setup["solver"].solve(args=(tau,), return_aux=True)
            if f_est is None:
                continue # invalid results are skipped, just as in the notebook
            n_valid += 1

            # all criteria of the notebook are computable and in a sane range
            gcc = caife.global_correlation_coefficients(
                f_est, self.setup["nll"], args=(tau,))
            pcs = caife.pairwise_correlation_scores(
                f_est, self.setup["nll"], args=(tau,))
            for correlation in [gcc, pcs]:
                self.assertGreaterEqual(correlation, 0.)
                self.assertLessEqual(correlation, 1.)
            self.assertTrue(np.isfinite(aux["cond"]))

        self.assertGreater(n_valid, 0) # at least one tau yields a valid result


if __name__ == '__main__':
    unittest.main()
