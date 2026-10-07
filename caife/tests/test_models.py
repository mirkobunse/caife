import unittest
from unittest import TestCase

import numpy as np

import caife
from caife import examples


class TestLinearCountModel(TestCase):
    def test(self):
        rng = np.random.default_rng(1491)
        y = rng.uniform(size=1_000) * 4
        X = (y + rng.normal(size=1_000)).reshape((-1, 1))

        # assert that target_bins are required to range from -inf to inf
        erroring_model = caife.LinearCountModel(
            target_bins=np.arange(5), # not ranging from -inf to inf
            representation=caife.UnivariateBinning(
                proxy_bins=np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            )
        )
        self.assertRaises( # erroring_model.fit(X, y) raises a ValueError
            ValueError,
            erroring_model.fit,
            X, # *args
            y,
        )

        # assert correct target representation
        target_bins = np.arange(5, dtype=float)
        target_bins[0] = -np.inf
        target_bins[-1] = np.inf
        model = caife.LinearCountModel(
            target_bins=target_bins,
            representation=caife.UnivariateBinning(
                proxy_bins=np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            )
        )
        # y ∈ [ 0, 1, 2, 3 ], X ∈ [ 0, 1, 2, 3, 4, 5 ]
        self.assertEqual(model.represent_target(y).min(), 0)
        self.assertEqual(model.represent_target(y).max(), 3)
        for value in [ np.nan, np.inf, -np.inf ]:
            self.assertRaises( # model.represent_target(inf | nan) raises a ValueError
                ValueError,
                model.represent_target,
                [ value ], # *args
            )
        model.fit(X, y)
        self.assertEqual(model.A_.shape, (6, 4))

        # systematics are ignored with a warning that points to the call of fit
        with self.assertWarnsRegex(UserWarning, "LinearSystematicsCountModel") as context:
            model.fit(X, y, systematics=rng.uniform(size=(1_000, 1)))
        self.assertEqual(context.filename, __file__)


class TestMultiTargetModel(TestCase):
    def test(self):
        rng = np.random.default_rng(1491)
        Y = rng.uniform(size=(1_000, 2)) * 4
        X = np.stack((
            (Y[:,0] + rng.normal(size=1_000)),
            (Y[:,1] + rng.normal(size=1_000)),
        )).T

        # assert that target_bins are required to range from -inf to inf
        representation = caife.GridBinning([
            np.concatenate(([-np.inf], np.arange(5), [np.inf])),
            np.concatenate(([-np.inf], np.arange(3), [np.inf])),
        ])
        erroring_model = caife.MultiTargetModel(
            target_bins=[np.arange(5), np.arange(3)], # not ranging from -inf to inf
            representation=representation,
        )
        self.assertRaises( # erroring_model.fit(X, y) raises a ValueError
            ValueError,
            erroring_model.fit,
            X, # *args
            Y,
        )

        # assert correct target representation
        target_bins = [
            np.arange(5, dtype=float), # 4 classes
            np.arange(3, dtype=float), # 2 classes
        ]
        for b_i in target_bins:
            b_i[0] = -np.inf
            b_i[-1] = np.inf
        model = caife.MultiTargetModel(
            target_bins=target_bins,
            representation=representation,
        )
        self.assertEqual(model.represent_targets(Y).min(), 0)
        self.assertEqual(model.represent_targets(Y).max(), 7)
        for value in [ np.nan, np.inf, -np.inf ]:
            self.assertRaises( # model.represent_target(inf | nan) raises a ValueError
                ValueError,
                model.represent_targets,
                [ value ], # *args
            )
        model.fit(X, Y)

        # systematics are ignored with a warning that points to the call of fit
        with self.assertWarnsRegex(UserWarning, "MultiTargetSystematicsModel") as context:
            model.fit(X, Y, systematics=rng.uniform(size=(1_000, 1)))
        self.assertEqual(context.filename, __file__)

        # instantiate standard models as a reference
        model_a = caife.LinearCountModel(
            target_bins=target_bins[0],
            representation=caife.UnivariateBinning(representation.proxy_bins[0]),
        )
        model_b = caife.LinearCountModel(
            target_bins=target_bins[1],
            representation=caife.UnivariateBinning(representation.proxy_bins[1]),
        )

        # check for equivalence in target and proxy views
        np.testing.assert_equal(
            model.represent_targets(Y, separate=True)[:,0],
            model_a.represent_target(Y[:,0]),
        )
        np.testing.assert_equal(
            model.represent_targets(Y, separate=True)[:,1],
            model_b.represent_target(Y[:,1]),
        )
        np.testing.assert_equal(
            caife.MultiTargetModel(
                target_bins=target_bins,
                representation=caife.GridBinning([representation.proxy_bins[0]]),
            ).fit(X[:,[0]], Y).proxy_view(X[:,[0]]),
            model_a.proxy_view(X[:,[0]]),
        )
        np.testing.assert_equal(
            caife.MultiTargetModel(
                target_bins=target_bins,
                representation=caife.GridBinning([representation.proxy_bins[1]]),
            ).fit(X[:,[1]], Y).proxy_view(X[:,[1]]),
            model_b.proxy_view(X[:,[1]]),
        )
        y_a = model_a.represent_target(Y[:,0])
        y_b = model_b.represent_target(Y[:,1])
        reference = np.zeros(model.n_bins_per_target)
        for i_a in range(model_a.n_bins_target):
            for i_b in range(model_b.n_bins_target):
                reference[i_a, i_b] = np.sum((y_a == i_a) & (y_b == i_b))
        np.testing.assert_equal(model.target_view(Y), reference)


class TestLinearSystematicsCountModel(TestCase):
    def setUp(self):
        self.target_bins = np.concatenate(([-np.inf], np.linspace(-4, 5, 10), [np.inf]))
        self.proxy_bins = np.concatenate(([-np.inf], np.linspace(-6, 8, 15), [np.inf]))
        (self.X, self.y, self.w, self.S), _ = examples.split_data(
            *examples.create_data(n_samples=30_000, rng=1491), rng=1491)

    def create_model(self):
        return caife.LinearSystematicsCountModel(
            self.target_bins,
            caife.UnivariateBinning(self.proxy_bins),
            solver_options={"gtol": 1e-6, "maxiter": 1_000},
        )

    def test_systematics_are_required(self):
        self.assertRaises( # fitting without systematics raises a ValueError
            ValueError,
            self.create_model().fit,
            self.X, # *args
            self.y,
        )

    def test_model(self):
        model = self.create_model().fit(
            self.X, self.y, sample_weight=self.w, systematics=self.S)
        n_bins_proxy = len(self.proxy_bins) - 1
        n_bins_target = len(self.target_bins) - 1

        # the fit produces one logistic regression per proxy bin and target bin,
        # each with one coefficient per systematic parameter, plus a bias term
        self.assertEqual(
            model.coeffs_.shape,
            (n_bins_proxy, n_bins_target, self.S.shape[1] + 1),
        )
        self.assertEqual(model.A_mask.shape, (n_bins_proxy, n_bins_target))

        # the bounds are taken from the systematics of the training data
        np.testing.assert_allclose(model.systematic_bounds[:, 0], self.S.min(axis=0))
        np.testing.assert_allclose(model.systematic_bounds[:, 1], self.S.max(axis=0))

        # each column of A(s) is a proxy distribution and therefore sums to one
        for s in [np.array([-1.]), np.array([0.]), np.array([1.])]:
            A = np.asarray(model.A(s))
            self.assertEqual(A.shape, (n_bins_proxy, n_bins_target))
            self.assertTrue(np.all(A >= 0))
            np.testing.assert_allclose(A.sum(axis=0), np.ones(n_bins_target), atol=1e-6)

        # in the toy data, the proxy is smeared around the target with a mean
        # that shifts by 0.5*s; since the proxy bins are one unit wide, A(s)
        # must shift each target bin's proxy distribution by about one bin as
        # s goes from -1 to 1
        i_proxy = np.arange(len(self.proxy_bins) - 1)
        A_low = np.asarray(model.A(np.array([-1.])))
        A_high = np.asarray(model.A(np.array([1.])))
        for i_target in range(3, len(self.target_bins) - 3): # skip the outer bins
            shift = (A_high[:, i_target] - A_low[:, i_target]) @ i_proxy
            self.assertGreater(shift, .5)
            self.assertLess(shift, 2.)

        # the model consumes a pair (f, s) of latents, unlike the systematics-
        # unaware LinearCountModel, which consumes f alone
        latents = model.create_latents(self.X)
        self.assertEqual(len(latents), 2)
        self.assertIsInstance(latents[0], caife.LatentSpectrum)
        self.assertIsInstance(latents[1], caife.LatentSystematics)

        f = np.asarray(model.target_view(self.y), dtype=float)
        g_pred = model((f, np.array([0.])))
        self.assertEqual(g_pred.shape, (len(self.proxy_bins) - 1,))
        self.assertTrue(np.all(np.asarray(g_pred) >= 0))

    def test_fit_without_sample_weight(self):
        model = self.create_model().fit(self.X, self.y, systematics=self.S)
        np.testing.assert_allclose(
            np.asarray(model.A(np.array([0.]))).sum(axis=0),
            np.ones(len(self.target_bins) - 1),
            atol=1e-6,
        )


class TestMultiTargetSystematicsModel(TestCase):
    def setUp(self):
        rng = np.random.default_rng(1491)
        n_samples = 20_000
        self.target_bins = [
            np.concatenate(([-np.inf], np.arange(1, 4, dtype=float), [np.inf])),
            np.concatenate(([-np.inf], np.arange(1, 4, dtype=float), [np.inf])),
        ]
        self.representation = caife.GridBinning([
            np.concatenate(([-np.inf], np.linspace(0, 4, 5), [np.inf])),
            np.concatenate(([-np.inf], np.linspace(0, 4, 5), [np.inf])),
        ])
        self.Y = rng.uniform(size=(n_samples, 2)) * 4
        self.S = rng.uniform(-1, 1, size=(n_samples, 1))
        self.X = np.stack(( # both proxies are shifted by the systematics
            self.Y[:, 0] + .5 * self.S[:, 0] + rng.normal(size=n_samples),
            self.Y[:, 1] + .5 * self.S[:, 0] + rng.normal(size=n_samples),
        )).T

    def create_model(self):
        return caife.MultiTargetSystematicsModel(
            self.target_bins,
            self.representation,
            solver_options={"gtol": 1e-6, "maxiter": 1_000},
        )

    def test_systematics_are_required(self):
        self.assertRaises( # fitting without systematics raises a ValueError
            ValueError,
            self.create_model().fit,
            self.X, # *args
            self.Y,
        )

    def test_model(self):
        model = self.create_model().fit(self.X, self.Y, systematics=self.S)
        n_bins_proxy = self.representation.n_output_features
        n_bins_multitarget = model.n_bins_multitarget

        np.testing.assert_array_equal(model.n_bins_per_target, [4, 4])
        self.assertEqual(n_bins_multitarget, 16)
        self.assertEqual(
            model.coeffs_.shape,
            (n_bins_proxy, n_bins_multitarget, self.S.shape[1] + 1),
        )

        # each column of A(s) is a proxy distribution and therefore sums to one
        A = np.asarray(model.A(np.array([0.])))
        self.assertEqual(A.shape, (n_bins_proxy, n_bins_multitarget))
        np.testing.assert_allclose(
            A.sum(axis=0), np.ones(n_bins_multitarget), atol=1e-6)

        # f is a multi-dimensional histogram over all target quantities
        latents = model.create_latents(self.X)
        self.assertEqual(len(latents), 2)
        self.assertIsInstance(latents[0], caife.LatentReshape)
        self.assertIsInstance(latents[1], caife.LatentSystematics)

        f = np.asarray(model.target_view(self.Y), dtype=float)
        self.assertEqual(f.shape, tuple(model.n_bins_per_target))

        g_pred = model((f, np.array([0.])))
        self.assertEqual(g_pred.shape, (self.representation.n_output_features,))
        self.assertTrue(np.all(np.asarray(g_pred) >= 0))


if __name__ == '__main__':
    unittest.main()
