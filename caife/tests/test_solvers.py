import unittest
from dataclasses import dataclass
from functools import partial
from unittest import TestCase

import numpy as np

import caife


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


@dataclass
class _PatternSolver(caife.solvers.AbstractSolver):
    """A solver whose i-th trial is valid iff `is_valid_fn(i)`; the i-th trial has the value `i`."""
    is_valid_fn: callable = lambda i: True

    def __post_init__(self):
        super().__post_init__()
        self.n_calls = 0

    def solve_latent(self, args):
        i = self.n_calls
        self.n_calls += 1
        return caife.solvers.LatentResult(
            ell=self.create_starting_vector(),
            is_valid=self.is_valid_fn(i),
            value=float(i),
            aux={"i": i},
        )


def _solve_with_pattern(is_valid_fn, **kwargs):
    """Solve with a `_PatternSolver`, returning the result and the number of trials."""
    solver = _PatternSolver(
        lambda params: np.array(0.),
        [_LatentVector(np.zeros(1))],
        is_valid_fn=is_valid_fn,
        **kwargs,
    )
    result, aux = solver.solve(return_aux=True)
    return result, aux, len(aux["latent_results"])


class TestMultiStart(TestCase):
    def test_defaults(self):
        # by default, the solver stops at the first valid trial
        _, _, n_trials = _solve_with_pattern(lambda i: True)
        self.assertEqual(n_trials, 1)
        _, _, n_trials = _solve_with_pattern(lambda i: i >= 4)
        self.assertEqual(n_trials, 5)

    def test_min_trials(self):
        # min_trials is enforced even if valid results are available earlier
        _, _, n_trials = _solve_with_pattern(lambda i: True, min_trials=5)
        self.assertEqual(n_trials, 5)

    def test_min_valid_trials(self):
        # every second trial is valid; the third valid trial is the sixth trial
        _, _, n_trials = _solve_with_pattern(lambda i: i % 2 == 1, min_valid_trials=3)
        self.assertEqual(n_trials, 6)

    def test_min_trials_and_min_valid_trials(self):
        # both criteria must be met: here, min_valid_trials takes longer ...
        _, _, n_trials = _solve_with_pattern(
            lambda i: i >= 3, min_trials=2, min_valid_trials=2)
        self.assertEqual(n_trials, 5)

        # ... and here, min_trials takes longer
        _, _, n_trials = _solve_with_pattern(
            lambda i: True, min_trials=4, min_valid_trials=2)
        self.assertEqual(n_trials, 4)

    def test_max_trials(self):
        # without any valid trial, the solver gives up after max_trials trials
        result, _, n_trials = _solve_with_pattern(lambda i: False, max_trials=7)
        self.assertEqual(n_trials, 7)
        self.assertIsNone(result)

        # max_trials also caps the search for more valid trials
        result, _, n_trials = _solve_with_pattern(
            lambda i: i == 0, min_valid_trials=5, max_trials=8)
        self.assertEqual(n_trials, 8)
        self.assertIsNotNone(result) # the single valid trial is still returned

    def test_best_valid_result(self):
        # the i-th trial has the value i, so the invalid first trial would be the
        # best one; the best valid trial is the second one, instead
        _, aux, n_trials = _solve_with_pattern(
            lambda i: i > 0, min_trials=4)
        self.assertEqual(n_trials, 4)
        self.assertEqual(aux["i"], 1)

    def test_invalid_configurations(self):
        for kwargs in [
                {"max_trials": 0},
                {"min_trials": 5, "max_trials": 4},
                {"min_valid_trials": 5, "max_trials": 4},
            ]:
            with self.assertRaises(ValueError):
                _solve_with_pattern(lambda i: True, **kwargs)


class TestAbstractSolver(TestCase):
    def test_solve(self):

        def nll(projected_components, result_components):
            np.testing.assert_equal(
                projected_components,
                tuple(x + 1 for x in result_components),
            )
            return np.array(-1)

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
            _, aux = _Solver(
                partial(nll, result_components=result_components),
                latent_vectors
            ).solve(return_aux=True)
            for latent_result in aux["latent_results"]:
                self.assertEqual(latent_result.value, -1)
                self.assertEqual(latent_result.aux, {"value": -1})


if __name__ == '__main__':
    unittest.main()
