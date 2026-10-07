import os
import subprocess
import sys
import unittest
import warnings
from unittest import TestCase

import jax
import numpy as np

import caife

_IMPORT_SCRIPT = """
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
import jax, caife
print("x64 =", jax.config.jax_enable_x64)
"""


def _import_caife(jax_enable_x64=None):
    """Import caife in a fresh Python process; return the resulting jax_enable_x64 and the log output."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_ENABLE_X64"}
    if jax_enable_x64 is not None:
        env["JAX_ENABLE_X64"] = jax_enable_x64
    output = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        env=env, capture_output=True, text=True, check=True,
    )
    return "x64 = True" in output.stdout, output.stderr


class TestPrecision(TestCase):
    def test_default(self):
        # without an explicit choice, caife enables 64 bits and logs this side effect
        is_x64, log = _import_caife()
        self.assertTrue(is_x64)
        self.assertIn("caife: caife has enabled 64-bit precision", log)

    def test_explicit_choice(self):
        # an explicit value of JAX_ENABLE_X64 is respected, without any log message
        for value, expected in [("0", False), ("false", False), ("1", True)]:
            is_x64, log = _import_caife(value)
            self.assertEqual(is_x64, expected)
            self.assertNotIn("caife has enabled", log)

    def test_solver_warning(self):
        def create_solver():
            return caife.ScipySolver(
                lambda f: caife.poisson_nll(f, np.ones(3)),
                caife.LatentSpectrum(n_samples=3, n_bins_target=3),
            )
        self.assertTrue(jax.config.jax_enable_x64) # set by importing caife
        try:
            jax.config.update("jax_enable_x64", False) # switch back after importing caife
            with self.assertWarnsRegex(UserWarning, "32-bit precision") as context:
                create_solver()
            self.assertEqual(context.filename, __file__) # the warning points to the user code
        finally:
            jax.config.update("jax_enable_x64", True)

        # with 64 bits, the solver does not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error") # turn warnings into errors
            create_solver()


if __name__ == '__main__':
    unittest.main()
