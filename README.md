# caife

caife is a Python package for unfolding. As such, it provides techniques for **C**omposable and **A**uto-differentiable **I**nversion of **F**redholm **E**quations. The sound of its name (spoken as "cave") is reminiscent of Plato's cave analogy, which has been used to describe the indirect measurement processes that unfolding facilitates.

caife provides robust and flexible fits, high-quality binnings, and adequate estimates of statistical and systemic uncertainties.

## Installation

```sh
pip install --upgrade pip setuptools wheel
pip install 'caife @ git+https://github.com/mirkobunse/caife'
```

Moreover, you will need a [JAX](https://jax.readthedocs.io/) backend. Typically, the CPU backend will be ideal:

```sh
pip install "jax[cpu]"
```

## Usage

For a quickstart quide, see the Jupyter notebook at `examples/quickstart.ipynb`.

For more detailed information, visit [the documentation](https://mirkobunse.github.io/caife).

To build the documentation locally, issue the following commands and open `docs/build/html/index.html` in your browser.

```bash
venv/bin/pip install -e .[docs]
venv/bin/sphinx-build -M html docs/source docs/build
```
