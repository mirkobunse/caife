```{toctree}
:hidden:

self
api
developer-guide
```

# Quickstart

caife is a Python package for unfolding. As such, it provides techniques for **C**omposable and **A**uto-differentiable **I**nversion of **F**redholm **E**quations. The sound of its name (spoken as "cave") is reminiscent of Plato's cave analogy, which has been used to describe the indirect measurement processes that unfolding facilitates.

caife provides robust and flexible fits, high-quality binnings, and adequate estimates of statistical and systemic uncertainties.


## Installation

```
pip install --upgrade pip setuptools wheel
pip install 'caife @ git+https://github.com/mirkobunse/caife'
```

Moreover, you will need a [JAX](https://jax.readthedocs.io/) backend. Typically, the CPU backend will be ideal:

```
pip install "jax[cpu]"
```

### Upgrading

To upgrade an existing installation of `caife`, run

```
pip install --force-reinstall --no-deps 'caife @ git+https://github.com/mirkobunse/caife@main'
```


## Usage

See the Jupyter notebook at `examples/quickstart.ipynb`.
