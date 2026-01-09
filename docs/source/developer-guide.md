# Developer guide

In the following, we introduce best practices regarding the implementation [workflow](#workflow) before going into detail about how to take out [custom implementations](#custom-implementations).


## Workflow

Before you push to the `main` branch, please test the code and the documentation locally.


### Unit testing

Run tests locally with the `unittest` package.

```bash
python -m venv venv
venv/bin/pip install -e .[tests]
venv/bin/python -m unittest
```

As soon as you push to the `main` branch, GitHub Actions will take out these unit tests, too.


### Documentation

After locally building the documentation, open `docs/build/html/index.html` in your browser.

```bash
venv/bin/pip install -e .[docs]
venv/bin/sphinx-build -M html docs/source docs/build
```

As soon as you push to the `main` branch, GitHub Actions will build the documentation, push it to the `gh-pages` branch, and publish the result on GitHub Pages: [https://mirkobunse.github.io/caife](https://mirkobunse.github.io/caife)


### Profiling

The most simple thing to keep track of is the total amount of allocated GPU memory. To view this amount during execution, start your program with pre-allocation of memory being disabled and view the current allocation amount in another terminal.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false snakemake -c1

nvidia-smi -l 1 # start this command in another terminal!
```


## Custom implementations

TODO.
