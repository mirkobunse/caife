# caife

caife is a Python package for unfolding. As such, it provides techniques for **C**omposable and **A**uto-differentiable **I**nversion of **F**redholm **E**quations. The sound of its name (spoken as "cave") is reminiscent of Plato's cave analogy, which has been used to describe the indirect measurement processes that unfolding facilitates.

caife provides robust and flexible fits, high-quality binnings, and adequate estimates of statistical and systemic uncertainties.

## Usage

TODO.

## Slurm setup

Use this setup if you're working on the Lamarr cluster.

```sh
# connect to the gateway of the cluster
ssh gwkilab

# from gwkilab, start a Slurm job
./srun.sh
```

## Examples and Sketches

See instructions from the following files:

- `examples/lene/README.md`
- `examples/lucas/README.md`
- `sketches/README.md`

## Profiling

The most simple thing to keep track of is the total amount of allocated GPU memory. To view this amount during execution, start your program with pre-allocation of memory being disabled and view the current allocation amount in another terminal.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false snakemake -c1

nvidia-smi -l 1 # start this command in another terminal!
```
