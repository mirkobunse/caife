# Redone funfolding

Even more fun in unfolding with funfolding.

## Slurm setup

Use this setup if you're working on the Lamarr cluster

```sh
# start a Slurm job from gwkilab
./srun.sh

# inside the job, install the project with its dependencies
pip install -e .
```

## Usage

ToDo.

## Profiling

The most simple thing to keep track of is the total amount of allocated GPU memory. To view this amount during execution, start your program with pre-allocation of memory being disabled and view the current allocation amount in another terminal.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false snakemake -c1

nvidia-smi -l 1 # start this command in another terminal!
```
