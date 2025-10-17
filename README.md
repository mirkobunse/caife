# Redone funfolding

The goal of this repository is to provide real example use cases of unfolding where funfolding can still be improved. Matters of improvement concern the robustness and flexibility of the fits, the quality of the binning, and the adequacy of statistical and systemic uncertainties.

## Slurm setup

Use this setup if you're working on the Lamarr cluster.

```sh
# connect to the gateway of the cluster
ssh gwkilab

# from gwkilab, start a Slurm job
./srun.sh
```

## Usage

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
