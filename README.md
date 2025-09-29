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

For both examples, get a copy of the [dnn_selections](https://github.com/icecube/dnn_selections) repository, e.g., as a zip archive.

### Stopping muon neutrinos (Lucas' example)

The use case of stopping muon neutrinos is given in `examples/lucas/unfolding_muons.ipynb`.

First, copy the data to your Lamarr cluster home. SSH to your slurm job, *with a forwarded SSH agent*, such that you can access the `vollmond` node from within your slurm job (don't attempt going via the tmux session, as it does not forward your agent). Execute the following:

```sh
scp vollmond:/cephfs/users/lwitthaus/data/high_level/23111.pkl ~/data/re-funfolding/
```

Second, install Conda at `/opt/miniconda3` from within your slurm job, indicating this installation directory during the interactive installation script. Also, **do not let the installer modify your shell profile.**

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_25.7.0-2-Linux-x86_64.sh
bash Miniconda3-py310_25.7.0-2-Linux-x86_64.sh # install at /opt/miniconda3
```

Third, instantiate the Conda environment.

```sh
source /opt/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate stmuons
pip install --upgrade pip setuptools wheel
pip install -e /path/to/dnn_selections git+ssh://git@github.com/lwitthaus/funalysis.git@b49c57440d156d553bcfe984668b59aa6042c91a#egg=funalysis git+ssh://git@github.com/icecube/ic3-labels.git@c4d53f722b1be0add69879740307dadce184c4b0#egg=ic3_labels git+ssh://git@github.com/lwitthaus/stoppingmuons.git@0f58008afbcac11212ead0c5b554fd4040b50055#egg=stoppingmuons
pip install notebook
```

Finally, start a Jupyter server:

```sh
jupyter notebook --ip="*"
```

### MC electron neutrinos (Lene's example)

The use case of MC electron neutrinos is given in `examples/lene/MC_Unfolding_NuE_[5e2,1.3e4,11]_[0,180].ipynb`.

First, copy the data to your Lamarr cluster home. SSH to your slurm job, *with a forwarded SSH agent*, such that you can access the `vollmond` node from within your slurm job (don't attempt going via the tmux session, as it does not forward your agent). Execute the following:

```sh
scp "vollmond:/cephfs/users/lrootsel/NuGen_datasets_complete_SnowStormParameters_214**_old_numpy.pkl" ~/data/re-funfolding/
```

Second, instantiate the virtual environment from within your slurm job.

```sh
cd examples/lene/
python -m venv --system-site-packages venv
venv/bin/pip install --upgrade pip setuptools wheel
venv/bin/pip install -r requirements.txt
```

Finally, start a Jupyter server:

```sh
venv/bin/jupyter notebook --ip="*"
```

Open the Jupyter notebook from your local machine after forwarding the server's SSH port.

## Profiling

The most simple thing to keep track of is the total amount of allocated GPU memory. To view this amount during execution, start your program with pre-allocation of memory being disabled and view the current allocation amount in another terminal.

```sh
XLA_PYTHON_CLIENT_PREALLOCATE=false snakemake -c1

nvidia-smi -l 1 # start this command in another terminal!
```
