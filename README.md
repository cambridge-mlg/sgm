# A Generative Model of Symmetry Transformations

This repository contains the experiments and code for ['A Generative Model of Symmetry Transformations' (NeurIPS 2024)](https://arxiv.org/abs/2403.01946).

## Getting Started

```bash
sudo apt-get install python3.11-venv
# ^ if not already installed
python3.11 -m venv ~/.virtualenvs/inv
source ~/.virtualenvs/inv/bin/activate
git clone --recurse-submodules git@github.com:cambridge-mlg/sgm.git
# ^ the --recurse-submodules flag is important!
cd learning-invariances
pip install --upgrade pip
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.16" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# ^ for GPU support, can be modified to get a different CUDA version
pip install orbax-checkpoint==0.4.0  --force --no-deps
# ^ really don't know why this is needed :(
pip install -e .
python3 -m ipykernel install --user --name=inv
# ^ optional, for easily running IPython/Jupyter notebooks with the virtual env.
cd src/utils/datasets/galaxy_mnist/
tfds build
# ^ for galaxy-mnist experiments
```

## Reproducing Results

Replicating the figures from the paper requires running the experiments in the `experiments` directory.
The code assumes the use of `slurm` and `wandb` for running experiments and logging results, respectively.

The following steps are required:
1. Running the inference network hyperparameter sweep:
    1. Creating the hyperparameter sweep with `python experiments/sweeps/create_inf_sweeps.py`.
    2. Launch the hyperparameter sweep jobs with `sh experiments/run_files.sh  experiments/jobs/sweep_name/`.
2. Training and saving the weights of the models using the hyperparameters found in step 1:
    1. Creating the training jobs with `experiments/sweeps/create_inf_best_sweeps.py`.
    2. Launch the training jobs with `sh experiments/run_files.sh  experiments/jobs/train_name/`.
3. Running the generative model hyperparameter sweep:
    1. Creating the hyperparameter sweep with `python experiments/sweeps/create_gen_sweeps.py`.
    2. Launch the hyperparameter sweep jobs with `sh experiments/run_files.sh  experiments/jobs/sweep_name/`.
4. Training and saving the weights of the generative models using the hyperparameters found in step 3:
    1. Creating the training jobs with `experiments/sweeps/create_gen_best_sweeps.py`.
    2. Launch the training jobs with `sh experiments/run_files.sh  experiments/jobs/train_name/`.
5. Running the VAE/AugVAE/InvVAE hyperparameter sweep(s):
    1. Creating the hyperparameter sweep(s) with `python experiments/sweeps/create_vae_augvae_sweeps.py`.
    2. Launch the hyperparameter sweep(s) jobs with `sh experiments/run_files.sh  experiments/jobs/sweep_name/`.
6. Running through either the `notebooks/sgm_figures.ipynb` or `notebooks/vae_figures.ipynb` notebooks to generate the figures.

Depending on the results being reproduced, the `create_XXX_sweps.py` scripts must be modified to specify the correct datasets, random seeds, number of training examples, etc.


