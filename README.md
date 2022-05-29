# learning-invariances
Let's try learn some invariances 

## Getting Started

```bash
module load python-3.9.6-gcc-5.4.0-sbr552h
# ^ for Cambridge HPC only
sudo apt-get install python3.9-venv
# ^ if not already installed
python3.9 -m venv ~/.virtualenvs/inv
source ~/.virtualenvs/inv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install 'jax[cuda11_cudnn805]' -f https://storage.googleapis.com/jax-releases/jax_releases.html
# Run this ^ if you want GPU support. Note however, that this requires CUDA 11.1 and CUDNN 8.05, for older versions try
# pip install --upgrade jax jaxlib==0.1.69+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html 
# where you can replace 'cuda101' with the appropriate string for your CUDA version. E.g. for CUDA 10.2 use cuda102.
# Note, however, that JAX is dropping support for CUDA versions lower than 11.1.
pip install -e .
python3.9 -m ipykernel install --user --name=inv
# ^ optional, for easily running IPython/Jupyter notebooks with the virtual env.
```

## TODO:

 - [ ] Setup expiriment sweep code
 - [ ] Run sweep over different augmentations in MNIST and see if LIVAE learns the mean
 - [ ] Fix z_sample rng
 - [ ] Run experiments for inv-VAEs with rotations in the data. We expect that in this case the inv encoders will perform better than non-inv ones. 
 - [ ] Make CNN implementation match that of [Dubois et al.](https://github.com/YannDubs/lossyless/blob/462af23a52d68f860e5ae2ff9c59f04cfb8c5fd5/lossyless/architectures.py#L235). That is, resize images to be of power 2, then have the CNN downsample in h & w dims at each stage. Use [this](https://jax.readthedocs.io/en/latest/_autosummary/jax.image.resize.html) rather than [this](https://pytorch.org/vision/main/generated/torchvision.transforms.Resize.html).
 - [ ] Try reproduce reconstruction behaviour of Dubois et al. That is, we want reconstructions to not show the symmetry to which they are supposed to be invariant. Perhaps we need more of a bottle neck.
 - [ ] Figure out why the transformed samples for the inv-VAE look less noisy than the original samples.