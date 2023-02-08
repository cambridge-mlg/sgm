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
 - [ ] Switch to simpler per-batch logging.


 ## Javi convo take-aways

  - Fix prior at loc=0 to avoid identifiability issues
  - Equivariance proof from AGW is only valid when transformation is closed (e.g. rotations in 0 to 2π)
  - Model from z -> x_hat can be a pretrained monster network from Google. Then q(z|x) just needs to learn to find the zs which produce prototypical inputs
  - q(z|x) being invariant to η is enough to make p(η) contain all of the information about the transformations. This is because we know the true poserior to be invariant, and so any q which is not invariant must be further from the true posterior, and if p(η) doesn't describe the transofrmations fully then q(z|x) must be explaining some. 
    - TODO: try to prove this using E_η{KL[q(x_hat|x)||p(x_hat|x)]} >= KL[E_η{q(x_hat|x)}||||p(x_hat|x)] (Note: E_η{p(x_hat|x)}= p(x_hat|x)).
  - q(z|x) should be partially invariant, since this solves identifiability issues. E.g. imagine a generative model of 6s and 9s and a prior on p(η) with deltas at 0 and π degrees. Then if q(z|x) is fully equivariant, we will generate both 6s and 9s from the same z. 
  - we may want a generative model in which η depends on x_hat. For example, consider gaussian convariances, here we can view σ_12 as a rotation and σ_1 and σ_2 as being shape params. However, σ_12 depends on σ_1 and σ_2. 2D ellipses might be a nice toy example? 
  - Do an experiment with rotating digits, make sure that rotating a prototype with rotations drawn from the prior and then encoding a z and then turning that into a new prototype. Hopefully new proto == orig proto.

  - Can we learn invariances without the generative model? Should be possible if we have a partially invariant η encoder???


  ## JAvi convo 6 Feb

    - Make prior on η depend on x rather than x_hat, and simply be rotationally invariant network? That way it doesn't depend on the quality of the z|x or xhat|x inference network? 
    - Can't model η independently since we know that the prior is not independent (i.e. to get a an x from an x)hat we could rotate or we could shear but not both?)