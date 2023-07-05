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
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# ^ for GPU support, can be modified to get a different CUDA version
pip install -e .
python3 -m ipykernel install --user --name=inv
# ^ optional, for easily running IPython/Jupyter notebooks with the virtual env.
```


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