# learning-invariances
Let's try learn some invariances 

## Getting Started

```bash
module load python/3.11.0-icl
# ^ for Cambridge HPC only
sudo apt-get install python3.11-venv
# ^ if not already installed
python3.11 -m venv ~/.virtualenvs/inv
source ~/.virtualenvs/inv/bin/activate
git clone --recurse-submodules git@github.com:JamesAllingham/learning-invariances.git
# ^ the --recurse-submodules flag is important!
cd learning-invariances
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# ^ for GPU support, can be modified to get a different CUDA version
pip install -e .
python3 -m ipykernel install --user --name=inv
# ^ optional, for easily running IPython/Jupyter notebooks with the virtual env.
```

## (Maybe) useful (deleted) code, from `contrastive` branch

 - `notebooks/ssilvae_vs_vae.ipynb`: code for moodifying a dataset, see below, and various IWLB comparisons.

    ```
    def modify_dataset(dataset_iter):

      modified_dataset = []
      for batch in dataset_iter:
          modified_images = get_proto(batch["image"])
          batch['image'] = modified_images
          modified_dataset.append(batch)

      return modified_dataset

    modified_train_data = modify_dataset(input_utils.start_input_pipeline(train_ds_1_epoch, ssilvae_config.get("prefetch_to_device", 1)))

    train_dataset = tf.data.Dataset.from_generator(
        lambda: (batch for batch in modified_train_data),
        output_signature={
            'image': tf.TensorSpec(shape=(1, 500, 28, 28, 1), dtype=tf.float32),  # image shape
            'mask': tf.TensorSpec(shape=(1, 500), dtype=tf.float32)    # mask shape
        }
    ).unbatch().unbatch().filter(lambda x: x['mask'] == 1.).map(drop_mask)
    ```

  - `notebooks/ssil.ipynb`: some code for playing with color transformations.
  - `notebooks/hais.ipynb`: code for testing `tfp`'s Hamiltonian Annealed Importance Sampling.
  - `notebooks/classification.ipynb`: simple code for training a NN with `ciclo` and `clu` loops/metrics/etc.
  - `notebooks/simplified.ipynb`: code for training a self-supervised prototype predictor. Includes code for resampling the distribution over transformations to make them more 'difficult'. 
  - `notebooks/vae.ipynb`: some results for testing IWLB.
  - `experiments/train_{vae|ssilvae}.py`: example training scripts to use with `experiments/create_jobs.py`
  - `experiments/configs/vae_dspirites`: config for training a VAE with dsprites data (which has slightly different data preprocessing to MNIST)

## Some old notes

### Javi convo take-aways

  - Fix prior at loc=0 to avoid identifiability issues
  - Equivariance proof from AGW is only valid when transformation is closed (e.g. rotations in 0 to 2π)
  - Model from z -> x_hat can be a pretrained monster network from Google. Then q(z|x) just needs to learn to find the zs which produce prototypical inputs
  - q(z|x) being invariant to η is enough to make p(η) contain all of the information about the transformations. This is because we know the true poserior to be invariant, and so any q which is not invariant must be further from the true posterior, and if p(η) doesn't describe the transofrmations fully then q(z|x) must be explaining some. 
    - TODO: try to prove this using E_η{KL[q(x_hat|x)||p(x_hat|x)]} >= KL[E_η{q(x_hat|x)}||||p(x_hat|x)] (Note: E_η{p(x_hat|x)}= p(x_hat|x)).
  - q(z|x) should be partially invariant, since this solves identifiability issues. E.g. imagine a generative model of 6s and 9s and a prior on p(η) with deltas at 0 and π degrees. Then if q(z|x) is fully equivariant, we will generate both 6s and 9s from the same z. 
  - we may want a generative model in which η depends on x_hat. For example, consider gaussian convariances, here we can view σ_12 as a rotation and σ_1 and σ_2 as being shape params. However, σ_12 depends on σ_1 and σ_2. 2D ellipses might be a nice toy example? 
  - Do an experiment with rotating digits, make sure that rotating a prototype with rotations drawn from the prior and then encoding a z and then turning that into a new prototype. Hopefully new proto == orig proto.

  - Can we learn invariances without the generative model? Should be possible if we have a partially invariant η encoder???


### JAvi convo 6 Feb

  - Make prior on η depend on x rather than x_hat, and simply be rotationally invariant network? That way it doesn't depend on the quality of the z|x or xhat|x inference network? 
  - Can't model η independently since we know that the prior is not independent (i.e. to get a an x from an x)hat we could rotate or we could shear but not both?)