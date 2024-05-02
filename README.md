# learning-invariances
Let's try learn some invariances 

## Getting Started

```bash
sudo apt-get install python3.11-venv
# ^ if not already installed
python3.11 -m venv ~/.virtualenvs/inv  # replace python3.11 with ~/.localpython/bin/python3.11 if necessary
source ~/.virtualenvs/inv/bin/activate
git clone --recurse-submodules git@github.com:JamesAllingham/learning-invariances.git
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

If Python 3.11 needs to be installed without apt-get or similar, e.g. on Cambridge HPC.
  
```bash
# install openssl (if missing)
wget https://www.openssl.org/source/openssl-1.1.1w.tar.gz
md5sum openssl-1.1.1w.tar.gz  # should be 3f76825f195e52d4b10c70040681a275
tar -xzvf openssl-1.1.1w.tar.gz
cd openssl-1.1.1w/
./config --prefix=$HOME/.openssl
make
make install
~/.openssl/bin/openssl version  # should be OpenSSL 1.1.1w  11 Sep 2023
# install python
export PYTHON_VERSION=3.11.5
export PYTHON_MAJOR=3
export LD_LIBRARY_PATH="$HOME/.openssl/lib:$LD_LIBRARY_PATH"
export CPPFLAGS="-I$HOME/.openssl/include $CPPFLAGS"
curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
md5sum Python-3.11.5.tgz  # should be b628f21aae5e2c3006a12380905bb640
tar -xvzf Python-${PYTHON_VERSION}.tgz
cd Python-${PYTHON_VERSION}
./configure \
    --prefix=$HOME/.localpython \
    --enable-optimizations \
    --enable-ipv6 \
    LDFLAGS=-Wl,-rpath=/opt/python/${PYTHON_VERSION}/lib,--disable-new-dtags \
    --with-openssl=$HOME/.openssl
make
make install
# install pip (might not be necessary)
~/.localpython/bin/python3 -c "import ssl; print(ssl.OPENSSL_VERSION)"  # should be OpenSSL 1.1.1w  11 Sep 2023
curl -O https://bootstrap.pypa.io/get-pip.py
~/.localpython/bin/pip3 install numpy  # should succeed
```

## TODOs:

  [ ] Implement and test the correct η-gradient loss term, which regularises the p(η|x_hat) densities against small pertubations on x_hat (i.e., T(x_hat, η_small)), rather than encouraging smoothness of the output as we are currently doing. See https://github.com/JamesAllingham/learning-invariances/pull/5#discussion_r1322016289.
  ```
  x_hat = jax.lax.stop_gradient(transform_image(x, η_rand - η_x_rand))
            def get_log_p_η_x_hat(η_temp):
                x_hat = transform_image(x_hat, η_temp)
                return model.apply(
                    {"params": params},
                    x_hat,
                    η_x,
                    train=train,
                    method=model.generative_net_ll,
                )

            log_p_η_x_hat, η_grad = jax.value_and_grad(get_log_p_η_x_hat)(
                jnp.zeros_like(η_rand)
            )
            η_grad_regularizer = jnp.abs(η_grad).mean()
  ```
  [ ] Revist the shearing transformation for MNIST. It is currently disabled, but it would be nice to show that we can still learn with it.

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
  - `experiments/configs/vae_dsprites`: config for training a VAE with dsprites data (which has slightly different data preprocessing to MNIST)

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




def make_paper_plot(idxs):
    # create a plot with 3 rows and 4 * len(idxs) columns
    top_row = []
    mid_row = []
    bot_row = []

    p_H_X_hat, p_H_X_hat_vars = model.bind({"params": final_state.params}).generative_net.unbind()

    for i, idx in enumerate(idxs):
        rng_local = random.fold_in(rng, i)
        x_ = val_batch['image'][0][idx]

        transformed_xs = jax.vmap(transform_image, in_axes=(None, 0))(
            x_,
            jnp.linspace(-jnp.array(config.model.bounds) * 0.5, jnp.array(config.model.bounds) * 0.5, 4)
        )
        top_row.append(transformed_xs)

        # row 2 is the prototypes
        xhats, ηs = get_proto(transformed_xs)
        mid_row.append(xhats)

        # row 3 is samples from the learned distribution p_H_X_hat
        p_H_x_hat = p_H_X_hat.apply(p_H_X_hat_vars, xhat)
        ηs = p_H_x_hat.sample(seed=rng_local, sample_shape=(4,))
        x_samples = jax.vmap(transform_image, in_axes=(0, 0))(xhats, ηs)
        bot_row.append(x_samples)

    top_row = jnp.concatenate(top_row, axis=0)
    mid_row = jnp.concatenate(mid_row, axis=0)
    bot_row = jnp.concatenate(bot_row, axis=0)

    all_rows = jnp.concatenate([top_row, mid_row, bot_row], axis=0)
    print(all_rows.shape)
    plot_img_array(all_rows, ncol=4*len(idxs), pad_value=255)


make_paper_plot([18, 22, 24, 27])
make_paper_plot([14, 15, 4, 10, 9])




def resample(self, x, train: bool = False):
        q_H_x = self.inference_net(x, train=train)
        η = q_H_x.sample(seed=self.make_rng("sample"))

        x_hat = transform_image(x, -η)

        p_H_x_hat = self.generative_net(x_hat, train=train)
        η_new = p_H_x_hat.sample(seed=self.make_rng("sample"))

        x_recon = transform_image(x, -η + η_new)
        return x_recon