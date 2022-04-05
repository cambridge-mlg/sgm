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