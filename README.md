# MAML using JAX

Implementation of MAML on Omniglot dataset

### Setting up this repo locally

It's preferred if this is installed in virtual environment, as it will avoid any potential version conflicts.

Setting up the virtual environment:
```bash
pip install virtualenv
virtualenv maml_cc
source maml_cc/bin/activate
```

Installing dependencies:
```bash
pip install jax tensorflow tensorflow_datasets matplotlib sklearn optax
```

GPU setup:
```bash
sudo apt install nvidia-cuda-toolkit
export PATH=/usr/lib/x86_64-linux-gnu${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcuda*
```
