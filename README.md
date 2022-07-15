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

CUDA setup:
```bash
sudo apt install nvidia-cuda-toolkit
```
Check if installation is complete `nvcc -V`:
```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
CUDNN setup:
Download **cuDNN 7.6.5** from this [link](https://developer.nvidia.com/rdp/form/cudnn-download-survey)
```bash
tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
sudo chmod a+r /usr/lib/cuda/include/cudnn.h /usr/lib/cuda/lib64/libcudnn*
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
```
Tensorflow setup:
```bash
pip install tensorflow==2.2.0
```
Check the installation by:
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Output should look like this :`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
