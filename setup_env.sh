#!/bin/sh

# Install Python3.9
sudo apt-get update
sudo apt-get install -y software-properties-common &&
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get install -y python3.9 python3.9-dev &&
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&
python3.9 get-pip.py &&
python3.9 -m pip install --upgrade pip

# Install CUDA
sudo apt-get update
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-11-7
export CUDA_HOME=/usr/local/cuda-11.7

sudo apt-get install -y libopenmpi-dev &&
sudo apt-get install -y openmpi* &&
python3.9 -m pip install mpi4py

# Set up virtual environment
virtualenv -p /usr/bin/python3.9 venv
source venv/bin/activate
python3.9 -m pip install -r requirements.txt

# Install HuggingFace Transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.27.0
python3.9 -m pip install -e .
cd ..
