#!/bin/bash

# Disable command echoing
set +v

# Create and activate a conda environment named UrbanEV
conda create -n UrbanEV_v1 python=3.8 -y
source activate UrbanEV_v1

# Install PyTorch and its dependencies
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install torch_geometric-temporal
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
pip install torch_geometric
pip install torch-geometric-temporal

# Install additional libraries
pip install statsmodels
pip install scikit-learn
pip install openpyxl


echo "Setup completed for UrbanEV environment."

# Prevent the script from closing immediately (pause equivalent)
read -p "Press [Enter] key to continue..."