#!/bin/bash

# Installation script for MDCath preprocessing dependencies

echo "Installing MDCath preprocessing dependencies..."

# Update pip first
pip install --upgrade pip

# Install core dependencies
echo "Installing core dependencies..."
pip install mdtraj torch pyemma numpy tqdm

# Install optional dependencies
echo "Installing optional dependencies..."
pip install matplotlib scipy

echo "Installation complete!"

# Test imports
python3 -c "
import mdtraj
import torch  
import pyemma
import numpy
import tqdm
print('All dependencies installed successfully!')
print(f'MDTraj version: {mdtraj.__version__}')
print(f'PyTorch version: {torch.__version__}')
print(f'PyEMMA version: {pyemma.__version__}')
print(f'NumPy version: {numpy.__version__}')
"