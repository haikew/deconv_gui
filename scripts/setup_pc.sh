#!/bin/bash
# Setup script for PC with optional GPU support
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Install Julia packages including CUDA for GPU acceleration
julia --project=julia -e 'using Pkg; Pkg.add(["ArgParse","Images","FileIO","TiffImages","DeconvOptim","PointSpreadFunctions","FFTW","CUDA","Colors"])'
