#!/bin/bash
# Setup script for Raspberry Pi (CPU only)
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Install Julia packages (without CUDA)
julia --project=julia -e 'using Pkg; Pkg.add(["ArgParse","Images","FileIO","TiffImages","DeconvOptim","PointSpreadFunctions","FFTW","Colors"])'
