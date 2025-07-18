# deconv_gui

This repository contains a small GUI written in **Python** that delegates the
deconvolution work to a **Julia** script. Two setup modes are provided:

1. **Raspberry Pi (CPU only)** – lightweight environment without GPU support.
2. **PC with optional GPU** – installs CUDA.jl so that the Julia backend can use
   a CUDA capable device when available.

## Environment setup

Use the helper scripts in `scripts/` to prepare **both the Python and Julia
environments**. For example on a Raspberry Pi run:

```bash
./scripts/setup_rpi.sh
```

On a PC run:

```bash
./scripts/setup_pc.sh
```

Both scripts create a Python virtual environment in `.venv` and install the
required packages from `requirements.txt`.  In addition they set up a Julia
project in the `julia/` directory and install the necessary Julia packages
(the PC version also adds `CUDA.jl` for GPU acceleration).

After setup you can start the GUI using the matching helper script:

```bash
./scripts/run_rpi.sh   # Raspberry Pi
./scripts/run_pc.sh    # PC
```

To inspect or modify the Julia environment manually run:

```bash
julia --project=julia -e 'using Pkg; Pkg.status()'
```

## Running tests

There are currently no automated tests in this repository. After adding code, tests should be placed in a `tests/` directory and executed with:

```bash
pytest
```

## Contributing

Please open an issue or pull request if you intend to contribute code or documentation.
