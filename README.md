# deconv_gui

This repository contains a small GUI written in **Python** that delegates the
deconvolution work to a **Julia** script.

## Installation on PC

In powershell
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
install julia from https://julialang.org/install/
```
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```
Install the Julia dependencies without CUDA:


## Installation on Raspberry Pi

Create a Python virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install julia by

```
curl -fsSL https://install.julialang.org | sh
```


```bash
julia --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```



## Usage

Start the GUI with:

```bash
python3 deconvolution_gui.py [--nogpu]
```

The `--nogpu` flag hides the GPU option in the interface.

To inspect or modify the Julia environment run:

```bash
julia --project=julia -e 'using Pkg; Pkg.status()'
```

## Running tests

There are currently no automated tests. After adding code, tests should be placed
in a `tests/` directory and executed with:

```bash
pytest
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for
details.

## Contributing

Please open an issue or pull request if you intend to contribute code or
documentation.
