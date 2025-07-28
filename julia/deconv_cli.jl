#!/usr/bin/env julia
# deconv_cli.jl — called from the Python GUI
# recieves parameters from the python GUI subprocess, read by command line arguments
# and performs 3-D deconvolution using the Richardson-Lucy algorithm. 
# Usage example:
#   julia --project=. deconv_cli.jl --roi 100:612,200:712 --zsize 64 \
#         --iter 20 --gpu false --sigma_z 15 input.tif output.tif

using ArgParse
using Images, FileIO, TiffImages
using DeconvOptim           # brings in the iterative deconvolution routines
using DeconvOptim: center_set!
using PointSpreadFunctions
using FFTW, CUDA, Colors, Printf

# ---------------------------- helper -----------------------------
"""Crop a 3-D array to (x0:x1, y0:y1, central z-slice of length `zsize`)."""
function crop_xy_z(arr, x0, x1, y0, y1, zsize)
    nx, ny, nz = size(arr)
    # Check bounds
    @assert zsize ≤ nz "zsize larger than stack depth"
    z0 = (nz - zsize) ÷ 2 + 1
    copy(@view arr[x0:x1, y0:y1, z0:z0+zsize-1])
end

# ---------------------------- CLI --------------------------------
p = ArgParseSettings()
@add_arg_table p begin
    "--roi"     ; default = ""                 # x0:x1,y0:y1  (1-based indexing)
    "--zsize"   ; arg_type = Int     ; default = 64
    "--iter"    ; arg_type = Int     ; default = 20
    "--gpu"     ; arg_type = Bool    ; default = false
    "--sigma_z" ; arg_type = Float64 ; default = 15.0
    "input"                                   # input TIFF file
    "output"                                  # output TIFF file
end
args = parse_args(p)

# --------------------------- read stack --------------------------
raw   = load(args["input"])
img3d = permutedims(Float32.(raw), (1,2,3))   # (X, Y, Z) ordering

# Parse the ROI
if args["roi"] != ""
    parts = split(args["roi"], ',')
    x0, x1 = parse.(Int, split(parts[1], ':'))
    y0, y1 = parse.(Int, split(parts[2], ':'))
else
    nx, ny, _ = size(img3d)
    x0, x1, y0, y1 = 1, nx, 1, ny
end
roi     = crop_xy_z(img3d, x0, x1, y0, y1, args["zsize"])
roi_sz  = size(roi)

# --------------------------- GPU? --------------------------------
use_gpu = args["gpu"] && CUDA.has_cuda()
roi     = use_gpu ? CuArray(roi) : roi

# --------------------------- PSF ---------------------------------
psf_small = psf(ModeLightsheet, (16,16,16),
                PSFParams(0.525, 0.25, 1.33);
                sampling = (0.5, 0.5, 6), sigma_z = args["sigma_z"])
psf_small ./= sum(psf_small)

psf_roi = zeros(Float32, roi_sz)
center_set!(psf_roi, psf_small)
psf_roi = ifftshift(psf_roi, 1:3)
psf_roi = use_gpu ? CuArray(psf_roi) : psf_roi

# ----------------------- RL iterative ----------------------------
@info "Starting RL" iterations = args["iter"] gpu = use_gpu
# reset progress bar on the Python side
println("PROGRESS 0")          
deconv = richardson_lucy_iterative(
    roi, psf_roi;
    iterations  = args["iter"],
    λ           = 0.01,
    conv_dims   = 1:3,
    regularizer = nothing,
)
# tell the Python side we are done
println("PROGRESS 100")        

# ----------------------- save TIFF -------------------------------
res  = Array(deconv)
norm = res ./ maximum(res)
gray = Gray{N0f16}.(norm)
timg = TiffImages.DenseTaggedImage(gray)

first(timg.ifds)[TiffImages.IMAGEDESCRIPTION] =
    "RL deconv $(join(roi_sz, '×')) iter $(args["iter"]) GPU $(use_gpu)"

TiffImages.save(args["output"], timg)
println("✓ Finished → ", args["output"])
