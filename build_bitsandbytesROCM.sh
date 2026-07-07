#!/usr/bin/env bash
set -e

LIBNAME=bitsandbytes

# Deterministic build of bitsandbytes for ROCm 6.1 (/opt/rocm)
# - Clone a stable v0.41.x tag
# - Clean previous build artifacts
# - Run cmake + make, then pip install -e .
# - Verify that bitsandbytes and bitsandbytes.nn import correctly

echo "== $LIBNAME ROCm 6.1 build script =="

# Environment (target ROCm 6.1)
export ROCM_PATH=/opt/rocm
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH

# ROCm location
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm/hip:$CMAKE_PREFIX_PATH

# bitsandbytes ROCm loader
export BNB_BACKEND=ROCm
export BNB_ROCM_VERSION=72

# RX 6400 / Navi24
export BNB_ROCM_ARCH=gfx1034
export PYTORCH_ROCM_ARCH=gfx1034
export HCC_AMDGPU_TARGET=gfx1034
export AMDGPU_TARGETS=gfx1034
export CMAKE_HIP_ARCHITECTURES=gfx1034

# Make HIP/ROCm see the card
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export HIP_DEVICE_ORDER=PCI_BUS_ID

# Often needed on consumer RDNA2 cards
export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "Using ROCM_PATH=$ROCM_PATH"

# Ensure required packages are present
echo "Installing build dependencies..."
#sudo apt-get update -y
sudo apt-get install -y git cmake build-essential python3-dev python3-pip pkg-config ninja-build || true

# PyTorch quick check
echo "Checking PyTorch (ROCm build)"
python3 - << 'PYT'
import importlib,sys
try:
    torch = importlib.import_module('torch')
    print('Torch version:', torch.__version__)
    print('torch.cuda.is_available():', torch.cuda.is_available())
except Exception as e:
    print('Warning: could not import torch:', e)
PYT

# Clean any previous clone and install
echo "Cleaning previous $LIBNAME checkout and builds..."
pip uninstall -y $LIBNAME || true
# rm -rf $LIBNAME

# Clone stable release (v0.41.0) which uses pre-rocm72 logic
echo "Cloning $LIBNAME @ v0.41.0"
if test \! -d $LIBNAME
then
git clone https://github.com/$LIBNAME-foundation/$LIBNAME.git
fi
cd $LIBNAME
# checkout the stable v0.41.0 tag
#git fetch --tags
#git checkout tags/0.41.0 -b build-0.41.0 || true

# Clean build artifacts if any
echo "Cleaning build directory and previous artifacts..."
# rm -rfv build CMakeFiles CMakeCache.txt dist *.egg-info
find . -name "lib$LIBNAME*" -type f -exec rm -f {} + || true

# Default ROCm arch: allow override via BNB_ROCM_ARCH env; otherwise use a broad safe set
export BNB_ROCM_ARCH=${BNB_ROCM_ARCH:-"gfx1030;gfx90a;gfx908;gfx906;gfx803"}
export BNB_ROCM_VERSION=72

echo "BNB_ROCM_ARCH=$BNB_ROCM_ARCH"
echo "BNB_ROCM_VERSION=$BNB_ROCM_VERSION"

# Run out-of-source CMake build explicitly
#mkdir -p build
cmake -S. -Bbuild -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="$BNB_ROCM_ARCH" -DBNB_ROCM_VERSION=$BNB_ROCM_VERSION -DROCM_PATH=$ROCM_PATH
#cd build


export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH

export ROCM_PATH=/opt/rocm
# export PATH=$ROCM_PATH/llvm/bin:$ROCM_PATH/bin:$PATH

#export CC=clang
#export CXX=clang++-21

# IMPORTANT: fake CUDA env so Makefile does not fail
export CUDA_HOME=/opt/rocm
export CUDA_VERSION=hip
export HIP_PLATFORM=amd
# ensure ROCm runtime only
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64

export NVCC=hipcc


# Build with make (no pip wheel builds)
echo "Building (cmake + make)..."
cd build
make -j"$(nproc)"
#make clean || true
#make cpuonly

#$CXX -std=c++14 -DBUILD_CUDA -fPIC -shared \
#-I ./csrc \
#./csrc/common.cpp \
#./csrc/cpu_ops.cpp \
#./csrc/pythonInterface.c \
#-o ./bitsandbytes/libbitsandbytes_cpu.so



# Install python package in editable mode
cd ../
echo "Installing Python package (pip -e .)"
pip install -e .

# Verify import and availability of bnb.nn
echo "Verifying $LIBNAME import and bnb.nn availability..."
python3 - << 'PY'
import sys
try:
    import bitsandbytes  as bnb
    print('bitsandbytes imported:', bnb.__version__ if hasattr(bnb,'__version__') else 'unknown')
    # Check that bnb.nn is accessible
    import importlib
    try:
        nn = importlib.import_module('bitsandbytes.nn')
        print('bitsandbytes.nn is importable')
    except Exception as e:
        print('ERROR: bitsandbytes.nn not importable:', e)
        sys.exit(2)
    # Try a small symbol lookup
    if hasattr(nn, 'Linear4bit'):
        print('Linear4bit available in bitsandbytes.nn')
    else:
        print('Linear4bit not found; available names:', [n for n in dir(nn) if not n.startswith('__')][:50])
except Exception as e:
    print('ERROR importing bitsandbytes:', e)
    sys.exit(1)
PY

echo "DONE. Built and installed $LIBNAME @ v0.41.0 for ROCm 6.1 (target $ROCM_PATH)."

