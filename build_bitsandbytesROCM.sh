#!/usr/bin/env bash
set -e

echo "== bitsandbytes ROCm rescue script =="

# ---------- 0. ENVIRONMENT OVERRIDES ----------
# RDNA2 default fallback (adjust if needed after rocminfo check)
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH

echo "Using HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"

# ---------------------------
# 0. FIX libxml2.so.2 (YOUR CURRENT BLOCKER)
# ---------------------------
echo "[2/7] Fixing libxml2 ABI mismatch..."

sudo apt-get install libxml2-dev libxml2-16 libxml2-utils
if [ ! -f /usr/lib/x86_64-linux-gnu/libxml2.so.2 ]; then
    echo "Creating compatibility symlink for libxml2..."
    sudo ln -sf /usr/lib/x86_64-linux-gnu/libxml2.so.16 \
                /usr/lib/x86_64-linux-gnu/libxml2.so.2 || true
fi

sudo ldconfig


# ---------- 1. CLEAN EXISTING INSTALL ----------
echo "Cleaning existing bitsandbytes..."
pip uninstall -y bitsandbytes || true
#rm -rf bitsandbytes

# ---------- 2. SYSTEM DEPENDENCIES ----------
echo "Installing build dependencies..."
#sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    python3-dev
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3-dev \
    python3-pip \
    pkg-config \
    g++ \
    gcc \
    libstdc++6 \
    zlib1g \
    libtinfo6 \
    libncurses5 || true
sudo apt-get install  libncurses6 || true

# ---------- 3. PYTORCH CHECK ----------
echo "Checking PyTorch ROCm..."
python3 - << 'EOF'
import torch
print("Torch version:", torch.__version__)
print("CUDA available (ROCm uses torch.cuda):", torch.cuda.is_available())
EOF

# ---------- 4. CLONE SOURCE ----------
echo "Cloning bitsandbytes..."
if test \! -d bitsandbytes
then
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
fi
cd bitsandbytes

# ---------- 5. CONFIGURE BUILD ----------
echo "Configuring CMake for HIP (ROCm backend)..."

# auto-detect arch fallback if not manually set
export BNB_ROCM_ARCH=$(rocminfo | grep gfx | head -n 1 | awk '{print $2}')
if [ -z "$BNB_ROCM_ARCH" ]; then
read -p "Press enter"
  echo "BNB_ROCM_ARCH not set, using generic fallback (gfx1030/gfx1100 safe guess)"
  export BNB_ROCM_ARCH="gfx1030;gfx1100;gfx90a"
fi



cmake -DCOMPUTE_BACKEND=hip \
      -DBNB_ROCM_ARCH="$BNB_ROCM_ARCH" \
            -DBNB_ROCM_VERSION=70 \
      -S .

# ---------- 6. BUILD ----------
echo "Building bitsandbytes (this may take a while)..."
make -j"4"

# ---------- 7. INSTALL ----------
echo "Installing Python package..."
pip install -e .

#TODO hacked
cd /root/AI/bitsandbytes/bitsandbytes/
ln -s libbitsandbytes_rocm72.so libbitsandbytes_rocm61.so
# ---------- 8. VERIFY ----------
echo "Verifying installation..."
python3 - << 'EOF'
import bitsandbytes as bnb
print("bitsandbytes imported successfully")

try:
    from bitsandbytes.nn import Linear4bit
    print("4-bit module OK")
except Exception as e:
    print("Warning:", e)
EOF

# ---------- 9. FINAL MESSAGE ----------
echo "DONE. If it still fails, it's almost always ROCm/GFX mismatch, not bitsandbytes."

