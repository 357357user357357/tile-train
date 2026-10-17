#!/usr/bin/env bash
# Source this for tch/libtorch builds & runs:
#   source env.sh && cargo run --release --example spike_mlp
export TORCH_DIR="$HOME/miniforge3/envs/tilelang/lib/python3.12/site-packages/torch"
export LIBTORCH="$TORCH_DIR"
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH_CXX11_ABI=0  # conda-forge torch = legacy C++ ABI
NVIDIA_LIBS="$(find "$HOME/miniforge3/envs/tilelang/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${TORCH_DIR}/lib:${LD_LIBRARY_PATH:-}"
