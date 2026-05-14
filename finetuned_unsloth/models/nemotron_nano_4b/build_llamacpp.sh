#!/bin/bash
# Build llama.cpp from source (no sudo needed) at a fixed shared path.
# Idempotent: re-running it skips the build if binaries are already present.
#
# Output: $LLAMACPP_DIR/build/bin/{llama-quantize, …} and llama.cpp's
# convert_hf_to_gguf.py at $LLAMACPP_DIR/convert_hf_to_gguf.py
#
# Run on a Delta compute node (compiler + cmake needed).

set -euo pipefail

LLAMACPP_DIR="${LLAMACPP_DIR:-/work/nvme/bekn/sislam3/llama.cpp}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"

echo "=== llama.cpp dir: $LLAMACPP_DIR"

if [ -x "$LLAMACPP_DIR/build/bin/llama-quantize" ] && [ -f "$LLAMACPP_DIR/convert_hf_to_gguf.py" ]; then
    echo "[llamacpp] already built — skipping"
    "$LLAMACPP_DIR/build/bin/llama-quantize" --version 2>&1 | head -2 || true
    exit 0
fi

mkdir -p "$(dirname "$LLAMACPP_DIR")"
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
    echo "[llamacpp] cloning $LLAMACPP_REPO"
    git clone --depth 1 "$LLAMACPP_REPO" "$LLAMACPP_DIR"
else
    echo "[llamacpp] already cloned"
fi

cd "$LLAMACPP_DIR"

# Build directory
BUILD=build
mkdir -p $BUILD

# CPU-only build is fine for the quantize tool (small CPU workload).
# Skip CUDA to avoid the heavy CUDA-compile pass — quantize doesn't need GPU.
echo "[llamacpp] cmake configure (CPU-only)"
cmake -B $BUILD \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF

echo "[llamacpp] cmake build (this takes ~5 min)"
# Only build llama-quantize — that's all we need for HF → GGUF → quantized.
# `llama-cli` got restructured/renamed in recent main; not needed.
cmake --build $BUILD --config Release -j "$(nproc)" --target llama-quantize

echo ""
echo "[llamacpp] built artifacts:"
ls -la $BUILD/bin/llama-quantize 2>&1
ls -la $LLAMACPP_DIR/convert_hf_to_gguf.py 2>&1

# install python deps for convert_hf_to_gguf.py (uses pip — into the active venv)
echo ""
echo "[llamacpp] installing convert_hf_to_gguf.py python deps"
pip install --quiet -r $LLAMACPP_DIR/requirements/requirements-convert_hf_to_gguf.txt 2>&1 || \
    pip install --quiet gguf "protobuf>=4.21.0" sentencepiece 2>&1 || true

echo ""; echo "[llamacpp] ready."
