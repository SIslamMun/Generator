#!/bin/bash
# Build a dedicated venv for Nemotron-3 Nano 4B fine-tuning on Delta.
#
# Delta-specific strategy:
#   - On aarch64, PyPI's torch==2.7.1 is CPU-only (no CUDA wheels exist).
#   - But Delta ships a `python/miniforge3_pytorch/2.7.0` module which
#     bundles torch 2.7+cu126 + Python 3.12. We INHERIT that via
#     --system-site-packages and layer Unsloth + mamba_ssm on top.
#   - mamba_ssm needs nvcc; the `cuda/12.9` module (or 12.6) provides it.
#
# Run once on a Delta GH200 compute node (NOT login — nvcc + GPU needed):
#     cd finetuned_unsloth/models/nemotron_nano_4b
#     bash install.sh
#
# Idempotent — re-run safely if a package failed.

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Resolve python3 — must come from miniforge3_pytorch/2.7.0 module
PYTHON_SRC="$(command -v python3)"
if [ -z "$PYTHON_SRC" ]; then
    echo "ERROR: no python3. Did 'module load python/miniforge3_pytorch/2.7.0' run?"
    exit 1
fi
"$PYTHON_SRC" --version

# Verify torch + CUDA are present in the module's Python (sanity)
"$PYTHON_SRC" - <<'PY'
import torch, sys
print(f"  torch       {torch.__version__}")
print(f"  torch cuda  {torch.version.cuda}")
print(f"  cuda avail  {torch.cuda.is_available()}")
if not torch.__version__.startswith("2.7"):
    sys.exit(f"ERROR: expected torch 2.7.x from module, got {torch.__version__}")
PY

# Verify nvcc — needed by mamba_ssm build
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not on PATH. Did 'module load cuda/12.9' (or 12.6) run?"
    exit 1
fi
nvcc --version | head -4

# Resolve venv.path from config.yaml
VENV_DIR="$(awk '/^venv:/{f=1;next} f && /^[[:space:]]+path:/{gsub(/"/,""); print $2; exit}' config.yaml)"
VENV_DIR="${VENV_DIR:-.venv-nemotron}"
VENV_DIR="$HERE/$VENV_DIR"
PYBIN="$VENV_DIR/bin/python"
PIPBIN="$VENV_DIR/bin/pip"

echo "=== venv: $VENV_DIR (--system-site-packages -> inherits module torch)"

if [ ! -x "$PYBIN" ]; then
    echo "[install] python -m venv --system-site-packages $VENV_DIR"
    "$PYTHON_SRC" -m venv --system-site-packages "$VENV_DIR"
fi
"$PYBIN" --version

# Sanity: torch is visible inside the venv too?
"$PYBIN" - <<'PY'
import torch
print(f"[venv] torch {torch.__version__} cuda={torch.version.cuda}")
PY

# Upgrade pip + small bootstrap
"$PIPBIN" install --upgrade --quiet pip wheel setuptools
"$PIPBIN" install --quiet pyyaml

# Parse deps from config.yaml.venv.install_deps (excluding torch — inherited)
DEPS_FILE="$(mktemp)"
trap "rm -f $DEPS_FILE" EXIT
"$PYBIN" - <<'PY' > "$DEPS_FILE"
import yaml, re
cfg = yaml.safe_load(open("config.yaml"))
# Skip torch/triton/torchvision — provided by the module; pinning here
# would force a CPU-only reinstall on aarch64.
SKIP = re.compile(r"^(torch(\b|==|>=)|triton(\b|==|>=)|torchvision(\b|==|>=))")
for d in cfg["venv"]["install_deps"]:
    if SKIP.match(d):
        continue
    print(d)
PY

echo "=== installing deps (torch/triton/torchvision INHERITED from module):"
sed 's/^/    /' "$DEPS_FILE"

MAMBA_LINES="$(grep -E '^(mamba_ssm|causal_conv1d)' "$DEPS_FILE" || true)"
OTHER_LINES="$(grep -vE '^(mamba_ssm|causal_conv1d)' "$DEPS_FILE" || true)"

echo "=== phase 1: base packages (transformers + unsloth + trl + ...)"
echo "$OTHER_LINES" | xargs -d '\n' -r "$PIPBIN" install --upgrade

echo "=== phase 2: mamba_ssm + causal_conv1d (no build isolation, builds against module torch)"
# These compile CUDA kernels — need nvcc + module torch visible.
#
# CRITICAL on aarch64: the setup.py for both packages tries to download
# prebuilt CUDA wheels from GitHub releases (which only exist for x86_64).
# Without these env vars, an x86_64 .so ends up inside an aarch64-named
# wheel and you get `ModuleNotFoundError: causal_conv1d_cuda` at import
# time. Force source build with these flags:
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="9.0"   # GH200 = SM 9.0
echo "  CAUSAL_CONV1D_FORCE_BUILD=$CAUSAL_CONV1D_FORCE_BUILD"
echo "  MAMBA_FORCE_BUILD=$MAMBA_FORCE_BUILD"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "$MAMBA_LINES" | xargs -d '\n' -r "$PIPBIN" install --no-build-isolation --no-cache-dir

set +x
echo "=== verify (light: pkg presence only — no heavy imports) ==="
# Skip importing unsloth/mamba_ssm here — unsloth's first import on aarch64
# triggers a CUDA-detect + kernel-cache pass that has been observed to hang
# for >45 min. We'll discover any real import problem at train.py time.
"$PYBIN" - <<'PY'
import importlib.metadata as md, sys
ok = True
for pkg in ("torch", "unsloth", "trl", "transformers", "mamba_ssm", "causal_conv1d"):
    try:
        v = md.version(pkg.replace("_", "-"))
        print(f"  {pkg:18s} {v}")
    except Exception as e:
        # fall back to the underscore name
        try:
            v = md.version(pkg)
            print(f"  {pkg:18s} {v}")
        except Exception as e2:
            print(f"  {pkg:18s} FAIL: not installed ({e2})")
            ok = False
if not ok:
    sys.exit(1)
print("\n[install] venv ready (metadata-only verify — actual imports happen at train.py)")
PY
