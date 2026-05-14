#!/bin/bash
# Build a dedicated uv-managed venv for Nemotron-3 Nano 4B fine-tuning.
# Heavy + version-pinned (mamba_ssm needs torch==2.7.1), so we keep it
# separate from the generator's .venv-delta.
#
# Run this once on a Delta compute node (NOT the login node — mamba_ssm
# needs nvcc to build):
#     cd finetuned_unsloth/models/nemotron_nano_4b
#     bash install.sh
#
# Idempotent — re-run safely if a package failed.

set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# uv lives in $HOME/.local/bin on Delta; fall back to system PATH otherwise.
if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi
echo "=== uv: $(uv --version)"

# Resolve venv.path from config.yaml without pulling pyyaml first
VENV_DIR="$(awk '/^venv:/{f=1;next} f && /^[[:space:]]+path:/{gsub(/"/,""); print $2; exit}' config.yaml)"
VENV_DIR="${VENV_DIR:-.venv-nemotron}"
VENV_DIR="$HERE/$VENV_DIR"
PYBIN="$VENV_DIR/bin/python"

echo "=== venv: $VENV_DIR"

# Create venv (uv venv is idempotent — re-creates only if missing/broken).
if [ ! -x "$PYBIN" ]; then
    echo "[install] uv venv $VENV_DIR"
    uv venv "$VENV_DIR" --python python3
fi
echo "=== python: $($PYBIN --version)"

# pyyaml only needed by this script to read config.yaml — install into the
# venv so the rest of the install reads config.yaml inside the same env.
uv pip install --python "$PYBIN" --quiet pyyaml

# Parse deps from config.yaml.venv.install_deps
DEPS_FILE="$(mktemp)"
trap "rm -f $DEPS_FILE" EXIT
"$PYBIN" - <<'PY' > "$DEPS_FILE"
import yaml
cfg = yaml.safe_load(open("config.yaml"))
for d in cfg["venv"]["install_deps"]:
    print(d)
PY

echo ""; echo "=== installing deps from config.yaml.venv.install_deps:"
sed 's/^/    /' "$DEPS_FILE"

# Two-phase install:
#   1. Base packages — torch first, then everything except mamba_ssm/causal_conv1d.
#      mamba_ssm's setup.py imports torch at build time, so torch MUST be present.
#   2. mamba_ssm + causal_conv1d with --no-build-isolation so they pick up the
#      already-installed torch (and don't try to fetch a different one).
MAMBA_LINES="$(grep -E '^(mamba_ssm|causal_conv1d)' $DEPS_FILE || true)"
OTHER_LINES="$(grep -vE '^(mamba_ssm|causal_conv1d)' $DEPS_FILE || true)"

echo ""; echo "=== phase 1: base packages ==="
# shellcheck disable=SC2086
echo "$OTHER_LINES" | xargs -d '\n' -r uv pip install --python "$PYBIN" --upgrade -q

echo ""; echo "=== phase 2: mamba_ssm + causal_conv1d (no build isolation) ==="
# shellcheck disable=SC2086
echo "$MAMBA_LINES" | xargs -d '\n' -r uv pip install --python "$PYBIN" --no-build-isolation -q

# Verify
echo ""; echo "=== verify ==="
"$PYBIN" - <<'PY'
import importlib, sys
for mod in ("torch", "unsloth", "trl", "transformers", "mamba_ssm", "causal_conv1d"):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod:18s} {v}")
    except Exception as e:
        print(f"  {mod:18s} FAIL: {e.__class__.__name__}: {e}")
        sys.exit(1)
print("\n[install] venv ready: $VENV_DIR")
PY
