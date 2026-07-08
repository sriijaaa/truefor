#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_environment.sh
#
# Run this ONCE on the RunPod pod (not locally). Installs PyTorch (CUDA 12.1),
# clones the ORIGINAL github.com/IDEA-Research/Grounded-SAM-2 repo and builds
# it (GroundingDINO's custom CUDA op + SAM2), downloads Florence-2-base via
# `transformers`, and downloads exactly the two checkpoint files needed
# (GroundingDINO SwinT-OGC + SAM 2.1 hiera-tiny) -- NOT their bundled
# download_ckpts.sh scripts, which pull several extra checkpoints we don't
# need and would waste bandwidth/disk/time on a $3 budget.
#
# MODEL SOURCING: Grounding DINO 1.0 (SwinT-OGC, Apache-2.0, no API token).
# Grounding DINO 1.5 is intentionally NOT used -- it is gated behind
# DeepDataSpace's paid API.
#
# BUILD RISK NOTE: GroundingDINO's custom CUDA attention kernel requires
# nvcc + a compatible host compiler. This script preflight-checks for nvcc
# BEFORE attempting the build, and if the build itself fails anyway, it
# retries the install with CUDA extensions disabled (FORCE_CUDA=0) so
# GroundingDINO falls back to its own built-in pure-PyTorch attention path
# (slower, same output) instead of leaving the whole setup broken. SAM2's
# CUDA extension is best-effort by default in its own setup.py -- no special
# handling needed there.
# ---------------------------------------------------------------------------
set -uo pipefail  # NOTE: no -e here -- the GroundingDINO build step needs to
                   # detect its own failure and retry rather than kill the script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${SCRIPT_DIR}/.venv"
MODELS_DIR="${SCRIPT_DIR}/models"
THIRD_PARTY_DIR="${SCRIPT_DIR}/third_party"
GSAM2_DIR="${THIRD_PARTY_DIR}/Grounded-SAM-2"
export HF_HOME="${MODELS_DIR}"
export TRANSFORMERS_CACHE="${MODELS_DIR}"

GDINO_CKPT_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
GDINO_CKPT_PATH="${MODELS_DIR}/groundingdino_swint_ogc.pth"
SAM2_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
SAM2_CKPT_PATH="${MODELS_DIR}/sam2.1_hiera_tiny.pt"

fail() {
    echo "" >&2
    echo "FATAL: $1" >&2
    echo "Aborting before spending any more budget on this run." >&2
    exit 1
}

echo "============================================================"
echo " Pico-Banana-400K / TruFor mask pipeline -- environment setup"
echo " (original Grounded-SAM-2 repo, per explicit instruction)"
echo "============================================================"

cat <<'EOF'
ESTIMATED download sizes (approximate, actual may vary by a GB or so):
  - PyTorch + torchvision (CUDA 12.1 wheels) ......... ~5.0 GB
  - transformers/accelerate/timm/opencv/etc deps ..... ~1.0 GB
  - Grounded-SAM-2 repo clone (source only) ...........~0.1 GB
  - Florence-2-base checkpoint ........................ ~0.9 GB
  - GroundingDINO SwinT-OGC checkpoint ................ ~0.7 GB
  - SAM 2.1 hiera-tiny checkpoint ..................... ~0.15 GB
  ------------------------------------------------------------
  TOTAL (estimate) .................................... ~7.9 GB

Ctrl+C now to abort. Continuing in 5 seconds...
EOF
sleep 5

echo ""
echo "[1/7] Checking for NVIDIA GPU and CUDA toolchain..."
command -v nvidia-smi &> /dev/null || fail "nvidia-smi not found. This does not look like a GPU pod."
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

HAVE_NVCC=0
if command -v nvcc &> /dev/null; then
    HAVE_NVCC=1
    echo "nvcc found: $(nvcc --version | tail -n 1)"
else
    echo "WARNING: nvcc not found on PATH. GroundingDINO's CUDA op build will be SKIPPED"
    echo "         and it will fall back to its slower pure-PyTorch attention path."
    echo "         If you want the compiled kernel, use a RunPod template with the full"
    echo "         CUDA devel/toolkit (not just the runtime), e.g. a '*-devel-*' image."
fi

echo ""
echo "[2/7] Creating virtualenv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}" || fail "venv creation failed."
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

echo ""
echo "[3/7] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 \
    || fail "PyTorch install failed."

echo ""
echo "[4/7] Installing shared pipeline dependencies..."
pip install \
    "transformers>=4.51.0" \
    accelerate timm einops huggingface_hub \
    opencv-python-headless scikit-image pandas pillow numpy tqdm matplotlib \
    || fail "Dependency install failed."

echo ""
echo "[5/7] Cloning + building the original Grounded-SAM-2 repo..."
mkdir -p "${THIRD_PARTY_DIR}"
if [[ -d "${GSAM2_DIR}/.git" ]]; then
    echo "Repo already cloned at ${GSAM2_DIR}, skipping clone."
else
    git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git "${GSAM2_DIR}" \
        || fail "git clone of Grounded-SAM-2 failed."
fi
cd "${GSAM2_DIR}" || fail "Could not cd into ${GSAM2_DIR} -- clone step above must have failed silently."

echo "  Installing SAM2 (best-effort CUDA extension, degrades gracefully on its own)..."
SAM2_BUILD_ALLOW_ERRORS=1 pip install -e . || fail "SAM2 editable install failed (this one should not normally fail even without CUDA -- check output above)."

echo "  Installing GroundingDINO (compiled CUDA op if nvcc available)..."
GDINO_BUILD_OK=0
if [[ $HAVE_NVCC -eq 1 ]]; then
    if CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(command -v nvcc)))}" \
        pip install --no-build-isolation -e grounding_dino; then
        GDINO_BUILD_OK=1
        echo "  GroundingDINO installed WITH compiled CUDA op."
    else
        echo "  WARNING: compiled build failed. Retrying with CUDA extension disabled..."
    fi
fi
if [[ $GDINO_BUILD_OK -eq 0 ]]; then
    FORCE_CUDA=0 CUDA_HOME="" pip install --no-build-isolation -e grounding_dino \
        || fail "GroundingDINO install failed even with CUDA extension disabled. Check the error above -- likely a missing system dependency."
    echo "  GroundingDINO installed WITHOUT compiled CUDA op (pure-PyTorch attention fallback, slower but functional)."
fi

cd "${SCRIPT_DIR}" || fail "Could not cd back to ${SCRIPT_DIR}."

echo ""
echo "[6/7] Downloading model checkpoints..."
mkdir -p "${MODELS_DIR}"

python3 - <<PYEOF || fail "Florence-2-base download failed."
from huggingface_hub import snapshot_download
print("  Downloading microsoft/Florence-2-base ...")
snapshot_download(repo_id="microsoft/Florence-2-base")
print("  Done: Florence-2-base")
PYEOF

if [[ -f "${GDINO_CKPT_PATH}" ]]; then
    echo "  GroundingDINO checkpoint already present, skipping download."
else
    echo "  Downloading GroundingDINO SwinT-OGC checkpoint..."
    curl -L --fail -o "${GDINO_CKPT_PATH}" "${GDINO_CKPT_URL}" \
        || fail "GroundingDINO checkpoint download failed."
fi

if [[ -f "${SAM2_CKPT_PATH}" ]]; then
    echo "  SAM 2.1 hiera-tiny checkpoint already present, skipping download."
else
    echo "  Downloading SAM 2.1 hiera-tiny checkpoint..."
    curl -L --fail -o "${SAM2_CKPT_PATH}" "${SAM2_CKPT_URL}" \
        || fail "SAM 2.1 checkpoint download failed."
fi

echo ""
echo "[7/7] Verifying everything is in place + sanity-checking CUDA..."

GDINO_CONFIG="${GSAM2_DIR}/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
[[ -f "$GDINO_CONFIG" ]] || fail "Expected GroundingDINO config not found at ${GDINO_CONFIG}. The Grounded-SAM-2 repo layout may have changed since this script was written -- check https://github.com/IDEA-Research/Grounded-SAM-2 manually and update config.py's GROUNDING_DINO_CONFIG_PATH."
[[ -f "$GDINO_CKPT_PATH" ]] || fail "GroundingDINO checkpoint missing after download step."
[[ -f "$SAM2_CKPT_PATH" ]] || fail "SAM2 checkpoint missing after download step."

SAM2_CONFIG_CHECK="${GSAM2_DIR}/sam2/sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
if [[ ! -f "$SAM2_CONFIG_CHECK" ]]; then
    echo "  WARNING: could not find sam2.1_hiera_t.yaml at the expected path (${SAM2_CONFIG_CHECK})."
    echo "           This is only a heads-up -- SAM2 resolves its config by name through Hydra,"
    echo "           not this literal path, so it may still work. If 04_generate_grounded_masks.py"
    echo "           fails to load SAM2, check the actual config location with:"
    echo "           find ${GSAM2_DIR} -name 'sam2.1_hiera_t.yaml'"
fi

python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM total (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
try:
    import groundingdino
    print('groundingdino: importable OK')
except ImportError as e:
    print('groundingdino: IMPORT FAILED --', e)
try:
    import sam2
    print('sam2: importable OK')
except ImportError as e:
    print('sam2: IMPORT FAILED --', e)
"

echo ""
echo "Disk usage after setup:"
du -sh "${MODELS_DIR}" 2>/dev/null || true
du -sh "${GSAM2_DIR}" 2>/dev/null || true
du -sh "${VENV_DIR}" 2>/dev/null || true
df -h "${SCRIPT_DIR}" | tail -n 1

echo ""
echo "============================================================"
echo " Setup complete. Activate with: source ${VENV_DIR}/bin/activate"
echo " Next: run a --dry_run pass (see README command printed by the assistant)."
echo "============================================================"
