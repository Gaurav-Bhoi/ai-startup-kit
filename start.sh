#!/usr/bin/env bash
set -euo pipefail

# Backblaze B2 (S3-Compatible)
: "${B2_REGION:?}"               # e.g. eu-central-003, us-west-002, etc.
: "${B2_MODELS_BUCKET:?}"        # your bucket name
: "${B2_ACCESS_KEY_ID:?}"        # KeyID
: "${B2_SECRET_ACCESS_KEY:?}"    # applicationKey

export AWS_ACCESS_KEY_ID="${B2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${B2_REGION}"

# B2 S3-Compatible endpoint
export S3_ENDPOINT="https://s3.${B2_REGION}.backblazeb2.com"

# ComfyUI connection (local)
export COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFYUI_HOST}:${COMFYUI_PORT}}"

# Optional: limit cache size on the network volume (0 = unlimited)
export MAX_CACHE_GB="${MAX_CACHE_GB:-0}"

# Persistent cache on RunPod volume
export CACHE_DIR="${CACHE_DIR:-/runpod-volume/models/checkpoints}"
mkdir -p "${CACHE_DIR}"

# Locate ComfyUI models/checkpoints folder in the image
COMFY_MODELS_DIR=""
for p in /comfyui/ComfyUI/models /comfyui/models /workspace/ComfyUI/models /ComfyUI/models; do
  if [ -d "${p}/checkpoints" ]; then
    COMFY_MODELS_DIR="${p}"
    break
  fi
done

if [ -z "${COMFY_MODELS_DIR}" ]; then
  echo "[startup] ERROR: Can't find ComfyUI models/checkpoints folder in the container."
  exit 1
fi

# Single symlink: make the whole cache appear in ComfyUI as "b2/<file>"
ln -sfn "${CACHE_DIR}" "${COMFY_MODELS_DIR}/checkpoints/b2"
echo "[startup] Linked cache: ${COMFY_MODELS_DIR}/checkpoints/b2 -> ${CACHE_DIR}"

# Start ComfyUI in background
COMFY_ROOT="$(dirname "$COMFY_MODELS_DIR")"
echo "[startup] Starting ComfyUI from: $COMFY_ROOT"
COMFY_ARGS="${COMFY_ARGS:---disable-metadata}"
python3 "$COMFY_ROOT/main.py" --listen "${COMFYUI_HOST}" --port "${COMFYUI_PORT}" ${COMFY_ARGS} >/tmp/comfyui.log 2>&1 &

# Start RunPod serverless handler
exec python3 -u /runpod_handler.py
