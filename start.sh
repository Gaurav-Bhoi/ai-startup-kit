#!/usr/bin/env bash
set -euo pipefail

: "${R2_ACCOUNT_ID:?}"
: "${R2_MODELS_BUCKET:?}"    
: "${R2_ACCESS_KEY_ID:?}"
: "${R2_SECRET_ACCESS_KEY:?}"

export AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

CACHE_DIR="/runpod-volume/models/checkpoints"
mkdir -p "${CACHE_DIR}"

download_if_missing () {
  local key="$1"
  local out="$2"
  if [ ! -f "${out}" ]; then
    echo "[startup] Downloading s3://${R2_MODELS_BUCKET}/${key}"
    aws s3 cp "s3://${R2_MODELS_BUCKET}/${key}" "${out}" --endpoint-url "${R2_ENDPOINT}"
  else
    echo "[startup] Using cached ${out}"
  fi
}

download_if_missing "checkpoints/RealVisXL_V5.0_fp16.safetensors" "${CACHE_DIR}/RealVisXL_V5.0_fp16.safetensors"
download_if_missing "checkpoints/animagine-xl-4.0-opt.safetensors" "${CACHE_DIR}/animagine-xl-4.0-opt.safetensors"

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

ln -sf "${CACHE_DIR}/RealVisXL_V5.0_fp16.safetensors" "${COMFY_MODELS_DIR}/checkpoints/RealVisXL_V5.0_fp16.safetensors"
ln -sf "${CACHE_DIR}/animagine-xl-4.0-opt.safetensors" "${COMFY_MODELS_DIR}/checkpoints/animagine-xl-4.0-opt.safetensors"

echo "[startup] Models ready in ${COMFY_MODELS_DIR}/checkpoints"

# Start ComfyUI in background
COMFY_ROOT="$(dirname "$COMFY_MODELS_DIR")"
echo "[startup] Starting ComfyUI from: $COMFY_ROOT"
python3 "$COMFY_ROOT/main.py" --listen 127.0.0.1 --port 8188 >/tmp/comfyui.log 2>&1 &

# Start RunPod serverless handler
exec python3 -u /runpod_handler.py

