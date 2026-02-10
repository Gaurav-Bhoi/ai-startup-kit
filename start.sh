#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Backblaze B2 (S3-Compatible)
# Put these in RunPod Endpoint -> Environment (use Secrets there)
# -----------------------------
: "${B2_REGION:?}"               # e.g. eu-central-003
: "${B2_MODELS_BUCKET:?}"        # your bucket name
: "${B2_ACCESS_KEY_ID:?}"        # KeyID
: "${B2_SECRET_ACCESS_KEY:?}"    # applicationKey

export AWS_ACCESS_KEY_ID="${B2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${B2_REGION}"

# B2 S3-Compatible endpoint
export S3_ENDPOINT="https://s3.${B2_REGION}.backblazeb2.com"

# -----------------------------
# Cache dirs on the RunPod network volume
# -----------------------------
export SD_CACHE_DIR="${SD_CACHE_DIR:-/runpod-volume/models/checkpoints}"
export LLM_CACHE_DIR="${LLM_CACHE_DIR:-/runpod-volume/models/llm}"
mkdir -p "${SD_CACHE_DIR}" "${LLM_CACHE_DIR}"

# -----------------------------
# Bucket prefixes (match YOUR B2 layout)
# Your screenshot shows: bucket/models/Qwen2.5-VL-32B-Instruct/...
# and SD like: bucket/models/realvis/RealVis...safetensors
# -----------------------------
export SD_PREFIX="${SD_PREFIX:-models/}"
export LLM_PREFIX="${LLM_PREFIX:-models/}"

# Optional: manifest file in bucket (recommended)
# export MODELS_MANIFEST_KEY="models/manifest.json"

# -----------------------------
# ComfyUI location + symlink checkpoints/b2 -> SD_CACHE_DIR
# -----------------------------
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

ln -sfn "${SD_CACHE_DIR}" "${COMFY_MODELS_DIR}/checkpoints/b2"
echo "[startup] Linked cache: ${COMFY_MODELS_DIR}/checkpoints/b2 -> ${SD_CACHE_DIR}"

# Tell handler where to find ComfyUI main.py (so it can start it lazily)
COMFY_ROOT="$(dirname "$COMFY_MODELS_DIR")"
export COMFY_MAIN="${COMFY_ROOT}/main.py"

# ComfyUI connection (local)
export COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFYUI_HOST}:${COMFYUI_PORT}}"
export COMFY_ARGS="${COMFY_ARGS:---disable-metadata}"

# Recommended: avoid VRAM conflicts (handler will stop comfy when loading Qwen)
export EXCLUSIVE_GPU_MODE="${EXCLUSIVE_GPU_MODE:-1}"

# Start RunPod handler (ComfyUI starts only when you call task=image)
exec python3 -u /runpod_handler.py
