#!/usr/bin/env bash
set -euo pipefail

# Backblaze B2 (S3-Compatible)
: "${B2_REGION:?}"
: "${B2_MODELS_BUCKET:?}"
: "${B2_ACCESS_KEY_ID:?}"
: "${B2_SECRET_ACCESS_KEY:?}"

export AWS_ACCESS_KEY_ID="${B2_ACCESS_KEY_ID}"
export AWS_SECRET_ACCESS_KEY="${B2_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${B2_REGION}"
export AWS_EC2_METADATA_DISABLED=true

# B2 S3-Compatible endpoint (handler reads S3_ENDPOINT)
export S3_ENDPOINT="https://s3.${B2_REGION}.backblazeb2.com"
export B2_ENDPOINT="${S3_ENDPOINT}"           # compatibility
export B2_MODELS_BUCKET="${B2_MODELS_BUCKET}" # handler reads this

# Prefix defaults (match your current layout)
export SD_PREFIX="${SD_PREFIX:-models/}"
export LLM_PREFIX="${LLM_PREFIX:-models/}"

# 24GB safety
export EXCLUSIVE_GPU_MODE="${EXCLUSIVE_GPU_MODE:-1}"
export QWEN_LOAD_4BIT="${QWEN_LOAD_4BIT:-1}"
export QWEN_DEFAULT_MAX_NEW_TOKENS="${QWEN_DEFAULT_MAX_NEW_TOKENS:-256}"
export QWEN_MAX_GPU_GIB="${QWEN_MAX_GPU_GIB:-22GiB}"
export QWEN_MAX_CPU_GIB="${QWEN_MAX_CPU_GIB:-64GiB}"

# Cache dirs on RunPod volume
export SD_CACHE_DIR="${SD_CACHE_DIR:-/runpod-volume/models/checkpoints}"
export LLM_CACHE_DIR="${LLM_CACHE_DIR:-/runpod-volume/models/llm}"
mkdir -p "${SD_CACHE_DIR}" "${LLM_CACHE_DIR}"

# ComfyUI config
export COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export COMFY_URL="${COMFY_URL:-http://${COMFYUI_HOST}:${COMFYUI_PORT}}"
export COMFY_ARGS="${COMFY_ARGS:---disable-metadata}"
export COMFY_PID_FILE="${COMFY_PID_FILE:-/tmp/comfyui.pid}"
export COMFY_LOG="${COMFY_LOG:-/tmp/comfyui.log}"
export COMFY_CHECKPOINT_SUBDIR="${COMFY_CHECKPOINT_SUBDIR:-b2}"

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

# Symlink: make SD cache appear in ComfyUI as "b2/<path>"
ln -sfn "${SD_CACHE_DIR}" "${COMFY_MODELS_DIR}/checkpoints/${COMFY_CHECKPOINT_SUBDIR}"
echo "[startup] Linked cache: ${COMFY_MODELS_DIR}/checkpoints/${COMFY_CHECKPOINT_SUBDIR} -> ${SD_CACHE_DIR}"

# Tell handler where ComfyUI main.py is
COMFY_ROOT="$(dirname "$COMFY_MODELS_DIR")"
export COMFY_MAIN="${COMFY_ROOT}/main.py"

# Start ComfyUI now (faster first image). Handler can stop/start it for Qwen.
python3 "$COMFY_MAIN" --listen "${COMFYUI_HOST}" --port "${COMFYUI_PORT}" ${COMFY_ARGS} >>"${COMFY_LOG}" 2>&1 &
echo $! > "${COMFY_PID_FILE}"

# Start RunPod serverless handler
exec python3 -u /runpod_handler.py
