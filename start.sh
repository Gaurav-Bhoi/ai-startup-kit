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

# B2 S3-Compatible endpoint
export S3_ENDPOINT="https://s3.${B2_REGION}.backblazeb2.com"
export B2_ENDPOINT="${S3_ENDPOINT}"
export B2_MODELS_BUCKET="${B2_MODELS_BUCKET}"

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

# ComfyUI runtime vars
export COMFYUI_PORT="${COMFYUI_PORT:-8188}"
export COMFY_LISTEN_HOST="${COMFY_LISTEN_HOST:-0.0.0.0}"   # bind here
export COMFY_CONNECT_HOST="${COMFY_CONNECT_HOST:-127.0.0.1}" # handler connects here
export COMFY_URL="${COMFY_URL:-http://${COMFY_CONNECT_HOST}:${COMFYUI_PORT}}"
export COMFY_ARGS="${COMFY_ARGS:---disable-metadata}"
export COMFY_PID_FILE="${COMFY_PID_FILE:-/tmp/comfyui.pid}"
export COMFY_LOG="${COMFY_LOG:-/tmp/comfyui.log}"
export COMFY_READY_TIMEOUT_S="${COMFY_READY_TIMEOUT_S:-300}"
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

# Symlink SD cache into ComfyUI checkpoints
ln -sfn "${SD_CACHE_DIR}" "${COMFY_MODELS_DIR}/checkpoints/${COMFY_CHECKPOINT_SUBDIR}"
echo "[startup] Linked cache: ${COMFY_MODELS_DIR}/checkpoints/${COMFY_CHECKPOINT_SUBDIR} -> ${SD_CACHE_DIR}"

# main.py path
COMFY_ROOT="$(dirname "$COMFY_MODELS_DIR")"
export COMFY_MAIN="${COMFY_ROOT}/main.py"
if [ ! -f "${COMFY_MAIN}" ]; then
  echo "[startup] ERROR: COMFY_MAIN not found at ${COMFY_MAIN}"
  exit 1
fi

# Start ComfyUI and VERIFY it is ready
mkdir -p "$(dirname "${COMFY_LOG}")"
python3 "${COMFY_MAIN}" --listen "${COMFY_LISTEN_HOST}" --port "${COMFYUI_PORT}" ${COMFY_ARGS} >>"${COMFY_LOG}" 2>&1 &
COMFY_PID=$!
echo "${COMFY_PID}" > "${COMFY_PID_FILE}"

echo "[startup] ComfyUI pid=${COMFY_PID}, waiting for readiness at ${COMFY_URL}/system_stats ..."

python3 - <<'PY'
import os, time, requests, sys

url = os.environ["COMFY_URL"].rstrip("/") + "/system_stats"
timeout = int(os.environ.get("COMFY_READY_TIMEOUT_S", "300"))
pid_file = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
log_path = os.environ.get("COMFY_LOG", "/tmp/comfyui.log")

def pid_alive(pid: int) -> bool:
    try:
        import signal
        os.kill(pid, 0)
        return True
    except Exception:
        return False

pid = None
try:
    with open(pid_file, "r", encoding="utf-8") as f:
        pid = int(f.read().strip())
except Exception:
    pass

t0 = time.time()
while time.time() - t0 < timeout:
    # if process died, fail fast
    if pid is not None and not pid_alive(pid):
        print("[startup] ComfyUI process exited early. Last 200 log lines:")
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-200:]
            for line in lines:
                print(line.rstrip())
        except Exception as e:
            print("[startup] Could not read log:", e)
        sys.exit(1)

    try:
        r = requests.get(url, timeout=2)
        if r.ok:
            print("[startup] ComfyUI READY")
            sys.exit(0)
    except Exception:
        pass
    time.sleep(1)

print("[startup] ComfyUI did not become ready in time. Last 200 log lines:")
try:
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[-200:]
    for line in lines:
        print(line.rstrip())
except Exception as e:
    print("[startup] Could not read log:", e)
sys.exit(1)
PY

# Start RunPod handler
exec python3 -u /runpod_handler.py
