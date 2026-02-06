# RunPod ComfyUI Serverless Startup Kit (Network Volume + R2)

Files:

- Dockerfile: based on runpod/worker-comfyui
- start.sh: starts ComfyUI + serverless handler
- runpod_handler.py: picks a checkpoint from R2 (cached on network volume) and runs a ComfyUI workflow

Key env vars (required):

- R2_ACCOUNT_ID
- R2_MODELS_BUCKET
- R2_ACCESS_KEY_ID
- R2_SECRET_ACCESS_KEY

Optional:

- CACHE_DIR (default /runpod-volume/models/checkpoints)
- MAX_CACHE_GB (default 0 = unlimited)
- COMFYUI_HOST (default 127.0.0.1)
- COMFYUI_PORT (default 8188)
- COMFY_ARGS (extra args passed to ComfyUI)
- MODELS_MANIFEST_KEY (optional alias manifest JSON in R2)
