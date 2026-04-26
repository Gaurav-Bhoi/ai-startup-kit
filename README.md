# RunPod ComfyUI + Qwen Serverless Kit

This image starts ComfyUI for image generation and a RunPod serverless handler for image/chat jobs. Models are cached on the RunPod network volume from S3-compatible storage such as Backblaze B2, Cloudflare R2, or plain S3.

## Files

- `Dockerfile`: builds from `runpod/worker-comfyui`
- `start.sh`: configures object storage, starts ComfyUI, then starts the RunPod handler
- `runpod_handler.py`: downloads model files safely, runs ComfyUI workflows, and loads Qwen with Transformers

## Required Storage Env

Use one of these groups.

Backblaze B2:

- `B2_REGION`
- `B2_MODELS_BUCKET`
- `B2_ACCESS_KEY_ID`
- `B2_SECRET_ACCESS_KEY`

Cloudflare R2:

- `R2_ACCOUNT_ID` or `R2_ENDPOINT`
- `R2_MODELS_BUCKET`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

Generic S3-compatible:

- `S3_ENDPOINT`
- `S3_MODELS_BUCKET`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Common Optional Env

- `SD_PREFIX`: default `models/`
- `LLM_PREFIX`: default `models/`
- `SD_CACHE_DIR`: default `/runpod-volume/models/checkpoints`
- `LLM_CACHE_DIR`: default `/runpod-volume/models/llm`
- `MIN_FREE_DISK_GIB`: default `5`; keeps free space reserved before model downloads
- `S3_SYNC_TIMEOUT_S`: default `21600`
- `S3_LOCK_STALE_TIMEOUT_S`: default `7200`
- `MAX_IMAGE_SIDE`: default `1536`
- `MAX_IMAGE_PIXELS`: default `1572864`
- `MAX_IMAGE_STEPS`: default `60`
- `MAX_QWEN_NEW_TOKENS`: default `1024`

## Failure Seen In Logs

The provided logs show Qwen2.5-VL-32B shards failing with `Errno 28: No space left on device` while downloading into `/runpod-volume/models/llm/Qwen2.5-VL-32B-Instruct`.

This means the RunPod network volume is too small for the model plus existing cache and reserve space. Increase the network volume, delete unused cached models, or use a smaller/quantized model folder.
