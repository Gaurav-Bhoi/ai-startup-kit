import base64
import os
import time
import uuid
import requests
import runpod

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

REALVIS_CKPT = os.environ.get("REALVIS_CKPT", "RealVisXL_V5.0_fp16.safetensors")
ANIMAGINE_CKPT = os.environ.get("ANIMAGINE_CKPT", "animagine-xl-4.0-opt.safetensors")

def _wait_comfy_ready(timeout_s: int = 90):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
            if r.ok:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time")

def _submit_and_get_png(workflow: dict, timeout_s: int = 180):
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    # Poll history until outputs appear
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        h.raise_for_status()
        hist = h.json()
        if prompt_id in hist and "outputs" in hist[prompt_id]:
            outputs = hist[prompt_id]["outputs"]
            # SaveImage node id = "7" in our workflow
            images = outputs.get("7", {}).get("images", [])
            if not images:
                raise RuntimeError(f"No images in outputs for prompt_id={prompt_id}")
            img0 = images[0]
            # Fetch the actual bytes from /view
            params = {
                "filename": img0["filename"],
                "subfolder": img0.get("subfolder", ""),
                "type": img0.get("type", "output"),
            }
            img = requests.get(f"{COMFY_URL}/view", params=params, timeout=30)
            img.raise_for_status()
            return prompt_id, img.content
        time.sleep(1)

    raise RuntimeError(f"Timed out waiting for ComfyUI output (prompt_id={prompt_id})")

def build_txt2img_workflow(
    ckpt_name: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
):
    # Node IDs are strings; keep them stable.
    return {
        "1": {  # checkpoint
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {  # positive
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {  # negative
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {  # empty latent
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {  # sampler
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {  # decode
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {  # save image
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "runpod"},
        },
    }

def handler(event):
    inp = event.get("input", {}) or {}

    model = (inp.get("model") or "realvis").lower()
    ckpt = REALVIS_CKPT if model in ["realvis", "real"] else ANIMAGINE_CKPT

    prompt = inp.get("prompt", "")
    negative = inp.get("negative_prompt", "")
    steps = int(inp.get("steps", 30))
    cfg = float(inp.get("cfg", 6))
    seed = int(inp.get("seed", int(time.time()) % 2_000_000_000))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    # Basic sanity
    width = max(256, (width // 8) * 8)
    height = max(256, (height // 8) * 8)

    _wait_comfy_ready()

    workflow = build_txt2img_workflow(
        ckpt_name=ckpt,
        prompt=prompt,
        negative=negative,
        seed=seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
    )

    prompt_id, png_bytes = _submit_and_get_png(workflow)

    return {
        "prompt_id": prompt_id,
        "model": model,
        "seed": seed,
        "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
    }

runpod.serverless.start({"handler": handler})
