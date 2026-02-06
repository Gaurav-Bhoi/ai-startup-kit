import base64
import json
import os
import subprocess
import time
import uuid
from typing import Dict, List, Optional, Tuple

import requests
import runpod

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

# S3-compatible config (Backblaze B2 / Cloudflare R2)
# Prefer B2_* / S3_* but keep R2_* as fallback for compatibility.
S3_ENDPOINT = (
    os.environ.get("B2_ENDPOINT")
    or os.environ.get("S3_ENDPOINT")
    or os.environ.get("R2_ENDPOINT")
)
S3_MODELS_BUCKET = (
    os.environ.get("B2_MODELS_BUCKET")
    or os.environ.get("S3_MODELS_BUCKET")
    or os.environ.get("R2_MODELS_BUCKET")
)
CHECKPOINTS_PREFIX = (
    os.environ.get("CHECKPOINTS_PREFIX")
    or os.environ.get("B2_CHECKPOINTS_PREFIX")
    or os.environ.get("R2_CHECKPOINTS_PREFIX")
    or "checkpoints/"
)

# Local cache (must match start.sh)
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/models/checkpoints")

# How cache is exposed to ComfyUI
# If your start.sh symlink is: .../checkpoints/b2 -> CACHE_DIR
# then keep this default as "b2"
COMFY_SUBDIR = os.environ.get("COMFY_CHECKPOINT_SUBDIR", "b2")

# Optional: alias manifest in S3 (lets you call model: "realvis" instead of filename)
# Example JSON:
# { "realvis": {"key":"checkpoints/RealVisXL_V5.0_fp16.safetensors"}, "animagine": {"key":"checkpoints/animagine-xl-4.0-opt.safetensors"} }
MODELS_MANIFEST_KEY = os.environ.get("MODELS_MANIFEST_KEY", "")  # e.g. "models/models.json"
MODELS_MANIFEST_TTL_S = int(os.environ.get("MODELS_MANIFEST_TTL_S", "300"))

# Cache list results (avoid listing bucket every request)
LIST_CACHE_TTL_S = int(os.environ.get("LIST_CACHE_TTL_S", "300"))

# Optional: limit local checkpoint cache size (GB). 0 = unlimited.
MAX_CACHE_GB = float(os.environ.get("MAX_CACHE_GB", "0"))

# ComfyUI readiness timeout (seconds)
COMFY_READY_TIMEOUT_S = int(os.environ.get("COMFY_READY_TIMEOUT_S", "300"))

_ALLOWED_EXT = (".safetensors", ".ckpt", ".pt", ".pth")


def _require_env():
    missing = []
    if not S3_ENDPOINT:
        missing.append("B2_ENDPOINT (or S3_ENDPOINT / R2_ENDPOINT)")
    if not S3_MODELS_BUCKET:
        missing.append("B2_MODELS_BUCKET (or S3_MODELS_BUCKET / R2_MODELS_BUCKET)")
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")


def _wait_comfy_ready(timeout_s: int = COMFY_READY_TIMEOUT_S):
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


def _aws(cmd: List[str]) -> dict:
    """Runs aws cli command and returns parsed json output when possible."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"AWS CLI failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    out = p.stdout.strip()
    if not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {"raw": out}


def _head_object_size(key: str) -> Optional[int]:
    """Best-effort: returns ContentLength for an object via awscli, or None if unavailable."""
    try:
        cmd = [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            S3_MODELS_BUCKET,
            "--key",
            key,
            "--endpoint-url",
            S3_ENDPOINT,
            "--output",
            "json",
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        meta = json.loads(out)
        size = meta.get("ContentLength")
        return int(size) if size is not None else None
    except Exception:
        return None


def _enforce_cache_limit():
    """Enforce MAX_CACHE_GB on CACHE_DIR by deleting oldest checkpoint files."""
    if MAX_CACHE_GB <= 0:
        return
    try:
        max_bytes = int(MAX_CACHE_GB * (1024**3))
    except Exception:
        return

    total = 0
    files = []
    for root, _, names in os.walk(CACHE_DIR):
        for name in names:
            if not name.endswith(_ALLOWED_EXT):
                continue
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            total += st.st_size
            files.append((st.st_mtime, st.st_size, path))

    if total <= max_bytes:
        return

    files.sort(key=lambda x: x[0])  # oldest first
    for _, size, path in files:
        if total <= max_bytes:
            break
        try:
            os.remove(path)
            total -= size
        except FileNotFoundError:
            continue


def _sanitize_relpath(rel: str) -> str:
    # prevent path traversal
    rel = rel.replace("\\", "/")
    while rel.startswith("/"):
        rel = rel[1:]
    if ".." in rel.split("/"):
        raise ValueError("Invalid path")
    return rel


def _key_to_local_path(key: str) -> Tuple[str, str]:
    """
    S3 key -> (local_path, comfy_ckpt_name)
    comfy_ckpt_name is relative to ComfyUI checkpoints folder.
    """
    if not key.startswith(CHECKPOINTS_PREFIX):
        raise ValueError(f"Checkpoint key must start with '{CHECKPOINTS_PREFIX}'")

    rel = key[len(CHECKPOINTS_PREFIX) :]
    rel = _sanitize_relpath(rel)

    if not rel.lower().endswith(_ALLOWED_EXT):
        raise ValueError(f"Unsupported checkpoint extension for '{rel}'")

    local_path = os.path.join(CACHE_DIR, rel)
    comfy_ckpt_name = f"{COMFY_SUBDIR}/{rel}"
    return local_path, comfy_ckpt_name


def _download_if_missing(key: str, local_path: str, timeout_s: int = 600):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # If file exists, trust it (but ensure non-empty)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    # Simple cross-process lock
    lock_path = local_path + ".lock"
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Timed out waiting for download lock: {lock_path}")
            time.sleep(0.5)

    tmp_path = local_path + ".part"
    try:
        # Another process may have finished while we waited
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return

        expected_size = _head_object_size(key)

        # Clean any stale partial
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

        src = f"s3://{S3_MODELS_BUCKET}/{key}"
        cmd = ["aws", "s3", "cp", src, tmp_path, "--endpoint-url", S3_ENDPOINT]
        _aws(cmd)

        # Validate size if we could fetch it
        if expected_size is not None:
            actual = os.path.getsize(tmp_path)
            if actual != expected_size:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Downloaded size mismatch for {key}: expected {expected_size} bytes, got {actual} bytes"
                )

        # Atomic move into place
        os.replace(tmp_path, local_path)
    finally:
        # Cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


_list_cache = {"ts": 0.0, "keys": []}  # type: ignore


def list_checkpoints(prefix: str) -> List[str]:
    """Lists all objects under prefix, paginated."""
    now = time.time()
    if now - _list_cache["ts"] < LIST_CACHE_TTL_S and _list_cache["keys"]:
        return _list_cache["keys"]

    keys: List[str] = []
    token: Optional[str] = None

    while True:
        cmd = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            S3_MODELS_BUCKET,
            "--prefix",
            prefix,
            "--endpoint-url",
            S3_ENDPOINT,
            "--output",
            "json",
        ]
        if token:
            cmd += ["--continuation-token", token]

        resp = _aws(cmd)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key")
            if k:
                keys.append(k)

        if resp.get("IsTruncated") is True and resp.get("NextContinuationToken"):
            token = resp["NextContinuationToken"]
        else:
            break

    # Keep only checkpoint-like files
    keys = [k for k in keys if k.lower().endswith(_ALLOWED_EXT)]

    _list_cache["ts"] = now
    _list_cache["keys"] = keys
    return keys


_manifest_cache = {"ts": 0.0, "data": None}  # type: ignore


def load_manifest() -> Dict[str, dict]:
    if not MODELS_MANIFEST_KEY:
        return {}

    now = time.time()
    if _manifest_cache["data"] is not None and (now - _manifest_cache["ts"] < MODELS_MANIFEST_TTL_S):
        return _manifest_cache["data"]

    # Download manifest to a temp path in cache
    local_manifest = os.path.join(CACHE_DIR, ".models_manifest.json")
    _download_if_missing(MODELS_MANIFEST_KEY, local_manifest, timeout_s=120)

    with open(local_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("Manifest must be a JSON object (dict)")

    _manifest_cache["ts"] = now
    _manifest_cache["data"] = data
    return data


def resolve_checkpoint_key(inp: dict) -> str:
    """
    Priority:
      1) checkpoint_key (full key)
      2) checkpoint (filename or subpath under checkpoints/)
      3) model (alias from manifest OR filename/subpath)
    """
    if "checkpoint_key" in inp and inp["checkpoint_key"]:
        return str(inp["checkpoint_key"])

    candidate = (inp.get("checkpoint") or inp.get("model") or "").strip()
    if not candidate:
        raise ValueError("Provide input.model or input.checkpoint or input.checkpoint_key")

    # If it already looks like a full key
    if candidate.startswith(CHECKPOINTS_PREFIX):
        return candidate

    # Try alias manifest
    manifest = load_manifest()
    alias = candidate.lower()
    if alias in manifest and isinstance(manifest[alias], dict) and manifest[alias].get("key"):
        return str(manifest[alias]["key"])

    # Otherwise treat as filename/subpath under checkpoints/
    rel = _sanitize_relpath(candidate)
    return f"{CHECKPOINTS_PREFIX}{rel}"


def _submit_and_get_png(workflow: dict, timeout_s: int = 180):
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
        h.raise_for_status()
        hist = h.json()
        if prompt_id in hist and "outputs" in hist[prompt_id]:
            outputs = hist[prompt_id]["outputs"]
            img0 = None
            for _node_id, _out in outputs.items():
                if isinstance(_out, dict) and _out.get("images"):
                    img0 = _out["images"][0]
                    break
            if img0 is None:
                raise RuntimeError(f"No images in outputs for prompt_id={prompt_id}")
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
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
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
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": "runpod"},
        },
    }


def handler(event):
    _require_env()
    _enforce_cache_limit()

    inp = event.get("input", {}) or {}

    # Utility: list models in bucket under CHECKPOINTS_PREFIX
    if (inp.get("action") or "").lower() in ("list_models", "list", "models"):
        keys = list_checkpoints(prefix=CHECKPOINTS_PREFIX)
        short = [k[len(CHECKPOINTS_PREFIX) :] for k in keys if k.startswith(CHECKPOINTS_PREFIX)]
        return {"count": len(keys), "keys": keys, "short": short}

    # Resolve + download checkpoint
    key = resolve_checkpoint_key(inp)
    local_path, comfy_ckpt_name = _key_to_local_path(key)
    _enforce_cache_limit()
    _download_if_missing(key, local_path)

    prompt = inp.get("prompt", "")
    negative = inp.get("negative_prompt", "")
    steps = int(inp.get("steps", 30))
    cfg = float(inp.get("cfg", 6))
    seed = int(inp.get("seed", int(time.time()) % 2_000_000_000))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))

    width = max(256, (width // 8) * 8)
    height = max(256, (height // 8) * 8)

    _wait_comfy_ready()

    workflow = build_txt2img_workflow(
        ckpt_name=comfy_ckpt_name,
        prompt=prompt,
        negative=negative,
        seed=seed,
        steps=steps,
        cfg=cfg,
        width=width,
        height=height,
    )

    _wait_comfy_ready()
    prompt_id, png_bytes = _submit_and_get_png(workflow)

    return {
        "prompt_id": prompt_id,
        "checkpoint_key": key,
        "ckpt_name": comfy_ckpt_name,
        "seed": seed,
        "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
    }


runpod.serverless.start({"handler": handler})
