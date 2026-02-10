import base64
import gc
import json
import os
import signal
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod

# -----------------------------
# Core config
# -----------------------------
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

# S3-compatible config (Backblaze B2 / Cloudflare R2)
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

# Where SD checkpoints live in your bucket (YOUR CASE looks like: models/<something>/<file>.safetensors)
SD_PREFIX = os.environ.get("SD_PREFIX", "models/")

# Where folder-based models live in your bucket (Qwen folder in your screenshot is: models/Qwen2.5-VL-32B-Instruct/...)
LLM_PREFIX = os.environ.get("LLM_PREFIX", "models/")

# Local caches (persistent RunPod volume)
SD_CACHE_DIR = os.environ.get("SD_CACHE_DIR", "/runpod-volume/models/checkpoints")
LLM_CACHE_DIR = os.environ.get("LLM_CACHE_DIR", "/runpod-volume/models/llm")

# In ComfyUI, we symlink SD_CACHE_DIR into checkpoints/<COMFY_SUBDIR>
COMFY_SUBDIR = os.environ.get("COMFY_CHECKPOINT_SUBDIR", "b2")

# Optional manifest in your bucket:
# {
#   "realvis": {"type":"sd", "key":"models/realvis/RealVisXL_V5.0_fp16.safetensors"},
#   "animagine": {"type":"sd", "key":"models/animagine/animagine-xl-4.0-opt.safetensors"},
#   "qwen32b": {"type":"qwen2_5_vl", "prefix":"models/Qwen2.5-VL-32B-Instruct/"}
# }
MODELS_MANIFEST_KEY = os.environ.get("MODELS_MANIFEST_KEY", "")  # e.g. "models/manifest.json"
MODELS_MANIFEST_TTL_S = int(os.environ.get("MODELS_MANIFEST_TTL_S", "300"))

# Cache list results (avoid listing bucket every request)
LIST_CACHE_TTL_S = int(os.environ.get("LIST_CACHE_TTL_S", "300"))

# ComfyUI readiness timeout (seconds)
COMFY_READY_TIMEOUT_S = int(os.environ.get("COMFY_READY_TIMEOUT_S", "300"))

# If you want the handler to stop ComfyUI when doing LLM inference (recommended to free VRAM)
EXCLUSIVE_GPU_MODE = os.environ.get("EXCLUSIVE_GPU_MODE", "1") == "1"

# Optional: load Qwen in 4bit (needs bitsandbytes installed; if not available, it will fallback to normal)
QWEN_LOAD_4BIT = os.environ.get("QWEN_LOAD_4BIT", "0") == "1"

_ALLOWED_CKPT_EXT = (".safetensors", ".ckpt", ".pt", ".pth")


# -----------------------------
# Utilities
# -----------------------------
def _require_env():
    missing = []
    if not S3_ENDPOINT:
        missing.append("B2_ENDPOINT (or S3_ENDPOINT / R2_ENDPOINT)")
    if not S3_MODELS_BUCKET:
        missing.append("B2_MODELS_BUCKET (or S3_MODELS_BUCKET / R2_MODELS_BUCKET)")
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")


def _aws(cmd: List[str]) -> Dict[str, Any]:
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


def _sanitize_relpath(rel: str) -> str:
    rel = rel.replace("\\", "/")
    while rel.startswith("/"):
        rel = rel[1:]
    if ".." in rel.split("/"):
        raise ValueError("Invalid path")
    return rel


def _download_file_if_missing(key: str, local_path: str, timeout_s: int = 3600):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

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
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return

        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        src = f"s3://{S3_MODELS_BUCKET}/{key}"
        cmd = ["aws", "s3", "cp", src, tmp_path, "--endpoint-url", S3_ENDPOINT, "--only-show-errors"]
        _aws(cmd)

        os.replace(tmp_path, local_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def _sync_dir_if_missing(prefix: str, local_dir: str, timeout_s: int = 6 * 3600):
    """
    Downloads an entire folder (S3 prefix) into local_dir via `aws s3 sync`.
    Uses a marker file so future calls are instant.
    """
    prefix = prefix.replace("\\", "/")
    if not prefix.endswith("/"):
        prefix += "/"

    os.makedirs(local_dir, exist_ok=True)

    marker = os.path.join(local_dir, ".sync_complete")
    if os.path.exists(marker):
        return

    lock_path = os.path.join(local_dir, ".sync.lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Timed out waiting for sync lock: {lock_path}")
            time.sleep(1.0)

    try:
        # Another process may have finished
        if os.path.exists(marker):
            return

        src = f"s3://{S3_MODELS_BUCKET}/{prefix}"
        cmd = [
            "aws",
            "s3",
            "sync",
            src,
            local_dir,
            "--endpoint-url",
            S3_ENDPOINT,
            "--only-show-errors",
        ]
        _aws(cmd)

        # Mark complete
        with open(marker, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))

    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


_list_cache = {"ts": 0.0, "sd_keys": [], "llm_prefixes": []}  # type: ignore


def list_sd_checkpoints(prefix: str) -> List[str]:
    now = time.time()
    if now - _list_cache["ts"] < LIST_CACHE_TTL_S and _list_cache["sd_keys"]:
        return _list_cache["sd_keys"]

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
            if k and k.lower().endswith(_ALLOWED_CKPT_EXT):
                keys.append(k)

        if resp.get("IsTruncated") is True and resp.get("NextContinuationToken"):
            token = resp["NextContinuationToken"]
        else:
            break

    _list_cache["ts"] = now
    _list_cache["sd_keys"] = keys
    return keys


_manifest_cache = {"ts": 0.0, "data": None}  # type: ignore


def load_manifest() -> Dict[str, dict]:
    if not MODELS_MANIFEST_KEY:
        return {}

    now = time.time()
    if _manifest_cache["data"] is not None and (now - _manifest_cache["ts"] < MODELS_MANIFEST_TTL_S):
        return _manifest_cache["data"]

    local_manifest = os.path.join(LLM_CACHE_DIR, ".models_manifest.json")
    _download_file_if_missing(MODELS_MANIFEST_KEY, local_manifest, timeout_s=300)

    with open(local_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError("Manifest must be a JSON object (dict)")

    _manifest_cache["ts"] = now
    _manifest_cache["data"] = data
    return data


# -----------------------------
# ComfyUI process management (lazy start)
# -----------------------------
_COMFY_PROC: Optional[subprocess.Popen] = None


def _find_comfy_main_py() -> str:
    """
    Try to find ComfyUI main.py in common images.
    You can override by exporting COMFY_MAIN=/path/to/main.py in start.sh.
    """
    if os.environ.get("COMFY_MAIN"):
        return os.environ["COMFY_MAIN"]

    candidates = [
        "/comfyui/main.py",
        "/workspace/ComfyUI/main.py",
        "/ComfyUI/main.py",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # Last resort: search known roots
    for root in ("/comfyui", "/workspace", "/"):
        try:
            out = subprocess.check_output(["bash", "-lc", f"ls -1 {root}/**/main.py 2>/dev/null | head -n 1"])
            cand = out.decode("utf-8").strip()
            if cand and os.path.isfile(cand):
                return cand
        except Exception:
            pass

    raise RuntimeError("Could not find ComfyUI main.py. Set COMFY_MAIN in start.sh.")


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


def start_comfy_if_needed():
    global _COMFY_PROC

    if _COMFY_PROC is not None:
        if _COMFY_PROC.poll() is None:
            return
        _COMFY_PROC = None

    main_py = _find_comfy_main_py()
    listen = os.environ.get("COMFYUI_HOST", "127.0.0.1")
    port = os.environ.get("COMFYUI_PORT", "8188")
    comfy_args = os.environ.get("COMFY_ARGS", "--disable-metadata")

    # Start in background
    log_path = os.environ.get("COMFY_LOG", "/tmp/comfyui.log")
    with open(log_path, "a", encoding="utf-8") as logf:
        _COMFY_PROC = subprocess.Popen(
            ["python3", main_py, "--listen", listen, "--port", str(port), *comfy_args.split()],
            stdout=logf,
            stderr=logf,
        )

    _wait_comfy_ready()


def stop_comfy_if_running():
    global _COMFY_PROC
    if _COMFY_PROC is None:
        return
    if _COMFY_PROC.poll() is not None:
        _COMFY_PROC = None
        return

    try:
        _COMFY_PROC.terminate()
        _COMFY_PROC.wait(timeout=30)
    except Exception:
        try:
            _COMFY_PROC.kill()
        except Exception:
            pass
    finally:
        _COMFY_PROC = None


# -----------------------------
# SD (ComfyUI) workflow
# -----------------------------
def _sd_key_to_local_and_ckpt_name(s3_key: str) -> Tuple[str, str]:
    """
    Download s3_key into SD_CACHE_DIR preserving relpath AFTER SD_PREFIX.
    ckpt_name must match ComfyUI "checkpoints/<COMFY_SUBDIR>/..."
    """
    if not s3_key.startswith(SD_PREFIX):
        raise ValueError(f"SD key must start with SD_PREFIX='{SD_PREFIX}' (got: {s3_key})")

    rel = _sanitize_relpath(s3_key[len(SD_PREFIX):])
    if not rel.lower().endswith(_ALLOWED_CKPT_EXT):
        raise ValueError(f"Unsupported checkpoint extension: {rel}")

    local_path = os.path.join(SD_CACHE_DIR, rel)
    ckpt_name = f"{COMFY_SUBDIR}/{rel}"
    return local_path, ckpt_name


def build_txt2img_workflow(
    ckpt_name: str,
    prompt: str,
    negative: str,
    seed: int,
    steps: int,
    cfg: float,
    width: int,
    height: int,
) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt_name}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
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
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "runpod"}},
    }


def _submit_and_get_png(workflow: dict, timeout_s: int = 180) -> Tuple[str, bytes]:
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


# -----------------------------
# Qwen2.5-VL (Transformers) runtime
# -----------------------------
_QWEN_STATE: Dict[str, Any] = {"dir": None, "model": None, "processor": None}


def _free_gpu_memory():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def unload_qwen():
    if _QWEN_STATE["model"] is not None:
        _QWEN_STATE["model"] = None
    if _QWEN_STATE["processor"] is not None:
        _QWEN_STATE["processor"] = None
    _QWEN_STATE["dir"] = None
    _free_gpu_memory()


def ensure_qwen_loaded(local_model_dir: str):
    """
    Loads Qwen2.5-VL from a LOCAL folder (downloaded from B2).
    """
    if _QWEN_STATE["model"] is not None and _QWEN_STATE["dir"] == local_model_dir:
        return

    if EXCLUSIVE_GPU_MODE:
        stop_comfy_if_running()

    unload_qwen()

    import torch
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration

    quant_config = None
    if QWEN_LOAD_4BIT:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
        except Exception:
            quant_config = None  # fallback to normal

    if quant_config is not None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_model_dir,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype="auto",
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            local_model_dir,
            device_map="auto",
            torch_dtype="auto",
        )

    processor = AutoProcessor.from_pretrained(local_model_dir)

    _QWEN_STATE["dir"] = local_model_dir
    _QWEN_STATE["model"] = model
    _QWEN_STATE["processor"] = processor


def _normalize_qwen_messages(inp: dict) -> List[dict]:
    """
    Accepts:
      - inp.messages in Qwen format (role + content list)
      - inp.messages in simple OpenAI-ish format (role + content string)
      - OR inp.prompt (+ optional image_url/image_base64) and makes messages
    """
    msgs = inp.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                out.append({"role": role, "content": content})
            else:
                out.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return out

    # Build from prompt (+ optional image)
    prompt = str(inp.get("prompt") or "")
    if not prompt:
        raise ValueError("For task=chat provide input.messages or input.prompt")

    content: List[dict] = []

    if inp.get("image_url"):
        content.append({"type": "image", "image": str(inp["image_url"])})
    elif inp.get("image_base64"):
        # RunPod payload limit exists; prefer image_url for large images.
        b64 = str(inp["image_base64"])
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        content.append({"type": "image", "image": f"data:image/png;base64,{b64}"})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def qwen_chat(inp: dict) -> Dict[str, Any]:
    """
    Returns: {"text": "..."}
    """
    from qwen_vl_utils import process_vision_info  # required by Qwen2.5-VL quickstart

    model = _QWEN_STATE["model"]
    processor = _QWEN_STATE["processor"]
    if model is None or processor is None:
        raise RuntimeError("Qwen is not loaded")

    messages = _normalize_qwen_messages(inp)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # move to the same device as model
    try:
        import torch
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
    except Exception:
        pass

    max_new_tokens = int(inp.get("max_new_tokens", 256))
    temperature = float(inp.get("temperature", 0.2))

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    # Remove prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {"text": out}


# -----------------------------
# Model resolution (manifest or fallback)
# -----------------------------
def resolve_model_spec(inp: dict) -> Dict[str, Any]:
    """
    Returns a spec dict describing what to run.
    Uses manifest if available. Otherwise fallback:
      - task=image: treat inp.model as SD checkpoint path relative to SD_PREFIX (or full key)
      - task=chat: treat inp.model as folder name relative to LLM_PREFIX (or full prefix)
    """
    task = (inp.get("task") or "").lower().strip()

    manifest = load_manifest()
    model_name = (inp.get("model") or "").strip()

    if model_name and model_name.lower() in manifest:
        entry = manifest[model_name.lower()]
        if not isinstance(entry, dict) or "type" not in entry:
            raise RuntimeError(f"Manifest entry for '{model_name}' must be an object with 'type'")
        spec = dict(entry)
        spec["name"] = model_name.lower()
        return spec

    # Fallback (no manifest or no alias)
    if task == "image":
        if not model_name:
            raise ValueError("For task=image provide input.model (checkpoint filename/path) or use manifest alias.")
        # Accept full key
        if model_name.startswith(SD_PREFIX):
            key = model_name
        else:
            rel = _sanitize_relpath(model_name)
            key = f"{SD_PREFIX}{rel}"
        return {"type": "sd", "key": key, "name": model_name}

    if task == "chat":
        if not model_name:
            raise ValueError("For task=chat provide input.model (folder name/path) or use manifest alias.")
        # Accept full prefix
        if model_name.startswith(LLM_PREFIX):
            prefix = model_name
        else:
            rel = _sanitize_relpath(model_name)
            prefix = f"{LLM_PREFIX}{rel}"
        if not prefix.endswith("/"):
            prefix += "/"
        return {"type": "qwen2_5_vl", "prefix": prefix, "name": model_name}

    raise ValueError("Unknown task. Use input.task='image' or 'chat'.")


# -----------------------------
# Main handler (single endpoint router)
# -----------------------------
def handler(event: dict):
    _require_env()

    inp = event.get("input", {}) or {}

    # Decide task:
    # - explicit input.task wins
    # - else: messages => chat, otherwise image
    task = (inp.get("task") or "").lower().strip()
    if not task:
        task = "chat" if inp.get("messages") else "image"

    # Quick utility: list manifest or list SD checkpoints
    if (inp.get("action") or "").lower() in ("list_models", "manifest"):
        return {"manifest_key": MODELS_MANIFEST_KEY, "models": load_manifest()}

    if (inp.get("action") or "").lower() in ("list_sd", "list_checkpoints", "list_image_models"):
        keys = list_sd_checkpoints(prefix=SD_PREFIX)
        short = [k[len(SD_PREFIX):] for k in keys if k.startswith(SD_PREFIX)]
        return {"prefix": SD_PREFIX, "count": len(keys), "keys": keys, "short": short}

    # Resolve model (manifest alias or fallback)
    inp["task"] = task
    spec = resolve_model_spec(inp)

    # ------------------ IMAGE (ComfyUI) ------------------
    if task == "image":
        if spec.get("type") != "sd":
            raise RuntimeError(f"task=image requires type=sd, got: {spec.get('type')}")

        # If we were running Qwen, unload it to free VRAM
        if EXCLUSIVE_GPU_MODE:
            unload_qwen()

        # Start ComfyUI when needed
        start_comfy_if_needed()

        s3_key = str(spec["key"])
        local_path, ckpt_name = _sd_key_to_local_and_ckpt_name(s3_key)
        _download_file_if_missing(s3_key, local_path)

        prompt = str(inp.get("prompt", ""))
        negative = str(inp.get("negative_prompt", ""))
        steps = int(inp.get("steps", 30))
        cfg = float(inp.get("cfg", 6))
        seed = int(inp.get("seed", int(time.time()) % 2_000_000_000))
        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1024))

        width = max(256, (width // 8) * 8)
        height = max(256, (height // 8) * 8)

        workflow = build_txt2img_workflow(
            ckpt_name=ckpt_name,
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
            "task": "image",
            "prompt_id": prompt_id,
            "model": spec.get("name"),
            "checkpoint_key": s3_key,
            "ckpt_name": ckpt_name,
            "seed": seed,
            "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
        }

    # ------------------ CHAT (Qwen folder) ------------------
    if task == "chat":
        if spec.get("type") not in ("qwen2_5_vl", "qwen2.5_vl", "qwen"):
            raise RuntimeError(
                f"task=chat currently supports type=qwen2_5_vl (set in manifest). Got: {spec.get('type')}"
            )

        # Download folder model from B2 (prefix -> local folder)
        prefix = str(spec["prefix"])
        # local dir name from last component of prefix
        folder_name = prefix.rstrip("/").split("/")[-1]
        local_dir = os.path.join(LLM_CACHE_DIR, folder_name)

        _sync_dir_if_missing(prefix, local_dir)

        ensure_qwen_loaded(local_dir)
        out = qwen_chat(inp)

        return {
            "task": "chat",
            "model": spec.get("name"),
            "model_prefix": prefix,
            "text": out["text"],
        }

    raise ValueError("Invalid task")


runpod.serverless.start({"handler": handler})
