import base64
import gc
import json
import os
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod


# -----------------------------
# Basic config
# -----------------------------
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFY_URL = os.environ.get("COMFY_URL", f"http://{COMFYUI_HOST}:{COMFYUI_PORT}")

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

# Prefixes inside the bucket (object keys)
# Your current layout examples:
#   SD:   models/realvis/RealVisXL_V5.0_fp16.safetensors
#   Qwen: models/Qwen2.5-VL-32B-Instruct/<many files>
#
# Keep SD_PREFIX as "models/" if you haven't separated SD checkpoints.
# Recommended later: SD_PREFIX="models/checkpoints/"
SD_PREFIX = os.environ.get("SD_PREFIX") or os.environ.get("CHECKPOINTS_PREFIX") or "models/"
LLM_PREFIX = os.environ.get("LLM_PREFIX", "models/")

# Local cache (RunPod volume)
SD_CACHE_DIR = os.environ.get("SD_CACHE_DIR", "/runpod-volume/models/checkpoints")
LLM_CACHE_DIR = os.environ.get("LLM_CACHE_DIR", "/runpod-volume/models/llm")

# ComfyUI checkpoints subdir name (symlink created by start.sh)
COMFY_SUBDIR = os.environ.get("COMFY_CHECKPOINT_SUBDIR", "b2")

# Timeouts
COMFY_READY_TIMEOUT_S = int(os.environ.get("COMFY_READY_TIMEOUT_S", "300"))
COMFY_POLL_TIMEOUT_S = int(os.environ.get("COMFY_POLL_TIMEOUT_S", "240"))
S3_SYNC_TIMEOUT_S = int(os.environ.get("S3_SYNC_TIMEOUT_S", str(6 * 3600)))

# 24GB safety knobs
EXCLUSIVE_GPU_MODE = os.environ.get("EXCLUSIVE_GPU_MODE", "1") == "1"  # stop comfy when running Qwen
QWEN_LOAD_4BIT = os.environ.get("QWEN_LOAD_4BIT", "1") == "1"
QWEN_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN_DEFAULT_MAX_NEW_TOKENS", "256"))
QWEN_MAX_GPU_GIB = os.environ.get("QWEN_MAX_GPU_GIB", "22GiB")  # GPU cap for device_map
QWEN_MAX_CPU_GIB = os.environ.get("QWEN_MAX_CPU_GIB", "64GiB")  # set based on your RAM

# Optional manifest in bucket (so you can call model aliases)
# Example:
# {
#   "realvis": {"type":"sd","key":"models/realvis/RealVisXL_V5.0_fp16.safetensors"},
#   "qwen32b": {"type":"qwen2_5_vl","prefix":"models/Qwen2.5-VL-32B-Instruct/"}
# }
MODELS_MANIFEST_KEY = os.environ.get("MODELS_MANIFEST_KEY", "")
MODELS_MANIFEST_TTL_S = int(os.environ.get("MODELS_MANIFEST_TTL_S", "300"))

_ALLOWED_CKPT_EXT = (".safetensors", ".ckpt", ".pt", ".pth")


# -----------------------------
# Helpers: env + aws
# -----------------------------
def _require_env():
    missing = []
    if not S3_ENDPOINT:
        missing.append("S3_ENDPOINT (or B2_ENDPOINT / R2_ENDPOINT)")
    if not S3_MODELS_BUCKET:
        missing.append("B2_MODELS_BUCKET (or S3_MODELS_BUCKET / R2_MODELS_BUCKET)")
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")


def _aws(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
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
        _aws(cmd, timeout=timeout_s)

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


def _sync_dir_once(prefix: str, local_dir: str):
    """
    Syncs an S3 prefix (folder) into local_dir exactly once per worker
    using a marker file.
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
            if time.time() - t0 > S3_SYNC_TIMEOUT_S:
                raise RuntimeError(f"Timed out waiting for sync lock: {lock_path}")
            time.sleep(1.0)

    try:
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
        _aws(cmd, timeout=S3_SYNC_TIMEOUT_S)

        with open(marker, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


# -----------------------------
# Manifest (optional)
# -----------------------------
_manifest_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


def load_manifest() -> Dict[str, dict]:
    if not MODELS_MANIFEST_KEY:
        return {}
    now = time.time()
    if _manifest_cache["data"] is not None and (now - _manifest_cache["ts"] < MODELS_MANIFEST_TTL_S):
        return _manifest_cache["data"]

    os.makedirs(LLM_CACHE_DIR, exist_ok=True)
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
# ComfyUI process control (PID file written by start.sh)
# -----------------------------
PID_FILE = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
COMFY_MAIN = os.environ.get("COMFY_MAIN", "")  # set in start.sh
COMFY_LOG = os.environ.get("COMFY_LOG", "/tmp/comfyui.log")
COMFY_ARGS = os.environ.get("COMFY_ARGS", "--disable-metadata")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_pid() -> Optional[int]:
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def stop_comfy_if_running():
    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        return
    try:
        os.kill(pid, 15)  # SIGTERM
    except Exception:
        return

    # wait a bit
    t0 = time.time()
    while time.time() - t0 < 30:
        if not _pid_alive(pid):
            break
        time.sleep(0.5)

    # force kill if needed
    if _pid_alive(pid):
        try:
            os.kill(pid, 9)
        except Exception:
            pass


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
    # if already running
    pid = _read_pid()
    if pid and _pid_alive(pid):
        return

    if not COMFY_MAIN or not os.path.isfile(COMFY_MAIN):
        raise RuntimeError("COMFY_MAIN not set or invalid. start.sh must export COMFY_MAIN.")

    os.makedirs(os.path.dirname(COMFY_LOG), exist_ok=True)

    # start in background and write pid
    with open(COMFY_LOG, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            ["python3", COMFY_MAIN, "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT), *COMFY_ARGS.split()],
            stdout=logf,
            stderr=logf,
        )
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    _wait_comfy_ready()


# -----------------------------
# SD: key -> local + comfy ckpt_name
# -----------------------------
def _sd_key_to_local_and_ckpt_name(s3_key: str) -> Tuple[str, str]:
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


def _submit_and_get_png(workflow: dict, timeout_s: int = COMFY_POLL_TIMEOUT_S) -> Tuple[str, bytes]:
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
# Qwen runtime (Transformers)
# -----------------------------
_QWEN: Dict[str, Any] = {"dir": None, "model": None, "processor": None}


def _free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def unload_qwen():
    _QWEN["model"] = None
    _QWEN["processor"] = None
    _QWEN["dir"] = None
    _free_gpu()


def ensure_qwen_loaded(local_dir: str):
    if _QWEN["model"] is not None and _QWEN["dir"] == local_dir:
        return

    if EXCLUSIVE_GPU_MODE:
        stop_comfy_if_running()
        _free_gpu()

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
            quant_config = None

    # Memory caps help on 24GB cards (some layers may be offloaded to CPU)
    max_memory = {0: QWEN_MAX_GPU_GIB, "cpu": QWEN_MAX_CPU_GIB}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_dir,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )
    processor = AutoProcessor.from_pretrained(local_dir)

    _QWEN["dir"] = local_dir
    _QWEN["model"] = model
    _QWEN["processor"] = processor


def _normalize_messages(inp: dict) -> List[dict]:
    """
    Accepts:
      - input.messages as OpenAI-ish [{role, content: "text"}]
      - or Qwen-style [{role, content: [{type:'text'...},{type:'image'...}]}]
      - or input.prompt (+ optional image_url/image_base64)
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

    prompt = str(inp.get("prompt") or "")
    if not prompt:
        raise ValueError("Provide input.messages or input.prompt for task=chat")

    content: List[dict] = []
    if inp.get("image_url"):
        content.append({"type": "image", "image": str(inp["image_url"])})
    elif inp.get("image_base64"):
        b64 = str(inp["image_base64"])
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        content.append({"type": "image", "image": f"data:image/png;base64,{b64}"})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def qwen_chat(inp: dict) -> Dict[str, Any]:
    # qwen-vl-utils is required for process_vision_info
    from qwen_vl_utils import process_vision_info

    model = _QWEN["model"]
    processor = _QWEN["processor"]
    if model is None or processor is None:
        raise RuntimeError("Qwen not loaded")

    messages = _normalize_messages(inp)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    try:
        import torch
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
    except Exception:
        pass

    max_new_tokens = int(inp.get("max_new_tokens", QWEN_DEFAULT_MAX_NEW_TOKENS))
    temperature = float(inp.get("temperature", 0.2))

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    out = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return {"text": out}


# -----------------------------
# Model resolution
# -----------------------------
def resolve_spec(task: str, inp: dict) -> Dict[str, Any]:
    """
    Priority:
      1) manifest alias (input.model matches key in manifest)
      2) fallback:
         - task=image: model is path relative to SD_PREFIX OR full key
         - task=chat : model is folder relative to LLM_PREFIX OR full prefix
    """
    manifest = load_manifest()
    model_name = (inp.get("model") or "").strip()
    if model_name and model_name.lower() in manifest:
        spec = manifest[model_name.lower()]
        if not isinstance(spec, dict) or "type" not in spec:
            raise RuntimeError(f"Manifest entry '{model_name}' must contain 'type'")
        spec = dict(spec)
        spec["name"] = model_name.lower()
        return spec

    if not model_name:
        raise ValueError("Provide input.model or configure MODELS_MANIFEST_KEY")

    if task == "image":
        # full key?
        if model_name.startswith(SD_PREFIX):
            key = model_name
        else:
            key = f"{SD_PREFIX}{_sanitize_relpath(model_name)}"
        return {"type": "sd", "key": key, "name": model_name}

    if task == "chat":
        # full prefix?
        if model_name.startswith(LLM_PREFIX):
            prefix = model_name
        else:
            prefix = f"{LLM_PREFIX}{_sanitize_relpath(model_name)}"
        if not prefix.endswith("/"):
            prefix += "/"
        return {"type": "qwen2_5_vl", "prefix": prefix, "name": model_name}

    raise ValueError("task must be 'image' or 'chat'")


# -----------------------------
# Main handler (single endpoint)
# -----------------------------
def handler(event: dict):
    _require_env()
    os.makedirs(SD_CACHE_DIR, exist_ok=True)
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)

    inp = event.get("input", {}) or {}

    # Utility actions
    action = (inp.get("action") or "").lower().strip()
    if action in ("manifest", "list_manifest"):
        return {"manifest_key": MODELS_MANIFEST_KEY, "manifest": load_manifest()}

    # Decide task
    task = (inp.get("task") or "").lower().strip()
    if not task:
        task = "chat" if inp.get("messages") else "image"

    spec = resolve_spec(task, inp)

    # ---------------- image (SD / ComfyUI) ----------------
    if task == "image":
        if spec.get("type") != "sd":
            raise RuntimeError(f"task=image needs type=sd, got {spec.get('type')}")

        # If Qwen was loaded, unload to free VRAM (optional)
        if EXCLUSIVE_GPU_MODE:
            unload_qwen()

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
            "checkpoint_key": s3_key,
            "ckpt_name": ckpt_name,
            "seed": seed,
            "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
        }

    # ---------------- chat (Qwen folder) ----------------
    if task == "chat":
        if spec.get("type") not in ("qwen2_5_vl", "qwen", "qwen2.5_vl"):
            raise RuntimeError(f"task=chat currently supports Qwen type, got {spec.get('type')}")

        prefix = str(spec["prefix"])
        folder_name = prefix.rstrip("/").split("/")[-1]
        local_dir = os.path.join(LLM_CACHE_DIR, folder_name)

        _sync_dir_once(prefix, local_dir)

        ensure_qwen_loaded(local_dir)
        out = qwen_chat(inp)

        return {
            "task": "chat",
            "model_prefix": prefix,
            "text": out["text"],
        }

    raise ValueError("Unknown task. Use input.task='image' or 'chat'.")


runpod.serverless.start({"handler": handler})
