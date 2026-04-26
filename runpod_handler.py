import base64
import gc
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import runpod


# =========================
# ENV / CONFIG
# =========================
# Comfy (handler connects here)
COMFY_CONNECT_HOST = os.environ.get("COMFY_CONNECT_HOST", "127.0.0.1")
COMFYUI_PORT = os.environ.get("COMFYUI_PORT", "8188")
COMFY_URL = os.environ.get("COMFY_URL", f"http://{COMFY_CONNECT_HOST}:{COMFYUI_PORT}")

# Comfy start params (listen host can be 0.0.0.0)
COMFY_LISTEN_HOST = os.environ.get("COMFY_LISTEN_HOST", "0.0.0.0")
COMFY_ARGS = os.environ.get("COMFY_ARGS", "--disable-metadata")
COMFY_MAIN = os.environ.get("COMFY_MAIN", "")  # set by start.sh
COMFY_PID_FILE = os.environ.get("COMFY_PID_FILE", "/tmp/comfyui.pid")
COMFY_LOG = os.environ.get("COMFY_LOG", "/tmp/comfyui.log")

COMFY_READY_TIMEOUT_S = int(os.environ.get("COMFY_READY_TIMEOUT_S", "300"))
COMFY_POLL_TIMEOUT_S = int(os.environ.get("COMFY_POLL_TIMEOUT_S", "240"))

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

# Your bucket layout:
#   models/realvis/RealVisXL...safetensors
#   models/Qwen2.5-VL-32B-Instruct/...
SD_PREFIX = os.environ.get("SD_PREFIX") or os.environ.get("CHECKPOINTS_PREFIX") or "models/"
LLM_PREFIX = os.environ.get("LLM_PREFIX", "models/")

# Local caches (RunPod volume)
SD_CACHE_DIR = os.environ.get("SD_CACHE_DIR", "/runpod-volume/models/checkpoints")
LLM_CACHE_DIR = os.environ.get("LLM_CACHE_DIR", "/runpod-volume/models/llm")

# ComfyUI checkpoints symlink name created by start.sh
COMFY_SUBDIR = os.environ.get("COMFY_CHECKPOINT_SUBDIR", "b2")

# Qwen behavior (24GB)
EXCLUSIVE_GPU_MODE = os.environ.get("EXCLUSIVE_GPU_MODE", "1") == "1"  # stop comfy for qwen
QWEN_LOAD_4BIT = os.environ.get("QWEN_LOAD_4BIT", "1") == "1"
QWEN_DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("QWEN_DEFAULT_MAX_NEW_TOKENS", "256"))
QWEN_MAX_GPU_GIB = os.environ.get("QWEN_MAX_GPU_GIB", "22GiB")
QWEN_MAX_CPU_GIB = os.environ.get("QWEN_MAX_CPU_GIB", "64GiB")
QWEN_USE_FAST = os.environ.get("QWEN_USE_FAST", "0") == "1"  # set 1 if you want fast processor

S3_SYNC_TIMEOUT_S = int(os.environ.get("S3_SYNC_TIMEOUT_S", str(6 * 3600)))
S3_LOCK_STALE_TIMEOUT_S = int(os.environ.get("S3_LOCK_STALE_TIMEOUT_S", str(2 * 3600)))
MIN_FREE_DISK_GIB = float(os.environ.get("MIN_FREE_DISK_GIB", "5"))
AWS_ERROR_MAX_CHARS = int(os.environ.get("AWS_ERROR_MAX_CHARS", "4000"))

MAX_IMAGE_SIDE = int(os.environ.get("MAX_IMAGE_SIDE", "1536"))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(1536 * 1024)))
MAX_IMAGE_STEPS = int(os.environ.get("MAX_IMAGE_STEPS", "60"))
MAX_CFG = float(os.environ.get("MAX_CFG", "20"))
MAX_QWEN_NEW_TOKENS = int(os.environ.get("MAX_QWEN_NEW_TOKENS", "1024"))
MAX_IMAGE_BASE64_BYTES = int(os.environ.get("MAX_IMAGE_BASE64_BYTES", str(15 * 1024 * 1024)))
ALLOW_PRIVATE_IMAGE_URLS = os.environ.get("ALLOW_PRIVATE_IMAGE_URLS", "0") == "1"

_ALLOWED_CKPT_EXT = (".safetensors", ".ckpt", ".pt", ".pth")
_GIB = 1024 ** 3


# =========================
# BASIC HELPERS
# =========================
def _require_env():
    missing = []
    if not S3_ENDPOINT:
        missing.append("S3_ENDPOINT (or B2_ENDPOINT / R2_ENDPOINT)")
    if not S3_MODELS_BUCKET:
        missing.append("B2_MODELS_BUCKET (or S3_MODELS_BUCKET / R2_MODELS_BUCKET)")
    if missing:
        raise RuntimeError(f"Missing env: {', '.join(missing)}")


def _format_bytes(num: int) -> str:
    value = float(max(0, num))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _truncate(text: str, limit: int = AWS_ERROR_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...truncated..."


def _aws(cmd: List[str], timeout: Optional[int] = None) -> Dict[str, Any]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if p.returncode != 0:
        stderr = _truncate(p.stderr.strip())
        if "No space left on device" in p.stderr:
            stderr = (
                "No space left on the local model cache volume while downloading from object storage. "
                "Increase the RunPod network volume, remove unused cached models, or use a smaller model.\n"
                f"{stderr}"
            )
        raise RuntimeError(f"AWS CLI failed: {' '.join(cmd)}\nSTDERR:\n{stderr}")
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


def _acquire_lock(lock_path: str, timeout_s: int):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
            return
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > max(S3_LOCK_STALE_TIMEOUT_S, timeout_s + 60):
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            except Exception:
                pass

            if time.time() - t0 > timeout_s:
                raise RuntimeError(f"Timed out waiting for cache lock: {lock_path}")
            time.sleep(1.0)


def _release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def _file_complete(path: str, expected_size: Optional[int] = None) -> bool:
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if expected_size is None:
        return size > 0
    return size == expected_size


def _remove_if_exists(path: str):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _ensure_disk_space(target_dir: str, bytes_needed: int, context: str):
    os.makedirs(target_dir, exist_ok=True)
    free = shutil.disk_usage(target_dir).free
    reserve = int(MIN_FREE_DISK_GIB * _GIB)
    required = max(0, bytes_needed) + reserve
    if free < required:
        raise RuntimeError(
            "Not enough free disk space for "
            f"{context}: need {_format_bytes(bytes_needed)} plus "
            f"{_format_bytes(reserve)} reserve, but only {_format_bytes(free)} is free. "
            "Increase the RunPod network volume, delete unused cached models, or choose a smaller model."
        )


def _head_s3_object_size(key: str) -> int:
    resp = _aws(
        [
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
        ],
        timeout=120,
    )
    return int(resp.get("ContentLength") or 0)


def _list_s3_objects(prefix: str) -> List[Dict[str, Any]]:
    objects: List[Dict[str, Any]] = []
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
            cmd.extend(["--continuation-token", token])

        resp = _aws(cmd, timeout=120)
        for item in resp.get("Contents") or []:
            key = str(item.get("Key") or "")
            if not key or key.endswith("/"):
                continue
            objects.append({"Key": key, "Size": int(item.get("Size") or 0)})

        if not resp.get("IsTruncated"):
            break

        token = resp.get("NextContinuationToken")
        if not token:
            raise RuntimeError(f"S3 listing for {prefix} was truncated without a continuation token")

    return objects


def _sync_manifest(objects: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    return {
        "prefix": prefix,
        "object_count": len(objects),
        "total_bytes": sum(int(obj["Size"]) for obj in objects),
    }


def _sync_marker_valid(marker_path: str, manifest: Dict[str, Any], objects: List[Dict[str, Any]], local_dir: str) -> bool:
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return False

    for key in ("prefix", "object_count", "total_bytes"):
        if saved.get(key) != manifest.get(key):
            return False

    prefix = str(manifest["prefix"])
    for obj in objects:
        key = str(obj["Key"])
        rel = _sanitize_relpath(key[len(prefix):])
        if not rel:
            continue
        if not _file_complete(os.path.join(local_dir, rel), int(obj["Size"])):
            return False

    return True


def _write_sync_marker(marker_path: str, manifest: Dict[str, Any]):
    tmp_path = marker_path + ".part"
    payload = {
        **manifest,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
    os.replace(tmp_path, marker_path)


def _download_file_if_missing(key: str, local_path: str, timeout_s: int = 3600, expected_size: Optional[int] = None):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if expected_size is None:
        expected_size = _head_s3_object_size(key)

    if _file_complete(local_path, expected_size):
        return

    lock_path = local_path + ".lock"
    tmp_path = local_path + ".part"

    _acquire_lock(lock_path, timeout_s)
    try:
        if _file_complete(local_path, expected_size):
            return

        if os.path.exists(local_path) and not _file_complete(local_path, expected_size):
            _remove_if_exists(local_path)
        _remove_if_exists(tmp_path)

        _ensure_disk_space(os.path.dirname(local_path), expected_size, key)

        src = f"s3://{S3_MODELS_BUCKET}/{key}"
        cmd = ["aws", "s3", "cp", src, tmp_path, "--endpoint-url", S3_ENDPOINT, "--only-show-errors"]
        _aws(cmd, timeout=timeout_s)

        if not _file_complete(tmp_path, expected_size):
            actual = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
            raise RuntimeError(
                f"Downloaded size mismatch for {key}: expected {_format_bytes(expected_size)}, "
                f"got {_format_bytes(actual)}"
            )

        os.replace(tmp_path, local_path)
    finally:
        _remove_if_exists(tmp_path)
        _release_lock(lock_path)


def _sync_dir_once(prefix: str, local_dir: str):
    prefix = prefix.replace("\\", "/")
    if not prefix.endswith("/"):
        prefix += "/"

    os.makedirs(local_dir, exist_ok=True)

    marker = os.path.join(local_dir, ".sync_complete")
    objects = _list_s3_objects(prefix)
    if not objects:
        raise RuntimeError(f"No objects found in s3://{S3_MODELS_BUCKET}/{prefix}")

    manifest = _sync_manifest(objects, prefix)
    if _sync_marker_valid(marker, manifest, objects, local_dir):
        return

    lock_path = os.path.join(local_dir, ".sync.lock")
    _acquire_lock(lock_path, S3_SYNC_TIMEOUT_S)
    try:
        objects = _list_s3_objects(prefix)
        manifest = _sync_manifest(objects, prefix)
        if _sync_marker_valid(marker, manifest, objects, local_dir):
            return

        bytes_needed = 0
        planned: List[Tuple[str, str, int]] = []
        for obj in objects:
            key = str(obj["Key"])
            if not key.startswith(prefix):
                raise RuntimeError(f"Unexpected key outside requested prefix: {key}")
            rel = _sanitize_relpath(key[len(prefix):])
            if not rel:
                continue

            local_path = os.path.join(local_dir, rel)
            size = int(obj["Size"])
            if _file_complete(local_path, size):
                continue

            if os.path.exists(local_path):
                _remove_if_exists(local_path)
            _remove_if_exists(local_path + ".part")
            bytes_needed += size
            planned.append((key, local_path, size))

        _ensure_disk_space(local_dir, bytes_needed, f"s3://{S3_MODELS_BUCKET}/{prefix}")

        for key, local_path, size in planned:
            _download_file_if_missing(key, local_path, timeout_s=S3_SYNC_TIMEOUT_S, expected_size=size)

        _write_sync_marker(marker, manifest)
    finally:
        _release_lock(lock_path)


# =========================
# COMFY PROCESS CONTROL
# =========================
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_pid() -> Optional[int]:
    try:
        with open(COMFY_PID_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def _comfy_ok() -> bool:
    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        return r.ok
    except Exception:
        return False


def stop_comfy_if_running():
    pid = _read_pid()
    if not pid or not _pid_alive(pid):
        return

    try:
        os.kill(pid, 15)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < 30:
        if not _pid_alive(pid):
            break
        time.sleep(0.5)

    if _pid_alive(pid):
        try:
            os.kill(pid, 9)
        except Exception:
            pass


def _wait_comfy_ready(timeout_s: int = COMFY_READY_TIMEOUT_S):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if _comfy_ok():
            return
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time")


def start_comfy_if_needed():
    pid = _read_pid()
    if pid and _pid_alive(pid) and _comfy_ok():
        return

    # stale pid or unhealthy
    if pid and _pid_alive(pid):
        stop_comfy_if_running()

    if not COMFY_MAIN or not os.path.isfile(COMFY_MAIN):
        raise RuntimeError("COMFY_MAIN not set/invalid. start.sh must export COMFY_MAIN.")

    os.makedirs(os.path.dirname(COMFY_LOG), exist_ok=True)

    with open(COMFY_LOG, "a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            ["python3", COMFY_MAIN, "--listen", COMFY_LISTEN_HOST, "--port", str(COMFYUI_PORT), *shlex.split(COMFY_ARGS)],
            stdout=logf,
            stderr=logf,
        )

    with open(COMFY_PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    _wait_comfy_ready()


# =========================
# SD / COMFY WORKFLOW
# =========================
def _sd_key_to_local_and_ckpt_name(s3_key: str) -> Tuple[str, str]:
    if not s3_key.startswith(SD_PREFIX):
        raise ValueError(f"SD key must start with SD_PREFIX='{SD_PREFIX}' (got: {s3_key})")

    rel = _sanitize_relpath(s3_key[len(SD_PREFIX):])
    if not rel.lower().endswith(_ALLOWED_CKPT_EXT):
        raise ValueError(f"Unsupported checkpoint extension: {rel}")

    local_path = os.path.join(SD_CACHE_DIR, rel)
    ckpt_name = f"{COMFY_SUBDIR}/{rel}"  # must match symlink under ComfyUI checkpoints
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
    sampler_name: str = "euler",
    scheduler: str = "normal",
):
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
                "sampler_name": sampler_name,
                "scheduler": scheduler,
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

    def do_post():
        return requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )

    # If Comfy died, restart once
    try:
        r = do_post()
    except requests.exceptions.ConnectionError:
        start_comfy_if_needed()
        r = do_post()

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


# =========================
# QWEN (TRANSFORMERS)
# =========================
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


def _clamp_int(value: Any, default: int, minimum: int, maximum: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a number") from exc
    return max(minimum, min(maximum, parsed))


def _fit_image_size(width: int, height: int) -> Tuple[int, int]:
    width = max(256, min(MAX_IMAGE_SIDE, (width // 8) * 8))
    height = max(256, min(MAX_IMAGE_SIDE, (height // 8) * 8))

    pixels = width * height
    if pixels <= MAX_IMAGE_PIXELS:
        return width, height

    scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
    width = max(256, (int(width * scale) // 8) * 8)
    height = max(256, (int(height * scale) // 8) * 8)
    return width, height


def _host_looks_private(hostname: str) -> bool:
    try:
        ip = ipaddress.ip_address(hostname)
        return any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_reserved,
                ip.is_multicast,
                ip.is_unspecified,
            )
        )
    except ValueError:
        pass

    if hostname.lower() == "localhost":
        return True

    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve image URL host: {hostname}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if any(
            (
                ip.is_private,
                ip.is_loopback,
                ip.is_link_local,
                ip.is_reserved,
                ip.is_multicast,
                ip.is_unspecified,
            )
        ):
            return True

    return False


def _validate_image_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("image_url must use http or https")
    if not parsed.hostname:
        raise ValueError("image_url must include a hostname")
    if not ALLOW_PRIVATE_IMAGE_URLS and _host_looks_private(parsed.hostname):
        raise ValueError("image_url resolves to a private, local, or reserved address")
    return url


def _validate_image_data_url(value: str) -> str:
    if value.startswith("data:"):
        header, _, payload = value.partition(",")
        if not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise ValueError("image_base64 data URL must be a base64-encoded image")
        b64 = payload
    else:
        b64 = value

    approx_bytes = (len(b64) * 3) // 4
    if approx_bytes > MAX_IMAGE_BASE64_BYTES:
        raise ValueError(
            f"image_base64 is too large: max {_format_bytes(MAX_IMAGE_BASE64_BYTES)}, "
            f"got about {_format_bytes(approx_bytes)}"
        )

    if value.startswith("data:"):
        return value
    return f"data:image/png;base64,{b64}"


def _normalize_content_item(item: Any) -> dict:
    if not isinstance(item, dict):
        return {"type": "text", "text": str(item)}

    item_type = item.get("type")
    if item_type == "text":
        return {"type": "text", "text": str(item.get("text", ""))}

    if item_type in ("image", "image_url"):
        image_value: Any = item.get("image")
        if image_value is None:
            image_value = item.get("image_url")
        if isinstance(image_value, dict):
            image_value = image_value.get("url")
        if not image_value:
            raise ValueError("Image content must include image or image_url")

        image = str(image_value)
        if image.startswith("data:"):
            image = _validate_image_data_url(image)
        else:
            image = _validate_image_url(image)
        return {"type": "image", "image": image}

    raise ValueError(f"Unsupported message content type: {item_type}")


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

    max_memory = {0: QWEN_MAX_GPU_GIB, "cpu": QWEN_MAX_CPU_GIB}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_dir,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
    )

    processor = AutoProcessor.from_pretrained(local_dir, use_fast=QWEN_USE_FAST)

    _QWEN["dir"] = local_dir
    _QWEN["model"] = model
    _QWEN["processor"] = processor


def _normalize_messages(inp: dict) -> List[dict]:
    msgs = inp.get("messages")
    if isinstance(msgs, list) and msgs:
        out = []
        for m in msgs:
            if not isinstance(m, dict):
                raise ValueError("Each message must be an object")
            role = str(m.get("role", "user"))
            content = m.get("content", "")
            if isinstance(content, list):
                out.append({"role": role, "content": [_normalize_content_item(item) for item in content]})
            else:
                out.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
        return out

    prompt = str(inp.get("prompt") or "")
    if not prompt:
        raise ValueError("Provide input.messages or input.prompt for task=chat")

    content: List[dict] = []
    if inp.get("image_url"):
        content.append({"type": "image", "image": _validate_image_url(str(inp["image_url"]))})
    elif inp.get("image_base64"):
        content.append({"type": "image", "image": _validate_image_data_url(str(inp["image_base64"]))})

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def qwen_chat(inp: dict) -> Dict[str, Any]:
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

    # If the model has a primary device (usually cuda:0), put inputs there
    try:
        dev = getattr(model, "device", None)
        if dev is not None and str(dev).startswith("cuda"):
            inputs = inputs.to(dev)
    except Exception:
        pass

    max_new_tokens = _clamp_int(
        inp.get("max_new_tokens"),
        QWEN_DEFAULT_MAX_NEW_TOKENS,
        1,
        MAX_QWEN_NEW_TOKENS,
        "max_new_tokens",
    )
    temperature = _clamp_float(inp.get("temperature"), 0.2, 0.0, 2.0, "temperature")

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


# =========================
# JSON CLEANER (this is what you want)
# =========================
def _parse_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None

    t = text.strip().replace("\r", "")

    # strip ```json ... ``` or ``` ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)

    # direct json
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # fallback: first {...} block
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    return None


# =========================
# MODEL RESOLUTION
# =========================
def resolve_sd_key(model_input: str) -> str:
    model_input = (model_input or "").strip()
    if not model_input:
        raise ValueError("Provide input.model for task=image")

    # full key?
    if model_input.startswith(SD_PREFIX):
        return model_input

    rel = _sanitize_relpath(model_input)
    return f"{SD_PREFIX}{rel}"


def resolve_llm_prefix(model_input: str) -> str:
    model_input = (model_input or "").strip()
    if not model_input:
        raise ValueError("Provide input.model for task=chat")

    if model_input.startswith(LLM_PREFIX):
        rel = model_input[len(LLM_PREFIX):]
    else:
        rel = model_input

    rel = _sanitize_relpath(rel).strip("/")
    if not rel:
        raise ValueError("Provide a specific chat model folder, not the whole LLM prefix")

    prefix = f"{LLM_PREFIX.rstrip('/')}/{rel}"
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


# =========================
# MAIN HANDLER
# =========================
def handler(event: dict):
    _require_env()
    os.makedirs(SD_CACHE_DIR, exist_ok=True)
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)

    inp = event.get("input", {}) or {}

    task = (inp.get("task") or "").lower().strip()
    if not task:
        task = "chat" if inp.get("messages") or inp.get("prompt") else "image"

    # --------- CHAT (QWEN) ----------
    if task == "chat":
        prefix = resolve_llm_prefix(str(inp.get("model") or ""))
        llm_base = LLM_PREFIX.rstrip("/") + "/"
        local_rel = _sanitize_relpath(prefix[len(llm_base):].rstrip("/"))
        local_dir = os.path.join(LLM_CACHE_DIR, local_rel)

        if _QWEN["model"] is None or _QWEN["dir"] != local_dir:
            _sync_dir_once(prefix, local_dir)
        ensure_qwen_loaded(local_dir)

        out = qwen_chat(inp)

        # ⭐ Key part: return proper JSON fields if Qwen returned JSON in text
        parsed = _parse_json_from_text(out.get("text", ""))
        if parsed is not None:
            return {
                "task": "chat",
                "model_prefix": prefix,
                **parsed,                  # prompt, negative_prompt, width, height, etc.
                "raw_text": out.get("text", ""),
            }

        return {
            "task": "chat",
            "model_prefix": prefix,
            "text": out.get("text", ""),
        }

    # --------- IMAGE (COMFY) ----------
    if task == "image":
        # optional: free Qwen if running on 24GB
        if EXCLUSIVE_GPU_MODE:
            unload_qwen()

        start_comfy_if_needed()

        sd_key = resolve_sd_key(str(inp.get("model") or ""))

        local_path, ckpt_name = _sd_key_to_local_and_ckpt_name(sd_key)
        _download_file_if_missing(sd_key, local_path)

        prompt = str(inp.get("prompt", ""))
        negative = str(inp.get("negative_prompt", ""))
        steps = _clamp_int(inp.get("steps"), 30, 1, MAX_IMAGE_STEPS, "steps")
        cfg = _clamp_float(inp.get("cfg"), 6, 0.0, MAX_CFG, "cfg")
        seed = _clamp_int(inp.get("seed"), int(time.time()) % 2_000_000_000, 0, 2_147_483_647, "seed")
        width = _clamp_int(inp.get("width"), 1024, 256, MAX_IMAGE_SIDE, "width")
        height = _clamp_int(inp.get("height"), 1024, 256, MAX_IMAGE_SIDE, "height")

        sampler_name = str(inp.get("sampler_name", "euler"))
        scheduler = str(inp.get("scheduler", "normal"))

        width, height = _fit_image_size(width, height)

        workflow = build_txt2img_workflow(
            ckpt_name=ckpt_name,
            prompt=prompt,
            negative=negative,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
            sampler_name=sampler_name,
            scheduler=scheduler,
        )

        prompt_id, png_bytes = _submit_and_get_png(workflow)

        return {
            "task": "image",
            "prompt_id": prompt_id,
            "checkpoint_key": sd_key,
            "ckpt_name": ckpt_name,
            "width": width,
            "height": height,
            "seed": seed,
            "image_base64": base64.b64encode(png_bytes).decode("utf-8"),
        }

    raise ValueError("Unknown task. Use input.task='chat' or 'image'.")


runpod.serverless.start({"handler": handler})
