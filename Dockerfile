FROM runpod/worker-comfyui:5.7.1-sdxl

USER root

# System deps (helps some vision/video stacks; safe to include)
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Python deps:
# - awscli + requests: for B2/R2 + comfy calls
# - accelerate: transformer runtime helper
# - qwen-vl-utils: Qwen VL helpers (vision)
# - transformers: install a recent version (Qwen often requires newer than distro)
RUN python3 -m pip install --no-cache-dir \
  awscli \
  requests \
  accelerate \
  qwen-vl-utils

# Install latest Transformers (often required for Qwen2.5-VL support)
RUN python3 -m pip install --no-cache-dir \
  "transformers>=4.45.0"

COPY runpod_handler.py /runpod_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
