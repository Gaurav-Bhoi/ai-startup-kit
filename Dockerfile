FROM runpod/worker-comfyui:5.7.1-sdxl

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Python deps:
# - awscli: access B2 via S3 API
# - accelerate/transformers: Qwen runtime
# - qwen-vl-utils: required for Qwen-VL vision parsing
# - bitsandbytes: 4-bit loading (important on 24GB)
RUN python3 -m pip install --no-cache-dir \
  awscli \
  requests \
  "transformers>=4.45.0" \
  accelerate \
  qwen-vl-utils \
  bitsandbytes

COPY runpod_handler.py /runpod_handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
