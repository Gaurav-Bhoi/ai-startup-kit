FROM runpod/worker-comfyui:5.7.1-sdxl

USER root
RUN python3 -m pip install --no-cache-dir awscli

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
