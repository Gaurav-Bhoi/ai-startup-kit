FROM runpod/worker-comfyui:5.7.1-sdxl

USER root
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*
COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]
