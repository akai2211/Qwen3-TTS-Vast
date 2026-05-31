# ============================================================
# Qwen3-TTS — кастомный Gradio UI для Vast.ai
#
# RTX 50xx (Blackwell, sm_120): база CUDA 12.9 + PyTorch cu128 wheels.
#   Образы на torch cu121 падают на 5090: "no kernel image is available".
#
# Шаблон Vast (в Docker options / env шаблона, НЕ в ENV Dockerfile):
#   -p 8000:8000 -p 1111:1111 -p 22:22 -e OPEN_BUTTON_PORT=8000
#   Launch Mode: Docker ENTRYPOINT
#   Extra filters для 50xx: cuda_max_good>=12.8, БЕЗ compute_cap<=900
# ============================================================

FROM vastai/base-image:cuda-12.9-mini-py312-2026-04-15

LABEL maintainer="akai2211"
LABEL description="Qwen3-TTS Gradio UI for Vast.ai (cu128 / RTX 50xx + Ada/Ampere)"
LABEL org.opencontainers.image.source="https://github.com/akai2211/Qwen3-TTS-Vast"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    APP_DIR=/app \
    WORKSPACE=/workspace \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8000 \
    HF_HOME=/workspace/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/cache/huggingface

# sox — зависимость qwen-tts; openssh — Vast ENTRYPOINT mode
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        ffmpeg libsndfile1 sox \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd /etc/ssh/sshd_config.d \
    && chmod 700 /run/sshd \
    && printf '%s\n' \
        'PermitRootLogin prohibit-password' \
        'PasswordAuthentication no' \
        'PubkeyAuthentication yes' \
        'ChallengeResponseAuthentication no' \
        'UsePAM yes' \
        > /etc/ssh/sshd_config.d/vast.conf

WORKDIR ${APP_DIR}

# PyTorch cu128 — prebuilt wheels, sm_120 (RTX 5090 / 50xx)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# flash-attn: только готовый wheel (без компиляции в CI)
RUN pip install --no-cache-dir --only-binary :all: flash-attn \
    || echo "INFO: flash-attn wheel not found — app falls back to sdpa/eager"

COPY src/app/ ${APP_DIR}/

COPY scripts/entrypoint.sh     /usr/local/bin/entrypoint.sh
COPY scripts/setup_ssh.sh      /usr/local/bin/setup_ssh.sh
COPY scripts/setup_portal.sh   /usr/local/bin/setup_portal.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
             /usr/local/bin/setup_ssh.sh \
             /usr/local/bin/setup_portal.sh

EXPOSE 8000 1111 22

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -sf -o /dev/null "http://127.0.0.1:${GRADIO_SERVER_PORT}/" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "/app/app.py"]
