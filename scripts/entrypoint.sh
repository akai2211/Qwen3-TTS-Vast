#!/usr/bin/env bash
# Entrypoint: /workspace persistence + SSH + Portal, then Gradio UI.

set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
WORKSPACE="${WORKSPACE:-/workspace}"

echo "────────────────────────────────────────────"
echo " Qwen3-TTS (Vast.ai)"
echo "  APP_DIR   = $APP_DIR"
echo "  WORKSPACE = $WORKSPACE"
echo "────────────────────────────────────────────"

mkdir -p "$WORKSPACE"/{cache/huggingface,outputs,log,custom_voices,designed_voices}

export HF_HOME="${HF_HOME:-$WORKSPACE/cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export WORKSPACE

if [[ -x /usr/local/bin/setup_ssh.sh ]]; then
    /usr/local/bin/setup_ssh.sh || true
fi

if [[ -x /usr/local/bin/setup_portal.sh ]]; then
    /usr/local/bin/setup_portal.sh || true
fi

echo "────────────────────────────────────────────"
echo " Запуск: $*"
echo "  UI: http://0.0.0.0:${GRADIO_SERVER_PORT:-8000}/"
echo "  HF cache: $HF_HOME"
echo "────────────────────────────────────────────"

cd "$APP_DIR"
exec "$@"
