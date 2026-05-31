#!/usr/bin/env bash
# SSH для Vast.ai (режим Docker ENTRYPOINT). Ключ: env SSH_PUBLIC_KEY.
# Отключить: ENABLE_SSH=0

set -euo pipefail

if [[ "${ENABLE_SSH:-1}" == "0" ]]; then
    echo "[ssh] ENABLE_SSH=0 — skip"
    exit 0
fi

if ! command -v sshd >/dev/null 2>&1; then
    echo "[ssh] openssh-server not installed — skip"
    exit 0
fi

mkdir -p /root/.ssh
chmod 700 /root/.ssh

auth_keys="/root/.ssh/authorized_keys"
touch "$auth_keys"
chmod 600 "$auth_keys"

if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
    if ! grep -qF "$SSH_PUBLIC_KEY" "$auth_keys" 2>/dev/null; then
        echo "$SSH_PUBLIC_KEY" >> "$auth_keys"
    fi
    echo "[ssh] public key from SSH_PUBLIC_KEY installed"
else
    echo "[ssh] warning: SSH_PUBLIC_KEY is empty"
    echo "       add key in Vast.ai → Account → Keys, then Edit → Apply"
fi

mkdir -p /run/sshd
chmod 700 /run/sshd

if pgrep -x sshd >/dev/null 2>&1; then
    echo "[ssh] sshd already running"
else
    /usr/sbin/sshd
    echo "[ssh] sshd started (internal port 22)"
fi

exit 0
