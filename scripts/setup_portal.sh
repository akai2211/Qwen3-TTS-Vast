#!/usr/bin/env bash
# Instance Portal для Vast.ai (образы vastai/base-image).
# Отключить: ENABLE_PORTAL=0

set -euo pipefail

if [[ "${ENABLE_PORTAL:-1}" == "0" ]]; then
    echo "[portal] ENABLE_PORTAL=0 — skip"
    exit 0
fi

if ! command -v supervisord >/dev/null 2>&1; then
    echo "[portal] supervisord not found — skip"
    exit 0
fi

if [[ ! -d /opt/portal-aio/caddy_manager ]]; then
    echo "[portal] portal-aio not in image — skip"
    exit 0
fi

mkdir -p /var/log/portal

if [[ -z "${PORTAL_CONFIG:-}" ]]; then
    export PORTAL_CONFIG="localhost:8000:8000:/:Qwen3-TTS|localhost:1111:11111:/:Instance Portal"
fi

export AUTH_EXCLUDE="${AUTH_EXCLUDE:-8000}"

ensure_vast_tls() {
    if [[ -f /etc/instance.key && -f /etc/instance.crt ]]; then
        export ENABLE_HTTPS=true
        return 0
    fi

    local instance_id="${CONTAINER_ID:-${VAST_CONTAINERLABEL#C.}}"
    if [[ -n "$instance_id" ]] && command -v openssl >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
        cat > /etc/openssl-san.cnf <<'EOF'
[req]
default_bits       = 2048
distinguished_name = req_distinguished_name
req_extensions     = v3_req

[req_distinguished_name]
countryName         = US
stateOrProvinceName = CA
organizationName    = Vast.ai Inc.
commonName          = vast.ai

[v3_req]
basicConstraints = CA:FALSE
keyUsage         = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName   = @alt_names

[alt_names]
IP.1   = 0.0.0.0
EOF
        if openssl req -newkey rsa:2048 \
            -subj "/C=US/ST=CA/CN=jupyter.vast.ai/" \
            -nodes -sha256 \
            -keyout /etc/instance.key \
            -out /etc/instance.csr \
            -config /etc/openssl-san.cnf 2>/dev/null \
            && curl -fsSL --max-time 30 \
                --data-binary @/etc/instance.csr \
                -X POST "https://console.vast.ai/api/v0/sign_cert/?instance_id=${instance_id}" \
                -o /etc/instance.crt; then
            if [[ -s /etc/instance.crt ]]; then
                export ENABLE_HTTPS=true
                return 0
            fi
        fi
        rm -f /etc/instance.key /etc/instance.csr /etc/instance.crt
    fi

    export ENABLE_HTTPS=false
}

persist_portal_env() {
    touch /etc/environment
    for var in ENABLE_HTTPS PORTAL_CONFIG AUTH_EXCLUDE OPEN_BUTTON_PORT; do
        if [[ -n "${!var:-}" ]]; then
            if grep -q "^${var}=" /etc/environment 2>/dev/null; then
                sed -i "s|^${var}=.*|${var}='${!var}'|" /etc/environment
            else
                echo "${var}='${!var}'" >> /etc/environment
            fi
        fi
    done
    if [[ -x /opt/instance-tools/bin/export_env.sh ]]; then
        # shellcheck disable=SC1091
        . /opt/instance-tools/bin/export_env.sh || true
    fi
}

patch_supervisor_portal_env() {
    local conf="/etc/supervisor/conf.d/tunnel_manager.conf"
    local https_val="${ENABLE_HTTPS:-false}"
    if [[ -f "$conf" ]] && ! grep -q 'ENABLE_HTTPS=' "$conf"; then
        sed -i "s|environment=PROC_NAME=|environment=ENABLE_HTTPS=\"${https_val}\",PROC_NAME=|" "$conf"
    fi
}

ensure_vast_tls
persist_portal_env
patch_supervisor_portal_env

echo "[portal] generating portal config"
if ! (
    cd /opt/portal-aio/caddy_manager
    /opt/portal-aio/venv/bin/python caddy_config_manager.py
); then
    touch /etc/portal.yaml
fi

if pgrep -x supervisord >/dev/null 2>&1; then
    echo "[portal] supervisord already running"
    exit 0
fi

supervisord -c /etc/supervisor/supervisord.conf
echo "[portal] supervisord started"

exit 0
