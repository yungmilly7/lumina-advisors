#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04/24.04 VPS. Run as root (or with
# sudo) after you've SSH'd in and uploaded/cloned the stockgraph folder to
# /opt/stockgraph. See deploy/DEPLOY.md for the full walkthrough.
set -euo pipefail

APP_DIR=/opt/stockgraph
APP_USER=stockgraph

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this as root (e.g. 'sudo bash deploy/setup.sh')." >&2
  exit 1
fi

if [ ! -d "$APP_DIR" ]; then
  echo "Expected the app at $APP_DIR but it's not there." >&2
  echo "Copy/clone the stockgraph folder to $APP_DIR first, then re-run this script." >&2
  exit 1
fi

echo "==> Installing Python, build tools, and Caddy..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip curl debian-keyring debian-archive-keyring apt-transport-https gnupg

if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq
  apt-get install -y -qq caddy
fi

echo "==> Creating a dedicated, unprivileged '$APP_USER' user..."
id -u "$APP_USER" &>/dev/null || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"

echo "==> Setting up the Python virtualenv and installing numpy/pandas..."
cd "$APP_DIR"
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

if [ ! -f "$APP_DIR/.env" ]; then
  echo "==> No .env found -- copying .env.example. Edit $APP_DIR/.env before starting the service:"
  echo "    (fill in SEC_USER_AGENT and ANTHROPIC_API_KEY, set STOCKGRAPH_DATA_MODE=auto)"
  cp .env.example .env
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> Installing the systemd service..."
cp deploy/stockgraph.service /etc/systemd/system/stockgraph.service
systemctl daemon-reload
systemctl enable stockgraph

echo "==> Installing the Caddy reverse-proxy config..."
echo "    Edit deploy/Caddyfile with your real domain FIRST if you haven't."
cp deploy/Caddyfile /etc/caddy/Caddyfile
systemctl enable caddy

cat <<'EOF'

==================================================================
Setup done. Two things left before you're live:

1. Edit /opt/stockgraph/.env (SEC_USER_AGENT, ANTHROPIC_API_KEY,
   STOCKGRAPH_DATA_MODE=auto).
2. Edit /etc/caddy/Caddyfile with your real domain (or use the bare-IP
   block for now -- see the comments in that file).

Then start everything:

   sudo systemctl start stockgraph
   sudo systemctl start caddy
   sudo systemctl status stockgraph   # first boot takes a few minutes
                                       # while it ingests + trains

Watch the logs while it boots:

   sudo journalctl -u stockgraph -f
==================================================================
EOF
