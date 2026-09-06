#!/usr/bin/env bash
set -euo pipefail
# Install Grafana Alloy and point it at Loki. Idempotent; safe to re-run.
#
# This is deliberately NOT part of setup.sh. Alloy is fleet infrastructure,
# not part of the application: a unit with no Alloy runs the kiosk perfectly
# well, and a fleet deploys Alloy once, centrally, rather than as a side
# effect of every app update. Ansible substitutes the same two values into
# deploy/alloy/config.alloy.template that this script does — the template is
# the single definition, this script is the manual path for a one-off device.
#
#   scripts/install-alloy.sh                       # uses the defaults below
#   LOKI_URL=http://10.0.0.2:3100/loki/api/v1/push scripts/install-alloy.sh
#
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# The monitoring host from the original single-Pi setup. Change it here, in
# the environment, or in /etc/alloy/config.alloy afterwards — it is one line.
LOKI_URL="${LOKI_URL:-http://192.168.1.10:3100/loki/api/v1/push}"

TEMPLATE="deploy/alloy/config.alloy.template"
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE"; exit 1; }

# The same id the app stamps on every log line, so a Grafana panel and a
# shell on the device agree on which unit they are looking at.
DEVICE_ID="$(./.venv/bin/python -c 'from wpu_client.device import device_id; print(device_id())' 2>/dev/null \
             || python3 -c 'from wpu_client.device import device_id; print(device_id())')"

echo "[1/4] Install Alloy"
if ! command -v alloy >/dev/null; then
  sudo mkdir -p /etc/apt/keyrings
  wget -q -O - https://apt.grafana.com/gpg.key \
    | sudo gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
  echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" \
    | sudo tee /etc/apt/sources.list.d/grafana.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y alloy
else
  echo "      already installed: $(alloy --version 2>/dev/null | head -1)"
fi

echo "[2/4] Write /etc/alloy/config.alloy"
echo "      loki   : $LOKI_URL"
echo "      device : $DEVICE_ID"
sudo mkdir -p /etc/alloy
# Drop the template header (everything through the ---8<--- marker), then
# substitute. A marker rather than a pattern per header line: the header is
# prose and gets rewrapped, and a stale pattern fails silently by leaking a
# comment into a config file nobody re-reads.
awk -v loki="$LOKI_URL" -v dev="$DEVICE_ID" '
  !body { if ($0 ~ /^\/\/---8<---$/) body = 1; next }
  { gsub(/@LOKI_URL@/, loki); gsub(/@DEVICE_ID@/, dev); print }
' "$TEMPLATE" | sudo tee /etc/alloy/config.alloy >/dev/null

echo "[3/4] Let Alloy read the journal"
# alloy runs as its own user; systemd-journal membership is what grants it the
# whole journal rather than just its own unit's lines.
sudo usermod -aG systemd-journal alloy 2>/dev/null || true

echo "[4/4] Enable and restart"
sudo systemctl enable alloy
sudo systemctl restart alloy
sleep 2
sudo systemctl --no-pager --lines=0 status alloy || true

cat <<EOM

Alloy is shipping this unit's slideshow-*.service journal to Loki as
device="$DEVICE_ID".

    sudo journalctl -u alloy -f          # is it shipping?
    .venv/bin/python main.py --check     # reports alloy among the checks

In Grafana:

    {application="wpu-client", device="$DEVICE_ID"}
    {application="wpu-client"} | unit = "slideshow-server.service"
EOM
