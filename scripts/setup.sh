#!/usr/bin/env bash
set -euo pipefail
# WPU client setup — idempotent; safe to re-run for updates on an already-set-up Pi.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

RUN_USER="$(id -un)"
RUN_UID="$(id -u)"

echo "[0/8] Normalise file permissions"
# Release zips have arrived on-device with the read bit stripped (mode 111),
# which makes every script unrunnable: bash must *read* a script, not just
# have +x on it. Re-assert sane modes on the unpacked tree.
chmod -R u+rwX,go+rX .
chmod +x scripts/*.sh

echo "[1/8] System packages"
sudo apt-get update
sudo apt-get -y upgrade
# Runtime system deps (no cmake/g++/libboost — those were only ever needed to
# compile dlib, which this project never actually depends on). GTK4 +
# Picamera2 + OpenCV libs. libcap-dev is needed to build python-prctl
# (a picamera2 dependency) — without it, `uv sync`/`pip install -e .` below
# fails with "You need to install libcap development headers".
sudo apt-get install -y \
    python3 python3-venv python3-dev \
    libgtk-4-1 gir1.2-gtk-4.0 python3-gi python3-gi-cairo \
    python3-picamera2 \
    libgl1 libglib2.0-0 libcap-dev \
    gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav

echo "[2/8] Python venv"
[ -d .venv ] || python3 -m venv --system-site-packages .venv   # system-site for picamera2/gi
./.venv/bin/python -m pip install --upgrade pip
# Prefer uv if present; else pip.
if command -v uv >/dev/null; then uv sync; else ./.venv/bin/pip install -e .; fi

echo "[3/8] Directories"
mkdir -p data/embeddings data/people data/stock_images models config
sudo mkdir -p /var/log/wpu-client && sudo chown "$RUN_USER" /var/log/wpu-client

echo "[4/8] Models (bundled in release — verify present)"
for m in models/face_recognition_sface_2021dec.onnx \
         models/face_detection_yunet_2023mar.onnx; do
  [ -f "$m" ] || { echo "MISSING $m — release zip incomplete"; exit 1; }
done
# mobilefacenet is research-only (config model: "mobilenet"); production runs
# sface, so its absence is not a provisioning failure.
[ -f models/mobilefacenet.onnx ] || echo "      note: models/mobilefacenet.onnx absent — research model unavailable"

echo "[5/8] Config"
# config/config.yaml is per-device and deliberately untracked/unshipped, so this
# always seeds from the example on a fresh unit and never clobbers a tuned one.
[ -f config/config.yaml ] || cp config/config.yaml.example config/config.yaml
echo "      endpoints in config/config.yaml — point them at the master server before first run:"
grep -E "^\s*(api_endpoint|wpu_endpoint|model):" config/config.yaml | sed 's/^/        /'

echo "[6/8] Install systemd units (paths/user/session substituted for this device)"
# The units in systemd/ are templates — the app may be unpacked anywhere and run
# as any user, so bake in what we actually detected rather than assuming
# /home/dreamvu/wpu_client and uid 1000.
if [ "${XDG_SESSION_TYPE:-}" = "x11" ]; then
  DISPLAY_ENV="Environment=DISPLAY=${DISPLAY:-:0}
Environment=XAUTHORITY=$HOME/.Xauthority"
else
  DISPLAY_ENV="Environment=WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
Environment=XDG_RUNTIME_DIR=/run/user/$RUN_UID"
fi
echo "      app dir : $APP_DIR"
echo "      user    : $RUN_USER (uid $RUN_UID)"
echo "      session : ${XDG_SESSION_TYPE:-unknown}"
for unit in slideshow-server.service slideshow-diagnostic.service; do
  awk -v app="$APP_DIR" -v usr="$RUN_USER" -v disp="$DISPLAY_ENV" '
    /^# TEMPLATE|^# scripts\/setup\.sh|^# values detected/ { next }
    { gsub(/@APP_DIR@/, app); gsub(/@RUN_USER@/, usr); gsub(/@DISPLAY_ENV@/, disp); print }
  ' "systemd/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload

echo "[7/8] Enable server mode by default, leave diagnostic installed-but-disabled"
sudo systemctl enable slideshow-server.service
sudo systemctl disable slideshow-diagnostic.service || true

echo "[8/8] (Re)start server service"
sudo systemctl restart slideshow-server.service
echo "Done. Switch modes with scripts/switch-mode.sh {server|diagnostic}"
