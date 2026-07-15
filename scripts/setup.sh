#!/usr/bin/env bash
set -euo pipefail
# WPU client setup — idempotent; safe to re-run for updates on an already-set-up Pi.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

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
    libgl1 libglib2.0-0 libcap-dev

echo "[2/8] Python venv"
[ -d .venv ] || python3 -m venv --system-site-packages .venv   # system-site for picamera2/gi
./.venv/bin/python -m pip install --upgrade pip
# Prefer uv if present; else pip.
if command -v uv >/dev/null; then uv sync; else ./.venv/bin/pip install -e .; fi

echo "[3/8] Directories"
mkdir -p data/embeddings data/people data/stock_images models config
sudo mkdir -p /var/log/wpu-client && sudo chown "$USER" /var/log/wpu-client

echo "[4/8] Models (bundled in release — verify present; both recognisers kept for v1)"
for m in models/mobilefacenet.onnx models/face_recognition_sface_2021dec.onnx \
         models/face_detection_yunet_2023mar.onnx; do
  [ -f "$m" ] || { echo "MISSING $m — release zip incomplete"; exit 1; }
done

echo "[5/8] Config"
[ -f config/config.yaml ] || cp config/config.yaml.example config/config.yaml

echo "[6/8] Install systemd units"
sudo cp systemd/slideshow-server.service systemd/slideshow-diagnostic.service /etc/systemd/system/
sudo systemctl daemon-reload

echo "[7/8] Enable server mode by default, leave diagnostic installed-but-disabled"
sudo systemctl enable slideshow-server.service
sudo systemctl disable slideshow-diagnostic.service || true

echo "[8/8] (Re)start server service"
sudo systemctl restart slideshow-server.service
echo "Done. Switch modes with scripts/switch-mode.sh {server|diagnostic}"
