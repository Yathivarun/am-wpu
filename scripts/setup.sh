#!/usr/bin/env bash
set -euo pipefail
# WPU client setup — idempotent; safe to re-run for updates on an already-set-up Pi.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

RUN_USER="$(id -un)"
RUN_UID="$(id -u)"

echo "[0/9] Normalise file permissions"
# Release zips have arrived on-device with the read bit stripped (mode 111),
# which makes every script unrunnable: bash must *read* a script, not just
# have +x on it. Re-assert sane modes on the unpacked tree.
chmod -R u+rwX,go+rX .
chmod +x scripts/*.sh

echo "[1/9] System packages"
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

echo "[2/9] Python venv"
[ -d .venv ] || python3 -m venv --system-site-packages .venv   # system-site for picamera2/gi
./.venv/bin/python -m pip install --upgrade pip
# Prefer uv if present; else pip. --frozen installs exactly what uv.lock pins;
# without it uv is free to re-resolve and has pulled NumPy 2.x back in, which
# breaks picamera2's C ABI on first import.
# UV_HTTP_TIMEOUT: the opencv wheel is ~42 MB and uv's 30s default expires
# mid-download on a slow link, failing the whole sync after 3 retries.
if command -v uv >/dev/null; then
  UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}" uv sync --frozen
else
  ./.venv/bin/pip install -e .
fi

echo "[3/9] Directories"
mkdir -p data/embeddings data/people data/stock_images models config
sudo mkdir -p /var/log/wpu-client && sudo chown "$RUN_USER" /var/log/wpu-client

echo "[4/9] Models (bundled in release — verify present)"
# All three are required. mobilefacenet.onnx is the production recogniser — a
# missing one is not a degraded install, it aborts the recognition thread at
# startup and the kiosk silently shows stock images forever.
for m in models/mobilefacenet.onnx \
         models/face_detection_yunet_2023mar.onnx \
         models/face_recognition_sface_2021dec.onnx; do
  [ -f "$m" ] || { echo "MISSING $m — release zip incomplete"; exit 1; }
done

echo "[5/9] Config"
# config/config.yaml is per-device and deliberately untracked/unshipped, so this
# always seeds from the example on a fresh unit and never clobbers a tuned one.
[ -f config/config.yaml ] || cp config/config.yaml.example config/config.yaml
echo "      settings in config/config.yaml — check these before first run:"
echo "        (endpoints must point at the master server; scale_mode must match the panel)"
grep -E "^\s*(api_endpoint|wpu_endpoint|sau_media_endpoint|model|scale_mode):" config/config.yaml \
  | sed 's/^/        /'

echo "[6/9] Install systemd units (paths/user/session substituted for this device)"
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
for unit in slideshow-server.service slideshow-diagnostic.service slideshow-only.service; do
  awk -v app="$APP_DIR" -v usr="$RUN_USER" -v disp="$DISPLAY_ENV" '
    /^# TEMPLATE|^# scripts\/setup\.sh|^# values detected/ { next }
    { gsub(/@APP_DIR@/, app); gsub(/@RUN_USER@/, usr); gsub(/@DISPLAY_ENV@/, disp); print }
  ' "systemd/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload

echo "[7/9] Enable server mode on boot; leave the other two off"
# Server mode is what a deployed kiosk runs, so it is the one enabled at boot.
# The other two are installed and ready but disabled — they exist to be turned
# on deliberately, one at a time.
#
# Enabled, NOT started. Whichever unit runs holds the camera and the display
# exclusively, and this script has just seeded config/config.yaml from the
# example: starting now would run a kiosk against whatever endpoint the
# template happens to name, and take the camera away from the seeding and
# benchmark tools you probably want next. It comes up on the next boot, or
# start it by hand once the config is right.
sudo systemctl enable slideshow-server.service
sudo systemctl disable slideshow-diagnostic.service 2>/dev/null || true
sudo systemctl disable slideshow-only.service 2>/dev/null || true
for unit in slideshow-server.service slideshow-diagnostic.service slideshow-only.service; do
  sudo systemctl stop "$unit" 2>/dev/null || true
done

echo "[8/9] Pre-flight check"
# Non-fatal: report what is not ready rather than abort a setup that has
# already done its work. `main.py --check` exits 1 on any failure, so a fleet
# tool can assert on it directly.
./.venv/bin/python main.py --check || true

echo "[9/9] Setup complete — nothing is running yet"
cat <<'EOF'
Done. Three services are installed; the camera is free.

    slideshow-server.service      ENABLED  — starts on boot (recognition + slideshow)
    slideshow-diagnostic.service  disabled — offline, local gallery
    slideshow-only.service        disabled — display only, no camera

Check config/config.yaml, then start server mode now (or just reboot):
    scripts/switch-mode.sh server

Switch modes (each stops the others):
    scripts/switch-mode.sh server | diagnostic | only | stop | status

Verify a unit before or after starting it:
    .venv/bin/python main.py --check

Run in the foreground with live logs instead (needs the camera free):
    .venv/bin/python main.py --service all              # server mode
    .venv/bin/python main.py --service all --diagnostic # diagnostic mode
EOF
