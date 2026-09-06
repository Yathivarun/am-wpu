#!/usr/bin/env bash
set -euo pipefail
# WPU client setup — idempotent; safe to re-run for updates on an already-set-up Pi.
#
#   scripts/setup.sh                  every step, in order
#   scripts/setup.sh --skip-apt       everything but the apt step
#   scripts/setup.sh venv config      only those steps
#   scripts/setup.sh --list           what the steps are
#
# Steps are separate functions so a deploy tool can call one without the rest:
# apt is the slow half and rarely needs re-running, while `config` or `units`
# are worth applying on their own after a change. Each step is independently
# idempotent, so any subset can be re-run at any time.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

STEPS=(perms apt venv dirs models config units enable check)

usage() {
  echo "usage: scripts/setup.sh [--skip-apt] [--list] [step ...]"
  echo "steps: ${STEPS[*]}"
}

# ── who this is being set up for ────────────────────────────────────────
# Not `id -un`: under Ansible this script often runs as root, and the kiosk
# must not. WPU_USER is the explicit answer, SUDO_USER the one that is already
# correct when a human ran `sudo scripts/setup.sh`.
RUN_USER="${WPU_USER:-${SUDO_USER:-$(id -un)}}"
if [ "$RUN_USER" = "root" ]; then
  echo "refusing to install the kiosk as root — set WPU_USER=<login> and re-run" >&2
  exit 1
fi
if ! RUN_HOME="$(getent passwd "$RUN_USER" | cut -d: -f6)" || [ -z "$RUN_HOME" ]; then
  echo "no such user: $RUN_USER" >&2
  exit 1
fi
RUN_UID="$(id -u "$RUN_USER")"

# Already root under a deploy tool? Then there is no sudo to reach for, and on
# a minimal image there may be no sudo installed at all.
SUDO="sudo"
[ "$(id -u)" -eq 0 ] && SUDO=""

# Run something as the kiosk user. When this script *is* that user, that is
# just running it; when it is root, drop privileges so the venv and the data
# directories do not end up owned by root.
as_run_user() {
  if [ "$(id -un)" = "$RUN_USER" ]; then
    "$@"
  else
    $SUDO -u "$RUN_USER" "$@"
  fi
}

# ── steps ───────────────────────────────────────────────────────────────

step_perms() {
  echo "== perms: normalise file permissions"
  # Release zips have arrived on-device with the read bit stripped (mode 111),
  # which makes every script unrunnable: bash must *read* a script, not just
  # have +x on it. Re-assert sane modes on the unpacked tree.
  chmod -R u+rwX,go+rX .
  chmod +x scripts/*.sh
}

step_apt() {
  echo "== apt: system packages"
  # Runtime system deps (no cmake/g++/libboost — those were only ever needed to
  # compile dlib, which this project never actually depends on). GTK4 +
  # Picamera2 + OpenCV libs. libcap-dev is needed to build python-prctl
  # (a picamera2 dependency) — without it, `uv sync`/`pip install -e .` below
  # fails with "You need to install libcap development headers".
  $SUDO apt-get update
  $SUDO apt-get -y upgrade
  $SUDO apt-get install -y \
      python3 python3-venv python3-dev \
      libgtk-4-1 gir1.2-gtk-4.0 python3-gi python3-gi-cairo \
      python3-picamera2 \
      libgl1 libglib2.0-0 libcap-dev \
      gir1.2-gstreamer-1.0 gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly gstreamer1.0-libav
}

step_venv() {
  echo "== venv: python environment"
  [ -d .venv ] || as_run_user python3 -m venv --system-site-packages .venv  # system-site for picamera2/gi
  as_run_user ./.venv/bin/python -m pip install --upgrade pip
  # Prefer uv if present; else pip. --frozen installs exactly what uv.lock pins;
  # without it uv is free to re-resolve and has pulled NumPy 2.x back in, which
  # breaks picamera2's C ABI on first import.
  #
  # Both branches need a raised timeout for the same reason: the opencv wheel
  # is ~42 MB, and over a slow link the defaults (uv 30s, pip 15s) expire
  # mid-download and fail the whole install after their retries.
  if command -v uv >/dev/null; then
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}" as_run_user uv sync --frozen
  else
    as_run_user ./.venv/bin/pip install --timeout 120 --retries 5 -e .
  fi
}

step_dirs() {
  echo "== dirs: data and log directories"
  as_run_user mkdir -p data/embeddings data/people data/stock_images models config
  $SUDO mkdir -p /var/log/wpu-client && $SUDO chown "$RUN_USER" /var/log/wpu-client
  # app.log had no cap; on an SD card an unbounded log is a slow disk-full that
  # takes the kiosk down with it. Fleet log shipping reads the journal instead
  # (scripts/install-alloy.sh) — this file is for `tail -f` on the device.
  $SUDO install -m 644 deploy/logrotate/wpu-client /etc/logrotate.d/wpu-client
}

step_models() {
  echo "== models: verify present"
  # All three are required. mobilefacenet.onnx is the production recogniser — a
  # missing one is not a degraded install, it aborts the recognition thread at
  # startup and the kiosk silently shows stock images forever.
  local missing=0
  for m in models/mobilefacenet.onnx \
           models/face_detection_yunet_2023mar.onnx \
           models/face_recognition_sface_2021dec.onnx; do
    [ -f "$m" ] || { echo "   MISSING $m"; missing=1; }
  done
  [ "$missing" -eq 0 ] || { echo "   models are tracked in the repo — the tree is incomplete"; exit 1; }
  echo "   3/3 present"
}

step_config() {
  echo "== config: seed config/config.yaml"
  # config/config.yaml is per-device and deliberately untracked/unshipped, so this
  # always seeds from the example on a fresh unit and never clobbers a tuned one.
  [ -f config/config.yaml ] || as_run_user cp config/config.yaml.example config/config.yaml
  echo "   check these before first run — endpoints must point at the master"
  echo "   server, and scale_mode must match the panel:"
  grep -E "^\s*(api_endpoint|wpu_endpoint|sau_media_endpoint|model|scale_mode):" config/config.yaml \
    | sed 's/^/        /'
}

step_units() {
  echo "== units: install systemd units for this device"
  # The units in systemd/ are templates — the app may be unpacked anywhere and run
  # as any user, so bake in what we actually detected rather than assuming
  # /home/dreamvu/wpu_client and uid 1000.
  local display_env
  if [ "${XDG_SESSION_TYPE:-}" = "x11" ]; then
    display_env="Environment=DISPLAY=${DISPLAY:-:0}
Environment=XAUTHORITY=$RUN_HOME/.Xauthority"
  else
    display_env="Environment=WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0}
Environment=XDG_RUNTIME_DIR=/run/user/$RUN_UID"
  fi
  echo "   app dir : $APP_DIR"
  echo "   user    : $RUN_USER (uid $RUN_UID)"
  echo "   session : ${XDG_SESSION_TYPE:-unknown}"
  for unit in slideshow-server.service slideshow-diagnostic.service slideshow-only.service; do
    awk -v app="$APP_DIR" -v usr="$RUN_USER" -v disp="$display_env" '
      /^# TEMPLATE|^# scripts\/setup\.sh|^# values detected/ { next }
      { gsub(/@APP_DIR@/, app); gsub(/@RUN_USER@/, usr); gsub(/@DISPLAY_ENV@/, disp); print }
    ' "systemd/$unit" | $SUDO tee "/etc/systemd/system/$unit" >/dev/null
  done
  $SUDO systemctl daemon-reload
}

step_enable() {
  echo "== enable: server mode on boot, the other two off"
  # Server mode is what a deployed kiosk runs, so it is the one enabled at boot.
  # The other two are installed and ready but disabled — they exist to be turned
  # on deliberately, one at a time.
  #
  # Enabled, NOT started. Whichever unit runs holds the camera and the display
  # exclusively, and setup has just seeded config/config.yaml from the example:
  # starting now would run a kiosk against whatever endpoint the template
  # happens to name, and take the camera away from the seeding and benchmark
  # tools you probably want next. It comes up on the next boot, or start it by
  # hand once the config is right.
  $SUDO systemctl enable slideshow-server.service
  $SUDO systemctl disable slideshow-diagnostic.service 2>/dev/null || true
  $SUDO systemctl disable slideshow-only.service 2>/dev/null || true
  for unit in slideshow-server.service slideshow-diagnostic.service slideshow-only.service; do
    $SUDO systemctl stop "$unit" 2>/dev/null || true
  done
}

step_check() {
  echo "== check: pre-flight"
  # Non-fatal: report what is not ready rather than abort a setup that has
  # already done its work. `main.py --check` exits 1 on any failure, so a fleet
  # tool can assert on it directly instead.
  as_run_user ./.venv/bin/python main.py --check || true
}

# ── dispatch ────────────────────────────────────────────────────────────

skip_apt=0
requested=()
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-apt) skip_apt=1 ;;
    --list)     echo "${STEPS[*]}"; exit 0 ;;
    -h|--help)  usage; exit 0 ;;
    -*)         usage >&2; exit 2 ;;
    *)
      # shellcheck disable=SC2076
      [[ " ${STEPS[*]} " =~ " $1 " ]] || { echo "unknown step: $1" >&2; usage >&2; exit 2; }
      requested+=("$1")
      ;;
  esac
  shift
done

[ ${#requested[@]} -gt 0 ] || requested=("${STEPS[@]}")

for step in "${requested[@]}"; do
  if [ "$step" = "apt" ] && [ "$skip_apt" -eq 1 ]; then
    echo "== apt: skipped (--skip-apt)"
    continue
  fi
  "step_$step"
done

# Only worth printing after a full run; a single step speaks for itself.
[ ${#requested[@]} -eq ${#STEPS[@]} ] || exit 0

cat <<'EOF'

Setup complete — nothing is running yet. Three services are installed and the
camera is free.

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
