#!/usr/bin/env bash
# Start one mode, stop the others, or stop everything.
#
# The three units are mutually exclusive (Conflicts= in each): server and
# diagnostic both hold the camera, and all three drive the fullscreen display.
# systemd stops whichever is running when you start another, so these are
# one-way switches, not a start-on-top-of.
#
# Only slideshow-server is enabled at boot (see setup.sh). Starting another
# mode here does NOT change that — it runs until you switch back or reboot.
# Use `enable`/`disable` to change what comes up on boot.
set -euo pipefail

UNITS=(slideshow-server.service slideshow-diagnostic.service slideshow-only.service)

unit_for() {
  case "$1" in
    server)     echo slideshow-server.service ;;
    diagnostic) echo slideshow-diagnostic.service ;;
    only)       echo slideshow-only.service ;;
    *)          return 1 ;;
  esac
}

usage() {
  cat <<'USAGE'
usage: switch-mode.sh <command>

  server              start recognition + slideshow (online, via the server)
  diagnostic          start recognition + slideshow (offline, local gallery)
  only                start the slideshow alone — no camera, no recognition
  stop                stop all three; releases the camera and the display
  status              what is running, and what starts on boot
  logs [mode]         follow the log of a mode (default: whichever is running)
  enable <mode>       make this mode the one that starts on boot
  disable <mode>      stop this mode starting on boot
  check               run pre-flight checks against the current config
USAGE
}

case "${1:-}" in
  server|diagnostic|only)
    sudo systemctl start "$(unit_for "$1")"
    ;;
  stop)
    sudo systemctl stop "${UNITS[@]}"
    echo "All modes stopped — camera and display released."
    ;;
  status) ;;                                   # fall through to the report
  logs)
    unit=""
    if [ -n "${2:-}" ]; then
      unit="$(unit_for "$2")" || { usage; exit 1; }
    else
      for u in "${UNITS[@]}"; do
        systemctl is-active --quiet "$u" && unit="$u" && break
      done
      [ -n "$unit" ] || { echo "Nothing is running. Name a mode: switch-mode.sh logs server"; exit 1; }
    fi
    exec journalctl -u "$unit" -f -n 200
    ;;
  enable|disable)
    unit="$(unit_for "${2:-}")" || { usage; exit 1; }
    # Only one mode may start on boot; enabling one disables the rest so the
    # units cannot race for the camera at startup.
    if [ "$1" = "enable" ]; then
      for u in "${UNITS[@]}"; do
        [ "$u" = "$unit" ] || sudo systemctl disable "$u" 2>/dev/null || true
      done
    fi
    sudo systemctl "$1" "$unit"
    ;;
  check)
    APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    exec "$APP_DIR/.venv/bin/python" "$APP_DIR/main.py" --check
    ;;
  *) usage; exit 1 ;;
esac

# Both is-active and is-enabled exit non-zero for the states we still want to
# print ("inactive", "disabled", "not-found"), so take their output and only
# substitute a placeholder when there was none at all.
report() {
  local out
  out="$("$@" 2>/dev/null || true)"
  echo "${out:-unknown}"
}

printf '%-30s %-10s %s\n' UNIT ACTIVE ON-BOOT
for u in "${UNITS[@]}"; do
  printf '%-30s %-10s %s\n' "$u" \
    "$(report systemctl is-active "$u")" \
    "$(report systemctl is-enabled "$u")"
done
