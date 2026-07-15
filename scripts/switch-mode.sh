#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  server)     sudo systemctl start slideshow-server.service ;;      # Conflicts= stops diagnostic
  diagnostic) sudo systemctl start slideshow-diagnostic.service ;;  # Conflicts= stops server
  *) echo "usage: switch-mode.sh {server|diagnostic}"; exit 1 ;;
esac
systemctl is-active slideshow-server.service slideshow-diagnostic.service || true
