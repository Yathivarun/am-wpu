#!/usr/bin/env bash
# Start one mode, stop the other, or stop both.
#
# Neither unit is enabled at boot (see setup.sh): whichever one runs holds the
# Pi camera exclusively, which would otherwise block seed_face.py, the
# benchmark scripts and any manual `python main.py` run. `stop` frees it.
set -euo pipefail
case "${1:-}" in
  server)     sudo systemctl start slideshow-server.service ;;      # Conflicts= stops diagnostic
  diagnostic) sudo systemctl start slideshow-diagnostic.service ;;  # Conflicts= stops server
  stop)
    sudo systemctl stop slideshow-server.service slideshow-diagnostic.service
    echo "Both services stopped — camera released."
    ;;
  status) ;;                                                        # fall through to the report
  *) echo "usage: switch-mode.sh {server|diagnostic|stop|status}"; exit 1 ;;
esac
systemctl is-active slideshow-server.service slideshow-diagnostic.service || true
