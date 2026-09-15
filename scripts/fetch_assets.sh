#!/usr/bin/env bash
set -euo pipefail
# Fetch the art this unit is missing, and verify what it already has, against
# data/assets.manifest.
#
#   ASSETS_URL=http://192.168.1.10/wpu scripts/fetch_assets.sh
#   ASSETS_URL=/mnt/share/wpu-assets    scripts/fetch_assets.sh
#   ASSETS_URL=rsync://host/wpu-assets  scripts/fetch_assets.sh
#   scripts/fetch_assets.sh --verify    check what is here; download nothing
#
# The transport is whatever ASSETS_URL's scheme says, because the decision of
# how a control node serves 50 Pis is not this script's to make: http(s) via
# curl, rsync, or a plain path (a mount, a USB stick, another checkout).
# Layout is identical in all three — the manifest's data/... paths hang
# directly off ASSETS_URL.
#
# Idempotent, and safe to interrupt: a file already present with the right
# checksum is skipped, so a re-run resumes rather than restarts. Nothing is
# ever deleted.
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

MANIFEST="data/assets.manifest"
[ -f "$MANIFEST" ] || { echo "missing $MANIFEST"; exit 1; }

VERIFY_ONLY=0
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=1

ASSETS_URL="${ASSETS_URL:-}"
if [ "$VERIFY_ONLY" -eq 0 ] && [ -z "$ASSETS_URL" ]; then
  echo "set ASSETS_URL to where the assets are served from, e.g."
  echo "    ASSETS_URL=http://192.168.1.10/wpu scripts/fetch_assets.sh"
  echo "or run with --verify to check what is already here."
  exit 2
fi
ASSETS_URL="${ASSETS_URL%/}"

fetch_one() {  # <relative path> <destination>
  local path="$1" dest="$2"
  case "$ASSETS_URL" in
    http://*|https://*)
      curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 10 \
           -o "$dest" "$ASSETS_URL/$path"
      ;;
    rsync://*)
      rsync -q "$ASSETS_URL/$path" "$dest"
      ;;
    *)
      cp "$ASSETS_URL/$path" "$dest"
      ;;
  esac
}

have=0; fetched=0; failed=0; corrupt=0
while IFS=$'\t' read -r want_sum want_size path; do
  case "$path" in ""|\#*) continue ;; esac

  if [ -f "$path" ]; then
    got_sum="$(sha256sum "$path" | cut -d' ' -f1)"
    if [ "$got_sum" = "$want_sum" ]; then
      have=$((have + 1))
      continue
    fi
    # Present but wrong. A truncated download from an interrupted run, or a
    # locally edited image the manifest has not caught up with — either way,
    # say so rather than silently replacing someone's work.
    echo "  DIFFERS  $path (have $got_sum, manifest says $want_sum)"
    corrupt=$((corrupt + 1))
    continue
  fi

  if [ "$VERIFY_ONLY" -eq 1 ]; then
    echo "  MISSING  $path"
    failed=$((failed + 1))
    continue
  fi

  mkdir -p "$(dirname "$path")"
  # Download to a temporary name and move into place only once the checksum
  # agrees, so an interrupted run never leaves a half-file that a later run
  # would report as DIFFERS.
  tmp="$path.part"
  if ! fetch_one "$path" "$tmp" 2>/dev/null; then
    echo "  FAILED   $path"
    rm -f "$tmp"
    failed=$((failed + 1))
    continue
  fi
  got_sum="$(sha256sum "$tmp" | cut -d' ' -f1)"
  if [ "$got_sum" != "$want_sum" ]; then
    echo "  BAD SUM  $path"
    rm -f "$tmp"
    failed=$((failed + 1))
    continue
  fi
  mv "$tmp" "$path"
  echo "  fetched  $path ($(numfmt --to=iec "$want_size" 2>/dev/null || echo "$want_size B"))"
  fetched=$((fetched + 1))
done < "$MANIFEST"

echo
echo "already present: $have   fetched: $fetched   missing/failed: $failed   mismatched: $corrupt"

# A unit that could not get all of its art still runs — it composes onto the
# scenes it has — so this is a non-zero exit for a deploy tool to notice, not
# a reason to stop. `main.py --check` reports the same gap under `scenes`.
if [ "$failed" -gt 0 ] || [ "$corrupt" -gt 0 ]; then
  exit 1
fi
echo "Complete asset set present."
