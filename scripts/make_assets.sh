#!/usr/bin/env bash
set -euo pipefail
# Regenerate data/assets.manifest, and pack the assets that are not tracked in
# git into a tarball for publishing to the asset host.
#
# Run this from a checkout that has the COMPLETE art set — that is the point:
# the manifest describes everything a fully-provisioned unit should have, and
# git carries only the subset that lets a fresh unit boot into a working
# kiosk. Whoever holds the full set runs this; devices run fetch_assets.sh.
#
#   scripts/make_assets.sh              manifest + wpu-assets.tar.gz
#   scripts/make_assets.sh --manifest   manifest only
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

MANIFEST="data/assets.manifest"
TARBALL="wpu-assets.tar.gz"

# Everything under these, minus the configs (text, always tracked) and the
# regenerable caches.
ROOTS=(data/base_scenes data/stock_images data/embeddings data/people)

echo "Scanning: ${ROOTS[*]}"
{
  echo "# WPU client asset manifest — sha256, bytes, path"
  echo "#"
  echo "# The complete art set. git tracks only enough of it for a fresh unit to"
  echo "# boot into a working kiosk; scripts/fetch_assets.sh pulls the rest and"
  echo "# verifies every entry against this file."
  echo "#"
  echo "# Regenerate with scripts/make_assets.sh from a checkout holding the"
  echo "# full set. Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  # scenes_config.json is excluded because it stays tracked in git — it is the
  # scene definitions, not art, and the boot set needs it. Everything else
  # under these roots is an asset, meta.json included: a gallery person's
  # metadata travels with the images it describes, and excluding it would
  # leave it neither tracked nor fetched.
  find "${ROOTS[@]}" -type f \
    ! -name 'scenes_config.json' ! -name '.gitkeep' \
    -print0 2>/dev/null \
  | sort -z \
  | xargs -0 -r sha256sum \
  | while read -r sum path; do
      printf '%s\t%s\t%s\n' "$sum" "$(stat -c%s "$path")" "$path"
    done
} > "$MANIFEST"

count=$(grep -vc '^#' "$MANIFEST" || true)
echo "Wrote $MANIFEST ($count assets)"

[ "${1:-}" = "--manifest" ] && exit 0

# Only the assets git does NOT track — the tracked subset is already on every
# device by virtue of having the code.
echo "Packing untracked assets into $TARBALL"
grep -v '^#' "$MANIFEST" | cut -f3 \
  | while read -r path; do
      git ls-files --error-unmatch "$path" >/dev/null 2>&1 || echo "$path"
    done > /tmp/wpu-assets-list.$$
packed=$(wc -l < /tmp/wpu-assets-list.$$)
tar -czf "$TARBALL" -T /tmp/wpu-assets-list.$$
rm -f /tmp/wpu-assets-list.$$
echo "Wrote $TARBALL ($packed files, $(du -h "$TARBALL" | cut -f1))"

cat <<EOF

Publish both to the asset host, keeping the data/ prefix in the layout:

    <asset host>/data/base_scenes/...
    <asset host>/data/stock_images/...

Devices then run:

    ASSETS_URL=http://<asset host> scripts/fetch_assets.sh
EOF
