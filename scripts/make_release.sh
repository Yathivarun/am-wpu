#!/usr/bin/env bash
set -euo pipefail
VER="${1:?usage: make_release.sh vX.Y.Z}"
OUT="wpu-client-${VER}.zip"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# Stamp the version so a deployed unit can say what it is running: `--check`
# reports it, and a fleet sweep compares it across 50 devices. Written as a
# file rather than patched into pyproject.toml because the zip is built from a
# checkout that must stay clean.
echo "$VER" > wpu_client/VERSION
FILELIST="$(mktemp)"
trap 'rm -f wpu_client/VERSION "$FILELIST"' EXIT

# Ship exactly what git tracks under these paths — NOT whatever a dev
# checkout happens to have lying around locally. `git ls-files` is the
# source of truth here rather than `zip -r` over the directory, because a
# dev box that has run fetch_assets.sh holds the COMPLETE art set on disk
# (hundreds of MB); zipping the directory would silently ship all of it.
# data/base_scenes/ + data/stock_images/ contribute only what git tracks —
# one background per scene directory, three stock images — the minimum a
# fresh unit needs to boot and pass `--check`. The rest is fetched after
# deploy with scripts/fetch_assets.sh, which is why data/assets.manifest
# (also tracked) has to ship too — without it fetch_assets.sh has nothing
# to fetch against.
#
# data/embeddings/ and data/people/ (diagnostic-mode seed data) are
# deliberately NOT shipped: neither is git-tracked, neither is read by
# server-mode recognition or by `--check` outside --diagnostic, and both are
# reseeded independently with scripts/seed_face.py. See make_assets.sh.
#
# scripts/benchmarks/ is tracked but dev-only tooling, not part of a
# deployed unit, so it is excluded explicitly.
TRACKED_PATHS=(
    main.py pyproject.toml uv.lock README.md
    wpu_client config scripts systemd deploy
    models/mobilefacenet.onnx models/face_recognition_sface_2021dec.onnx
    models/face_detection_yunet_2023mar.onnx
    data/base_scenes data/stock_images data/assets.manifest
)

git ls-files -- "${TRACKED_PATHS[@]}" | grep -v '^scripts/benchmarks/' > "$FILELIST"
# .python-version and wpu_client/VERSION are intentionally untracked (see
# .gitignore) but still required on the device — uv pins the interpreter
# from the former, main.py/--check read the version we just stamped from the
# latter — so both are added by name rather than picked up by git ls-files.
echo ".python-version" >> "$FILELIST"
echo "wpu_client/VERSION" >> "$FILELIST"

zip -q "$OUT" -@ < "$FILELIST"
echo "Wrote $OUT ($VER, $(wc -l < "$FILELIST") files)"
