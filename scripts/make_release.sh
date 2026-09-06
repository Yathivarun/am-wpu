#!/usr/bin/env bash
set -euo pipefail
VER="${1:?usage: make_release.sh vX.Y.Z}"
OUT="wpu-client-${VER}.zip"

# Stamp the version so a deployed unit can say what it is running: `--check`
# reports it, and a fleet sweep compares it across 50 devices. Written as a
# file rather than patched into pyproject.toml because the zip is built from a
# checkout that must stay clean.
echo "$VER" > wpu_client/VERSION
trap 'rm -f wpu_client/VERSION' EXIT

# Ship code + runtime models (both recognisers) + config + the scene art and
# stock images that are tracked + the 3 seeded people + scripts/systemd/deploy.
# Exclude git, venv, caches, dataset, benchmarks output, archive.
#
# data/base_scenes/ is not optional: base mode composes every slide onto those
# backgrounds, so a zip without them installs a kiosk that shows stock images
# forever. The full art set is fetched separately (scripts/fetch_assets.sh);
# what is tracked here is the subset that makes a unit work out of the box.
zip -r "$OUT" \
    main.py pyproject.toml uv.lock README.md .python-version \
    wpu_client/ config/ scripts/ systemd/ deploy/ \
    models/mobilefacenet.onnx models/face_recognition_sface_2021dec.onnx \
    models/face_detection_yunet_2023mar.onnx \
    data/base_scenes/ \
    data/stock_images/ \
    data/embeddings/varun data/embeddings/samvaran data/embeddings/kevin \
    data/people/Varun.png data/people/Samvaran.png data/people/Kevin.png \
    -x '*/__pycache__/*' '*.pyc' '*/.venv/*' '*/dataset/*' '*/archive/*' \
       'scripts/benchmarks/*' \
       'config/config.yaml' 'config/config.local.yaml'
echo "Wrote $OUT ($VER)"
