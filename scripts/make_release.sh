#!/usr/bin/env bash
set -euo pipefail
VER="${1:?usage: make_release.sh vX.Y.Z}"
OUT="wpu-client-${VER}.zip"
# Ship code + runtime models (both recognisers, v1) + config + stock images +
# the 3 seeded people + scripts/systemd. Exclude git, venv, caches, dataset,
# benchmarks output, archive.
zip -r "$OUT" \
    main.py pyproject.toml uv.lock README.md .python-version \
    wpu_client/ config/ scripts/ systemd/ \
    models/mobilefacenet.onnx models/face_recognition_sface_2021dec.onnx \
    models/face_detection_yunet_2023mar.onnx \
    data/stock_images/ \
    data/embeddings/varun data/embeddings/samvaran data/embeddings/kevin \
    data/people/Varun.png data/people/Samvaran.png data/people/Kevin.png \
    -x '*/__pycache__/*' '*.pyc' '*/.venv/*' '*/dataset/*' '*/archive/*' \
       'scripts/benchmarks/*'
echo "Wrote $OUT"
