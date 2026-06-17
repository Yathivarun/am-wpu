#!/usr/bin/env python3
"""
Stage-A sandbox test harness — validate the WPU identify path WITHOUT a Raspberry Pi.

Runs the EXACT on-device recognition pipeline (YuNet detect -> SFace alignCrop ->
feature -> L2-normalise) on a static image, then POSTs the 128-D vector to the
sandbox `/api/v1/identify/` endpoint with model=sface, exactly like
FaceRecognitionService does. If a visit_id comes back it also calls
`/api/v1/wpu/images` and checks the signed URLs are reachable.

This deliberately re-implements the embedding steps instead of importing
wpu_client.services.face_recognition.face_service, because that module imports
picamera2 at the top level and will not load on a non-Pi machine.

Usage:
    python tools/test_identify_sandbox.py --image /path/to/registered_person.jpg
    python tools/test_identify_sandbox.py --image face.jpg --base-url http://192.168.1.11:28000
    python tools/test_identify_sandbox.py --image face.jpg --no-wpu     # skip image fetch

Requirements: opencv-python, numpy (same as the client). HTTP uses the stdlib,
so no extra deps are needed to run it from a laptop on the sandbox LAN.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

# Model files live alongside the service code (gitignored, deployed separately).
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "wpu_client", "services", "face_recognition",
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_FILENAME = "face_recognition_sface_2021dec.onnx"


def build_embedding(image_path: str, models_dir: str, min_face_size: int):
    """Run YuNet + SFace on one image and return the L2-normalised 128-D vector.

    Mirrors FaceRecognitionService._capture_and_process exactly:
      - YuNet params (320x320, score 0.6, nms 0.3, top_k 5000), input size set per-image
      - largest face by bbox area (cols 2,3 = w,h)
      - alignCrop on the BGR image with the RAW YuNet row (bbox + 5 landmarks)
      - feature() -> flatten -> L2-normalise
    """
    yunet_path = os.path.join(models_dir, YUNET_FILENAME)
    sface_path = os.path.join(models_dir, SFACE_FILENAME)
    for p in (yunet_path, sface_path):
        if not os.path.exists(p):
            sys.exit(f"ERROR: model not found: {p}")

    # cv2.imread returns BGR — same byte order the Pi's capture_array and the
    # registration server's cv2.imdecode produce. Do NOT convert to RGB.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"ERROR: failed to decode image: {image_path}")
    h, w = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
    detector.setInputSize((w, h))
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        sys.exit("ERROR: no face detected in image")

    # Drop non-finite / runaway rows, same guard the service uses.
    coord_limit = 10 * max(h, w)
    valid = []
    for row in faces:
        arr = np.asarray(row, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            continue
        if float(np.max(np.abs(arr[:14]))) > coord_limit:
            continue
        valid.append(row)
    if not valid:
        sys.exit("ERROR: no valid face rows after filtering")

    largest = max(valid, key=lambda r: float(r[2]) * float(r[3]))
    fw, fh = int(largest[2]), int(largest[3])
    print(f"[detect] {len(valid)} face(s); largest bbox {fw}x{fh} at "
          f"({int(largest[0])},{int(largest[1])})")
    if fw < min_face_size or fh < min_face_size:
        print(f"[warn]  face smaller than min_face_size={min_face_size}; the Pi "
              f"would SKIP this frame")

    aligned = recognizer.alignCrop(image, largest)
    feat = recognizer.feature(aligned).flatten().astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    print(f"[embed]  {feat.shape[0]}-D, L2-norm={np.linalg.norm(feat):.6f}, "
          f"first3=[{feat[0]:.4f}, {feat[1]:.4f}, {feat[2]:.4f}]")
    return feat.tolist()


def post_identify(base_url: str, vector, n: int, model: str) -> dict:
    """POST the vector as form data — same shape IdentifyRequest produces."""
    url = base_url.rstrip("/") + "/api/v1/identify/"
    form = {
        "type": "face",
        "n": str(n),
        "model": model,
        "face_vector": ",".join(str(v) for v in vector),
    }
    data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    print(f"\n[POST]   {url}  (type=face, n={n}, model={model}, dim={len(vector)})")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: identify returned HTTP {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: could not reach {url} ({e.reason}). Is the ekanth sandbox up?")
    print("[resp]   " + json.dumps(body, indent=2))
    return body


def fetch_wpu_images(base_url: str, visit_id: str) -> list:
    url = base_url.rstrip("/") + "/api/v1/wpu/images?" + urllib.parse.urlencode(
        {"visit_uuid": visit_id}
    )
    print(f"\n[GET]    {url}")
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"[warn]   wpu/images HTTP {e.code}: {e.read().decode(errors='replace')}")
        return []
    except urllib.error.URLError as e:
        print(f"[warn]   could not reach wpu/images ({e.reason})")
        return []
    urls = body.get("signed_urls", [])
    print(f"[resp]   {len(urls)} signed URL(s)")
    return urls


def check_url_reachable(url: str) -> None:
    """GET a few bytes of the first signed URL to confirm the Pi could load it.

    Presigned URLs embed the server's MINIO_ENDPOINT; if that points at an
    address the client can't reach (e.g. internal 'minio:9000' or the wrong
    host), the slideshow would silently show nothing.
    """
    host = urllib.parse.urlparse(url).netloc
    print(f"[image]  probing first signed URL (host={host}) ...")
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-1023"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            n = len(resp.read())
        print(f"[image]  OK — fetched {n} bytes. The Pi can load visitor images.")
    except Exception as e:
        print(f"[image]  UNREACHABLE: {e}\n"
              f"         -> identify works but the slideshow image fetch will fail.\n"
              f"         -> set the sandbox server's MINIO_ENDPOINT to a host:port the "
              f"Pi can reach (sandbox MinIO host port is 29000).")


def main():
    ap = argparse.ArgumentParser(description="Stage-A WPU identify sandbox test")
    ap.add_argument("--image", required=True, help="Path to a registered person's photo")
    ap.add_argument("--base-url", default="http://192.168.1.11:28000",
                    help="Sandbox base URL (default: http://192.168.1.11:28000)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="sface")
    ap.add_argument("--min-face-size", type=int, default=100)
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--no-wpu", action="store_true", help="Skip the wpu/images fetch")
    args = ap.parse_args()

    vector = build_embedding(args.image, args.models_dir, args.min_face_size)
    resp = post_identify(args.base_url, vector, args.n, args.model)

    name = resp.get("name")
    visit_id = resp.get("visit_id")
    if name:
        print(f"\n==> MATCH: {name}  (distance={resp.get('distance')}, "
              f"confidence={resp.get('confidence')}, match_type={resp.get('match_type')})")
    else:
        print(f"\n==> NO MATCH ({resp.get('message')}). "
              f"Check the person is registered in the sandbox with an sface 128-D vector.")

    if visit_id and not args.no_wpu:
        urls = fetch_wpu_images(args.base_url, visit_id)
        if urls:
            check_url_reachable(urls[0])


if __name__ == "__main__":
    main()
