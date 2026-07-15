#!/usr/bin/env python3
"""
Stage-A sandbox test harness — validate the WPU identify path WITHOUT a Raspberry Pi.

Runs the EXACT on-device recognition pipeline (YuNet detect -> align -> feature ->
L2-normalise, via the shared embedder module) on a static image, then POSTs the
resulting vector to the sandbox `/api/v1/identify/` endpoint with the matching
`model`, exactly like FaceRecognitionService does. If a visit_id comes back it
also calls `/api/v1/wpu/images` and checks the signed URLs are reachable.

This imports `wpu_client.services.face_recognition.sface_embedder` (not
`face_service`, which imports picamera2 at the top level and would not load on
a non-Pi machine) so the embedding steps stay byte-identical to the live path
by construction rather than by hand-kept-in-sync duplication.

Usage:
    python scripts/test_identify_sandbox.py --image /path/to/registered_person.jpg
    python scripts/test_identify_sandbox.py --image face.jpg --model mobilenet
    python scripts/test_identify_sandbox.py --image face.jpg --base-url http://192.168.1.11:28000
    python scripts/test_identify_sandbox.py --image face.jpg --no-wpu     # skip image fetch

Requirements: opencv-python, numpy, onnxruntime (same as the client).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wpu_client.services.face_recognition.sface_embedder import EMBEDDERS, build_embedder  # noqa: E402

# config `model` value → server gallery id, same mapping as face_service.py.
_SERVER_MODEL_ID = {"sface": "sface", "mobilenet": "auraface"}


def build_embedding(image_path: str, model_name: str, min_face_size: int):
    """Run YuNet + <model_name> on one image and return the L2-normalised vector.

    Mirrors FaceRecognitionService._capture_and_process exactly (same embedder
    class the live service and the seed tool both use):
      - YuNet params (320x320, score 0.6, nms 0.3, top_k 5000), input size set per-image
      - largest face by bbox area
      - model-specific align + embed + L2-normalise
    """
    # cv2.imread returns BGR — same byte order the Pi's capture_array and the
    # registration server's cv2.imdecode produce. Do NOT convert to RGB.
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"ERROR: failed to decode image: {image_path}")

    embedder = build_embedder(model_name)
    embedder.load()
    vector, bbox = embedder.embed_largest(image, min_face_size=min_face_size)
    if bbox is None:
        sys.exit("ERROR: no face detected in image")
    fw, fh = bbox[2], bbox[3]
    print(f"[detect] largest bbox {fw}x{fh} at ({bbox[0]},{bbox[1]})")
    if vector is None:
        print(f"[warn]  face smaller than min_face_size={min_face_size}; the Pi "
              f"would SKIP this frame")
        sys.exit(1)
    print(f"[embed]  {vector.shape[0]}-D ({model_name}), "
          f"first3=[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}]")
    return vector.tolist()


def post_identify(base_url: str, vector, n: int, model: str) -> dict:
    """POST the vector as form data — same shape IdentifyRequest produces."""
    server_model_id = _SERVER_MODEL_ID.get(model, model)
    url = base_url.rstrip("/") + "/api/v1/identify/"
    form = {
        "type": "face",
        "n": str(n),
        "model": server_model_id,
        "face_vector": ",".join(str(v) for v in vector),
    }
    data = urllib.parse.urlencode(form).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    print(f"\n[POST]   {url}  (type=face, n={n}, model={server_model_id}, dim={len(vector)})")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"ERROR: identify returned HTTP {e.code}: {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: could not reach {url} ({e.reason}). Is the sandbox up?")
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
    ap.add_argument("--model", default="sface", choices=sorted(EMBEDDERS))
    ap.add_argument("--min-face-size", type=int, default=100)
    ap.add_argument("--no-wpu", action="store_true", help="Skip the wpu/images fetch")
    args = ap.parse_args()

    vector = build_embedding(args.image, args.model, args.min_face_size)
    resp = post_identify(args.base_url, vector, args.n, args.model)

    name = resp.get("name")
    visit_id = resp.get("visit_id")
    if name:
        print(f"\n==> MATCH: {name}  (distance={resp.get('distance')}, "
              f"confidence={resp.get('confidence')}, match_type={resp.get('match_type')})")
    else:
        print(f"\n==> NO MATCH ({resp.get('message')}). "
              f"Check the person is registered in the sandbox with a {args.model} vector.")

    if visit_id and not args.no_wpu:
        urls = fetch_wpu_images(args.base_url, visit_id)
        if urls:
            check_url_reachable(urls[0])


if __name__ == "__main__":
    main()
