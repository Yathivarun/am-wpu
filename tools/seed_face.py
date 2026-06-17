#!/usr/bin/env python3
"""
Seed a face into the local diagnostic gallery (offline diagnostic mode).

Given a face image, a name, and a gender, this runs the SAME YuNet + SFace
pipeline the live service uses, then writes:

    diagnostic_gallery/<slug>/embedding.npy   # 128-D L2-normalised SFace vector
    diagnostic_gallery/<slug>/meta.json       # {"name": ..., "gender": ...}

At runtime, `python main.py --diagnostic` matches live faces against these
entries and the matched person's gender selects which face_stock_images/<gender>/
sketches the slideshow shows.

Usage:
    python tools/seed_face.py --name "Ekan" --gender male --face /path/to/ekan.jpg
    python tools/seed_face.py --name "Asha" --gender female --face asha.png --gallery-dir diagnostic_gallery
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# Import the shared embedder (camera-free) from the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wpu_client.services.face_recognition.sface_embedder import SFaceEmbedder  # noqa: E402


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description="Seed a face into the diagnostic gallery")
    ap.add_argument("--name", required=True, help="Person's display name")
    ap.add_argument("--gender", required=True, choices=["male", "female"],
                    help="Selects which face_stock_images/<gender>/ sketches are shown")
    ap.add_argument("--face", required=True, help="Path to a face image of this person")
    ap.add_argument("--gallery-dir", default="diagnostic_gallery",
                    help="Gallery root directory (default: diagnostic_gallery)")
    ap.add_argument("--min-face-size", type=int, default=0,
                    help="Reject the seed if the detected face is smaller than this (px)")
    args = ap.parse_args()

    if not os.path.exists(args.face):
        sys.exit(f"ERROR: face image not found: {args.face}")

    image = cv2.imread(args.face, cv2.IMREAD_COLOR)  # BGR, same as live capture
    if image is None:
        sys.exit(f"ERROR: failed to decode image: {args.face}")

    embedder = SFaceEmbedder()
    embedder.load()
    vector, bbox = embedder.embed_largest(image, min_face_size=args.min_face_size)
    if vector is None:
        if bbox is None:
            sys.exit("ERROR: no face detected in the seed image")
        sys.exit(f"ERROR: detected face {bbox[2]}x{bbox[3]} is below --min-face-size={args.min_face_size}")

    slug = slugify(args.name)
    person_dir = os.path.join(args.gallery_dir, slug)
    os.makedirs(person_dir, exist_ok=True)
    np.save(os.path.join(person_dir, "embedding.npy"), vector)
    with open(os.path.join(person_dir, "meta.json"), "w") as f:
        json.dump({"name": args.name, "gender": args.gender}, f, indent=2)

    print(f"[seed]  face detected at {bbox}, embedded {vector.shape[0]}-D "
          f"(first3=[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}])")
    print(f"[saved] {person_dir}/embedding.npy + meta.json  (name='{args.name}', gender={args.gender})")


if __name__ == "__main__":
    main()
