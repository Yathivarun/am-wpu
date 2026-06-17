#!/usr/bin/env python3
"""
Seed a face into the local diagnostic gallery (offline diagnostic mode).

Given a face image (and optionally the person's sketch images), this runs the
SAME YuNet + SFace pipeline the live service uses, then writes:

    diagnostic_gallery/<slug>/embedding.npy   # 128-D L2-normalised SFace vector
    diagnostic_gallery/<slug>/meta.json       # {"name": ...}
    diagnostic_gallery/<slug>/sketches/       # this person's face-swap sketch(es)

At runtime, `python main.py --diagnostic` matches live faces against these
entries; on a match the slideshow shows THAT person's sketches.

The sketches can be supplied now (--sketches) or dropped into the sketches/
folder later (e.g. when the face-swap render is delivered) — no re-seed needed.

Usage:
    # seed now, add the sketch later:
    python tools/seed_face.py --name "Ekan" --face /path/to/ekan.jpg
    # seed with sketches in one go:
    python tools/seed_face.py --name "Ekan" --face ekan.jpg --sketches /path/to/ekan_sketches/
"""

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

# Import the shared embedder (camera-free) from the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wpu_client.services.face_recognition.sface_embedder import SFaceEmbedder  # noqa: E402

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp")


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


def main():
    ap = argparse.ArgumentParser(description="Seed a face into the diagnostic gallery")
    ap.add_argument("--name", required=True, help="Person's display name")
    ap.add_argument("--face", required=True, help="Path to a face image of this person")
    ap.add_argument("--sketches", default=None,
                    help="Optional dir OR single image to copy into this person's sketches/ "
                         "folder (can be added later instead)")
    ap.add_argument("--gender", default=None, choices=["male", "female"],
                    help="Optional metadata only (no longer used to pick sketches)")
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
    sketches_dir = os.path.join(person_dir, "sketches")
    os.makedirs(sketches_dir, exist_ok=True)

    np.save(os.path.join(person_dir, "embedding.npy"), vector)
    meta = {"name": args.name}
    if args.gender:
        meta["gender"] = args.gender
    with open(os.path.join(person_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Optionally copy sketches now.
    copied = 0
    if args.sketches:
        srcs = []
        if os.path.isdir(args.sketches):
            for ext in IMAGE_EXTS:
                srcs.extend(glob.glob(os.path.join(args.sketches, ext)))
        elif os.path.isfile(args.sketches):
            srcs = [args.sketches]
        else:
            print(f"[warn] --sketches path not found: {args.sketches}")
        for src in srcs:
            shutil.copy2(src, sketches_dir)
            copied += 1

    print(f"[seed]  face detected at {bbox}, embedded {vector.shape[0]}-D "
          f"(first3=[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}])")
    print(f"[saved] {person_dir}/embedding.npy + meta.json  (name='{args.name}')")
    if copied:
        print(f"[saved] copied {copied} sketch(es) into {sketches_dir}/")
    else:
        print(f"[note]  sketches/ is empty — drop this person's face-swap sketch(es) into "
              f"{sketches_dir}/ when ready (no re-seed needed)")


if __name__ == "__main__":
    main()
