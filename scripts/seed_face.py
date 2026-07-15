#!/usr/bin/env python3
"""
Seed a face into the local diagnostic gallery (offline diagnostic mode).

Given a face image (and optionally the person's sketch images), this runs the
SAME YuNet + <model> pipeline the live service uses for each requested model,
then writes:

    data/embeddings/<slug>/embedding_<model>.npy   # per-model L2-normalised vector
    data/embeddings/<slug>/meta.json                # {"name": ..., "models": [...]}
    data/embeddings/<slug>/sketches/                # this person's face-swap sketch(es)

By default both supported models (sface, mobilenet) are seeded, so the person
matches regardless of which `model:` config.yaml picks at runtime. At runtime,
`python main.py --diagnostic` matches live faces against these entries; on a
match the slideshow shows THAT person's sketches.

The sketches can be supplied now (--sketches) or dropped into the sketches/
folder later (e.g. when the face-swap render is delivered) — no re-seed needed.

Usage:
    # seed both models now, add the sketch later:
    python scripts/seed_face.py --name "Ekan" --face /path/to/ekan.jpg
    # seed with sketches in one go:
    python scripts/seed_face.py --name "Ekan" --face ekan.jpg --sketches /path/to/ekan_sketches/
    # seed only one model:
    python scripts/seed_face.py --name "Ekan" --face ekan.jpg --models mobilenet
"""

import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

# Import the shared embedders (camera-free) from the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from wpu_client.services.face_recognition.sface_embedder import EMBEDDERS, build_embedder  # noqa: E402

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
    ap.add_argument("--gallery-dir", default="data/embeddings",
                    help="Gallery root directory (default: data/embeddings)")
    ap.add_argument("--models", default=",".join(sorted(EMBEDDERS)),
                    help=f"Comma-separated models to seed (default: all of {sorted(EMBEDDERS)})")
    ap.add_argument("--min-face-size", type=int, default=0,
                    help="Reject the seed if the detected face is smaller than this (px)")
    args = ap.parse_args()

    if not os.path.exists(args.face):
        sys.exit(f"ERROR: face image not found: {args.face}")

    image = cv2.imread(args.face, cv2.IMREAD_COLOR)  # BGR, same as live capture
    if image is None:
        sys.exit(f"ERROR: failed to decode image: {args.face}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in EMBEDDERS]
    if unknown:
        sys.exit(f"ERROR: unknown model(s) {unknown} (expected one of {sorted(EMBEDDERS)})")

    slug = slugify(args.name)
    person_dir = os.path.join(args.gallery_dir, slug)
    sketches_dir = os.path.join(person_dir, "sketches")
    os.makedirs(sketches_dir, exist_ok=True)

    seeded_models = []
    for model_name in models:
        embedder = build_embedder(model_name)
        embedder.load()
        vector, bbox = embedder.embed_largest(image, min_face_size=args.min_face_size)
        if vector is None:
            if bbox is None:
                sys.exit(f"ERROR: no face detected in the seed image (model={model_name})")
            sys.exit(
                f"ERROR: detected face {bbox[2]}x{bbox[3]} is below "
                f"--min-face-size={args.min_face_size} (model={model_name})"
            )
        np.save(os.path.join(person_dir, f"embedding_{model_name}.npy"), vector)
        seeded_models.append(model_name)
        print(f"[seed]  {model_name}: face detected at {bbox}, embedded {vector.shape[0]}-D "
              f"(first3=[{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}])")

    meta = {"name": args.name, "models": seeded_models}
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

    print(f"[saved] {person_dir}/  embedding_{{{','.join(seeded_models)}}}.npy + meta.json  "
          f"(name='{args.name}')")
    if copied:
        print(f"[saved] copied {copied} sketch(es) into {sketches_dir}/")
    else:
        print(f"[note]  sketches/ is empty — drop this person's face-swap sketch(es) into "
              f"{sketches_dir}/ when ready (no re-seed needed)")


if __name__ == "__main__":
    main()
