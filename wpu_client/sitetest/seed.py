"""Seed visitors into the diagnostic gallery while the unit is running.

Three sources for the face, best first:

  --from-capture  the newest capture-store frame with an unrecognised face —
                  the visitor stands in front of the running display, then this
                  runs; no camera hand-over, and it is the same camera, mount
                  and light the unit will match them under.
  --photo FILE    any image with the face in it.
  --camera        stop the running unit, grab frames, restart it.

Cutouts: --sau/--fru files, or --proxy-from <slug|auto> to borrow another
seeded person's cutouts (there is no sketch server on site; a proxy exercises
the whole display path with stand-in art and is marked as such in meta.json).
--scenes DIR copies pre-composed scenes for scene_source=precomposed tests.

The running app re-reads the gallery within diagnostic_gallery_reload_s, so a
new person is recognised without a restart.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from wpu_client.services.face_recognition.capture_store import crop_quality, face_crop
from wpu_client.services.face_recognition.diagnostic_gallery import DiagnosticGallery
from wpu_client.services.face_recognition.local_assets import resolve_local_cutouts
from wpu_client.services.face_recognition.sface_embedder import EMBEDDERS, build_embedder
from wpu_client.sitetest import system

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_").lower()


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ── where the face comes from ───────────────────────────────────────────


def from_capture(within_s: float, captures_root: Path) -> tuple[np.ndarray, list, dict]:
    """Newest capture with an unrecognised face: (frame, bbox, record)."""
    records = system.latest_captures(within_s, captures_root)
    for rec in reversed(records):
        unknown = [f for f in rec.get("faces", []) if f.get("status") == "unrecognized"]
        if not unknown or not rec.get("frame"):
            continue
        face = max(unknown, key=lambda f: f["size"])
        path = Path(rec["_root"]) / rec["dir"] / rec["frame"]
        frame = cv2.imread(str(path))
        if frame is not None:
            return frame, face["bbox"], {"capture": str(path), "t": rec["t"], "size": face["size"]}
    raise SystemExit(
        f"No capture with an unrecognised face in the last {within_s:.0f} s. "
        "Have the visitor face the display (app running with captures on) and retry, "
        "or use --photo / --camera."
    )


def from_camera(frames: int = 8) -> tuple[np.ndarray, None, dict]:
    """Borrow the camera from the running unit, keep the frame with the largest face."""
    unit = system.active_unit()
    if unit:
        system.sudo(["systemctl", "stop", unit])
    try:
        from picamera2 import Picamera2

        from wpu_client.services.face_recognition.face_service import CAMERA_MAIN_FORMAT, CAMERA_MAIN_SIZE

        cam = Picamera2()
        cam.configure(
            cam.create_preview_configuration(main={"size": CAMERA_MAIN_SIZE, "format": CAMERA_MAIN_FORMAT})
        )
        cam.start()
        try:
            cam.set_controls({"ExposureValue": 0.2})
            time.sleep(1.5)  # let AE settle
            embedder = build_embedder("sface")
            embedder.load()
            best, best_area = None, 0
            for _ in range(frames):
                frame = cam.capture_array("main")
                rows = embedder.detect_faces(frame)
                if rows:
                    area = max(float(r[2]) * float(r[3]) for r in rows)
                    if area > best_area:
                        best, best_area = frame.copy(), area
                time.sleep(0.3)
        finally:
            cam.stop()
            cam.close()
    finally:
        if unit:
            system.sudo(["systemctl", "start", unit])
    if best is None:
        raise SystemExit("No face seen by the camera — stand ~1 m away facing it and retry.")
    return best, None, {"camera": True, "unit_restarted": unit}


def from_photo(path: str) -> tuple[np.ndarray, None, dict]:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Cannot read image: {path}")
    return frame, None, {"photo": str(Path(path).resolve())}


# ── embedding ───────────────────────────────────────────────────────────


def embed_all(frame: np.ndarray, bbox: Optional[list], min_face: int) -> tuple[dict, list]:
    """model -> vector for every model, of the face at `bbox` (or the largest)."""
    vectors, used = {}, None
    for model in sorted(EMBEDDERS):
        embedder = build_embedder(model)
        embedder.load()
        rows = embedder.detect_faces(frame)
        if not rows:
            raise SystemExit("No face found in the seed image.")
        if bbox is not None:
            row = max(rows, key=lambda r: _iou([float(v) for v in r[:4]], bbox))
        else:
            row = max(rows, key=lambda r: float(r[2]) * float(r[3]))
        x, y, w, h = (int(v) for v in row[:4])
        if min(w, h) < min_face:
            raise SystemExit(f"Face is {w}x{h} px, below --min-face {min_face}. Move closer and retry.")
        vectors[model] = embedder.embed_row(frame, row)
        used = [x, y, w, h]
    return vectors, used


# ── cutouts ─────────────────────────────────────────────────────────────


def proxy_source(gallery: DiagnosticGallery, spec: str, gender: Optional[str], exclude: str):
    """The gallery entry whose cutouts a new visitor borrows."""
    entries = [e for e in gallery.entries() if e.slug != exclude]
    if spec != "auto":
        entry = gallery.get_entry(slugify(spec))
        if entry is None:
            raise SystemExit(f"--proxy-from: no seeded person '{spec}'")
        return entry
    complete = [e for e in entries if all(resolve_local_cutouts(e.sketch_dir, e.meta).values())]
    same = [e for e in complete if gender and e.gender == gender]
    pick = (same or complete or [None])[0]
    if pick is None:
        raise SystemExit("--proxy-from auto: no seeded person has both cutouts to borrow")
    return pick


def seed(
    name: str,
    gender: Optional[str],
    height_cm: Optional[int],
    source: str,
    *,
    photo: Optional[str] = None,
    within_s: float = 120,
    sau: Optional[str] = None,
    fru: Optional[str] = None,
    proxy_from: Optional[str] = None,
    scenes: Optional[str] = None,
    force: bool = False,
    min_face: int = 80,
    threshold: float = 0.5,
) -> dict:
    """Seed one person. Returns the details for the result file."""
    gdir = system.gallery_dir()
    slug = slugify(name)
    person = gdir / slug
    if person.exists() and not force:
        raise SystemExit(f"'{slug}' is already seeded — pass --force to re-seed")

    if source == "capture":
        frame, bbox, origin = from_capture(within_s, system.captures_dir())
    elif source == "camera":
        frame, bbox, origin = from_camera()
    else:
        frame, bbox, origin = from_photo(photo)
    vectors, used = embed_all(frame, bbox, min_face)
    crop = face_crop(frame, used, 0.25)
    quality = crop_quality(crop) if crop is not None else {}

    gallery = DiagnosticGallery(str(gdir))
    gallery.load()
    # Who would this person be mistaken for? Closest *other* seeded person.
    confusable = None
    for e in gallery.entries():
        if e.slug == slug or "mobilenet" not in e.vectors:
            continue
        d = float(1 - np.dot(vectors["mobilenet"], e.vectors["mobilenet"]))
        if confusable is None or d < confusable["distance"]:
            confusable = {"slug": e.slug, "name": e.name, "distance": round(d, 4)}

    proxy = proxy_source(gallery, proxy_from, gender, slug) if proxy_from else None

    staging = gdir.parent / f".seeding_{slug}"  # same filesystem, outside the gallery
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "sketches").mkdir(parents=True)
    for model, vec in vectors.items():
        np.save(staging / f"embedding_{model}.npy", vec.astype(np.float32))
    cv2.imwrite(str(staging / "seed_source.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if crop is not None:
        cv2.imwrite(str(staging / "seed_face.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    meta = {
        "name": name,
        "models": sorted(vectors),
        "seeded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_source": origin,
    }
    if gender:
        meta["gender"] = gender
    if height_cm:
        meta["height_cm"] = int(height_cm)
    cut = {"sau": sau, "fru": fru}
    if proxy is not None:
        borrowed = resolve_local_cutouts(proxy.sketch_dir, proxy.meta)
        cut = {k: cut[k] or (str(v) if v else None) for k, v in borrowed.items()}
        meta["proxy_of"] = proxy.slug
        proxy_scenes = Path(proxy.sketch_dir).parent / "scenes"
        if not scenes and proxy_scenes.is_dir():
            scenes = str(proxy_scenes)
    for key in ("sau", "fru"):
        if cut[key]:
            src = Path(cut[key])
            if not src.is_file():
                raise SystemExit(f"--{key}: no such file {src}")
            dest = f"{key}_{src.name}"
            shutil.copy2(src, staging / "sketches" / dest)
            meta[key] = dest
    if scenes:
        sdir = Path(scenes)
        files = [p for p in sdir.iterdir() if p.suffix.lower() in IMAGE_EXTS] if sdir.is_dir() else []
        if files:
            (staging / "scenes").mkdir()
            for p in files:
                shutil.copy2(p, staging / "scenes" / p.name)
        meta["scenes"] = len(files)
    (staging / "meta.json").write_text(json.dumps(meta, indent=2))

    # Built outside the gallery and moved in whole, so the running app never
    # loads half a person.
    gdir.mkdir(parents=True, exist_ok=True)
    if person.exists():
        shutil.rmtree(person)
    staging.rename(person)

    return {
        "slug": slug,
        "dir": str(person),
        "face_px": used,
        "face_quality": quality,
        "origin": origin,
        "cutouts": {k: meta.get(k) for k in ("sau", "fru")},
        "proxy_of": meta.get("proxy_of"),
        "scenes": meta.get("scenes", 0),
        "closest_other": confusable,
        "confusable": bool(confusable and confusable["distance"] < threshold),
    }
