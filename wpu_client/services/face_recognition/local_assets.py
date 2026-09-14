"""Diagnostic-mode asset resolution — the local counterpart to asset_fetcher.

Base mode downloads each person's two alpha cutouts (SAU body, FRU face) from
the server. Diagnostic mode is offline, so the same two cutouts are read from
that person's seeded gallery folder instead:

    data/embeddings/<slug>/sketches/

Everything after this point is identical between the two modes — the same
base_composer entry points, the same data/base_scenes/ backgrounds, the same
per-person and per-pair caching. Only the source of the cutouts differs.

Which file is which
───────────────────
The server labels its two cutouts explicitly (`sau_signed_url` /
`fru_signed_url`). A hand-populated gallery folder carries no labels, so each
image's role is worked out in three passes, most explicit first:

1. `sau` / `fru` keys in the person's meta.json, each naming a file in
   sketches/. Always wins — this is the escape hatch when the guesses below
   get it wrong.
2. Filename convention: a stem containing "sau", "body" or "person" is the
   body, one containing "fru" or "face" is the face.
3. Shape. A body cutout is a full standing figure, markedly taller than it is
   wide; a face cutout is close to square. Of the images left unclaimed, the
   most elongated is taken as the body and the most square as the face.

Pass 3 is what makes existing galleries work unchanged: the seeded folders in
this repo hold files named after their source (`person.png`, `d6.png`) rather
than their role, and requiring a rename would break every provisioned device.
Dimensions are read from a 1/8-scale decode, which is cheap and preserves the
aspect ratio exactly enough for a "tall vs square" decision.

Like asset_fetcher, everything here is best-effort: a folder with one image,
no images, or three ambiguous ones returns whatever could be established and
None for the rest. The caller composes what it can and falls back otherwise.
"""

import logging
import os
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Filename stems containing one of these claim that role in pass 2.
BODY_HINTS = ("sau", "body", "person")
FACE_HINTS = ("fru", "face")

# meta.json keys that name a cutout explicitly (pass 1).
BODY_META_KEYS = ("sau", "sau_cutout", "body")
FACE_META_KEYS = ("fru", "fru_cutout", "face")

# height / width above which pass 3 is willing to call an image a body. A
# standing figure runs 2-3x taller than wide; a face cutout is ~1. The gap is
# wide enough that the exact cut-off does not matter much.
BODY_MIN_ASPECT = 1.6


def _candidate_images(sketches_dir: Path) -> list[Path]:
    """Every image file directly inside a gallery folder, sorted by name."""
    try:
        return sorted(
            p for p in sketches_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    except OSError:
        return []


def _aspect(path: Path) -> float | None:
    """height / width, from a 1/8-scale decode. None if unreadable.

    IMREAD_REDUCED_COLOR_8 decodes at an eighth of each dimension, so a
    1521x3941 body cutout costs a 190x492 decode instead of a full one. The
    ratio survives the reduction, which is all pass 3 needs.
    """
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None or img.shape[1] == 0:
        return None
    return float(img.shape[0]) / float(img.shape[1])


def _from_meta(sketches_dir: Path, meta: dict | None, keys: tuple[str, ...]) -> Path | None:
    """Pass 1: a meta.json key naming a file inside the gallery folder.

    The value is treated as a bare filename — anything with a path separator
    is rejected rather than followed, so a stray '../' in hand-edited metadata
    cannot reach outside the person's own folder.
    """
    if not meta:
        return None
    for key in keys:
        value = meta.get(key)
        if not value or not isinstance(value, str):
            continue
        if os.path.basename(value) != value:
            logger.warning(f"Ignoring meta.json {key!r}={value!r} — must be a bare filename")
            continue
        candidate = sketches_dir / value
        if candidate.is_file():
            return candidate
        logger.warning(f"meta.json {key!r} names a missing file: {candidate}")
    return None


def _from_filename(candidates: list[Path], hints: tuple[str, ...]) -> Path | None:
    """Pass 2: first candidate whose stem contains one of `hints`."""
    for path in candidates:
        stem = path.stem.lower()
        if any(hint in stem for hint in hints):
            return path
    return None


def resolve_local_cutouts(sketches_dir: str | Path, meta: dict | None = None) -> dict:
    """Locate a seeded person's SAU and FRU cutouts.

    Args:
        sketches_dir: data/embeddings/<slug>/sketches/
        meta: that person's parsed meta.json, if available — consulted for
            explicit `sau`/`fru` filenames before anything is guessed.

    Returns:
        {"sau": Path | None, "fru": Path | None}
    """
    result: dict = {"sau": None, "fru": None}
    sketches_dir = Path(sketches_dir)
    if not sketches_dir.is_dir():
        logger.info(f"No gallery sketches folder: {sketches_dir}")
        return result

    candidates = _candidate_images(sketches_dir)
    if not candidates:
        logger.info(f"No cutout images in {sketches_dir}")
        return result

    # Pass 1 — explicit metadata.
    result["sau"] = _from_meta(sketches_dir, meta, BODY_META_KEYS)
    result["fru"] = _from_meta(sketches_dir, meta, FACE_META_KEYS)

    # Pass 2 — filename convention, over whatever pass 1 did not claim.
    unclaimed = [p for p in candidates if p not in (result["sau"], result["fru"])]
    if result["sau"] is None:
        result["sau"] = _from_filename(unclaimed, BODY_HINTS)
        unclaimed = [p for p in unclaimed if p != result["sau"]]
    if result["fru"] is None:
        result["fru"] = _from_filename(unclaimed, FACE_HINTS)
        unclaimed = [p for p in unclaimed if p != result["fru"]]

    # Pass 3 — shape. Only measure if something is still missing.
    if (result["sau"] is None or result["fru"] is None) and unclaimed:
        measured = [(p, _aspect(p)) for p in unclaimed]
        measured = [(p, a) for p, a in measured if a is not None]
        if measured:
            if result["sau"] is None:
                tallest, aspect = max(measured, key=lambda item: item[1])
                if aspect >= BODY_MIN_ASPECT:
                    result["sau"] = tallest
                    measured = [(p, a) for p, a in measured if p != tallest]
            if result["fru"] is None and measured:
                squarest, aspect = min(measured, key=lambda item: abs(item[1] - 1.0))
                if aspect < BODY_MIN_ASPECT:
                    result["fru"] = squarest

    logger.info(
        f"Local cutouts in {sketches_dir}: "
        f"sau={result['sau'].name if result['sau'] else 'none'} "
        f"fru={result['fru'].name if result['fru'] else 'none'} "
        f"({len(candidates)} image(s) present)"
    )
    return result
