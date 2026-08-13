"""Base-mode scene composition — compose a person's SAU (body) and FRU
(face) alpha cutouts onto local scene backgrounds, for one or two people.

This is the on-device replacement for the server pre-rendering every
person×scene combination: the server stores one cutout pair per person, and
the Pi composes on demand. Geometry is ported from the server's
app/sketch/{sau,fru}_postprocess.py, reimplemented in cv2/numpy to match the
style of duo_composer.py (whose building blocks this module reuses) rather
than pulling PIL into the client.

SAU and FRU are different composition mechanics and are NEVER cross-composed
— each produces its own separate slides:

  SAU (body)  scale + bottom-center anchor, no rotation. `anchor` is where
              the FEET land, so the paste is centred horizontally on it and
              sits directly above it.
  FRU (face)  eye-anchored affine warp (scale + rotate + translate), the
              same transform diagnostic duo composition already uses.

Scene assets (supplied later — code must not assume they exist):
    data/base_scenes/
        sau_single/  scenes_config.json  {"<id>": {"scale": f, "anchor": [x, y]}}
        sau_duo/     scenes_config.json  {"<id>": {"scale_1": f, "anchor_1": [x, y],
                                                   "scale_2": f, "anchor_2": [x, y]}}
        fru_single/  scenes_config.json  {"<id>": {"gender": [...],
                                                   "face_anchor": {...}}}
        fru_duo/     scenes_config.json  same shape as data/duo_scenes
        <id>.png     scene backgrounds, filename stem == json key

These directories are entirely separate from data/duo_scenes/ and
data/duo_output/, which remain diagnostic-mode-only.

Every entry point returns the output directory on success or None when
nothing was composable, and logs rather than raising — a missing asset must
never take down the recognition loop.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from wpu_client.services.face_recognition.duo_composer import (
    detect_eye_kps,
    load_alpha_bgra,
    warp_and_blend_face,
)

logger = logging.getLogger(__name__)

# Composed slides are photographic full-screen backgrounds, so JPEG is a far
# better fit than PNG here: ~6x smaller on a real 2240x1918 scene with no
# visible difference at projection distance, which matters on a kiosk that
# caches per-person AND per-pair output. (Diagnostic duo_output keeps writing
# PNG — untouched.)
OUTPUT_SUFFIX = ".jpg"
JPEG_QUALITY = 90

_SCENE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Output filename prefixes. SAU and FRU slides share one flat display/ dir
# (that flatness is what lets the slideshow's existing glob-a-directory
# mechanism pick them up unmodified), so each composer's on-disk cache check
# must look only at its OWN prefix. Checking "is the dir non-empty" instead
# would make whichever composer runs second mistake the first one's output
# for its own and skip composing altogether.
BODY_PREFIX = "sau_scene_"
FACE_PREFIX = "fru_scene_"


def _load_scenes_config(scenes_config_path: Path) -> dict:
    """Read a scenes_config.json, or {} if missing/unreadable.

    Missing is the expected state until the scene assets are supplied, so
    this logs at info rather than warning.
    """
    if not scenes_config_path.exists():
        logger.info(f"Base scenes config not present yet: {scenes_config_path}")
        return {}
    try:
        with open(scenes_config_path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning(f"Unreadable scenes config {scenes_config_path}: {e}")
        return {}


def _find_scene_file(scenes_dir: Path, scene_id: str) -> Path | None:
    """Locate the background image whose stem matches a config key."""
    for ext in _SCENE_EXTENSIONS:
        candidate = scenes_dir / f"{scene_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _cache_hit(output_dir: Path, prefix: str) -> bool:
    """Whether THIS composer already has output in the shared display dir.

    Scoped to `prefix` so body and face composition cache independently even
    though they write side by side (see the *_PREFIX constants above).
    """
    try:
        if not output_dir.is_dir():
            return False
        return any(p.name.startswith(prefix) for p in output_dir.iterdir())
    except OSError:
        return False


def _write_scene(scene_bgra: np.ndarray, output_dir: Path, name: str) -> bool:
    """Write one composed scene as JPEG. Returns whether it was written."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{name}{OUTPUT_SUFFIX}"
        bgr = cv2.cvtColor(scene_bgra, cv2.COLOR_BGRA2BGR)
        return bool(
            cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        )
    except (OSError, cv2.error) as e:
        logger.warning(f"Could not write composed scene {name}: {e}")
        return False


def _gender_allows(gender: str | None, meta: dict) -> bool:
    """Whether a scene accepts this person's gender.

    Permissive when gender is unknown — mirrors compose_duo. The server's
    gender field may not be rolled out yet, and a None gender excluding every
    scene would silently hide all slides.
    """
    if not gender:
        return True
    return gender in meta.get("gender", ["male", "female", "unknown"])


def _paste_body(scene_bgra: np.ndarray, cutout_bgra: np.ndarray, scale: float, anchor) -> bool:
    """Alpha-blend a body cutout onto a scene at `anchor`, scaled by `scale`.

    `anchor` is the point where the subject's FEET land: the resized cutout is
    centred horizontally on anchor[0] and its BOTTOM edge sits at anchor[1].

    Unlike PIL's paste, cv2 has no automatic clipping — an anchor near a scene
    edge yields a negative origin or a rect overhanging the scene, which would
    raise a shape-mismatch error. The paste rect is therefore explicitly
    intersected with the scene bounds and the cutout cropped to match.

    Returns False if the cutout lands entirely outside the scene.
    """
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(scale) or scale <= 0:
        return False

    src_h, src_w = cutout_bgra.shape[:2]
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(cutout_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    try:
        anchor_x = int(round(float(anchor[0])))
        anchor_y = int(round(float(anchor[1])))
    except (TypeError, ValueError, IndexError):
        return False

    paste_x = anchor_x - new_w // 2
    paste_y = anchor_y - new_h  # anchor is the feet, so the bottom edge

    scene_h, scene_w = scene_bgra.shape[:2]

    # Intersect the paste rect with the scene, and crop the source to match.
    dst_x1, dst_y1 = max(0, paste_x), max(0, paste_y)
    dst_x2 = min(scene_w, paste_x + new_w)
    dst_y2 = min(scene_h, paste_y + new_h)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        logger.debug("Body paste fully outside scene bounds — skipping")
        return False

    src_x1, src_y1 = dst_x1 - paste_x, dst_y1 - paste_y
    src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)

    src_roi = resized[src_y1:src_y2, src_x1:src_x2]
    alpha = (src_roi[:, :, 3].astype(np.float32) / 255.0)[..., None]
    scene_roi = scene_bgra[dst_y1:dst_y2, dst_x1:dst_x2, :3].astype(np.float32)
    blended = scene_roi * (1.0 - alpha) + src_roi[:, :, :3].astype(np.float32) * alpha
    scene_bgra[dst_y1:dst_y2, dst_x1:dst_x2, :3] = blended.astype(np.uint8)
    return True


def compose_single_body(
    sau_path: Path,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose one person's body cutout onto every sau_single scene.

    No gender filtering — the server's own body crops config has none, so
    this ports what actually exists rather than inventing a rule.
    """
    if _cache_hit(output_dir, BODY_PREFIX):
        logger.info(f"Base body cache hit on disk: {output_dir}")
        return str(output_dir)

    if sau_path is None or not Path(sau_path).exists():
        logger.info("Base body compose: no SAU cutout available")
        return None

    cutout = load_alpha_bgra(Path(sau_path))
    if cutout is None:
        logger.warning(f"Base body compose: could not read cutout {sau_path}")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written = 0
    for scene_id, meta in scenes_config.items():
        scene_file = _find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue
        if not _paste_body(scene, cutout, meta.get("scale", 1.0), meta.get("anchor", (0, 0))):
            continue
        if _write_scene(scene, output_dir, f"{BODY_PREFIX}{scene_id}"):
            written += 1

    if not written:
        logger.warning("Base body compose: no scenes produced any output")
        return None
    logger.info(f"Base body compose: {written} scene(s) written to {output_dir}")
    return str(output_dir)


def compose_single_face(
    detector: cv2.FaceDetectorYN,
    fru_path: Path,
    gender: str | None,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose one person's face cutout onto every matching fru_single scene.

    Same eye-anchored warp as diagnostic duo composition, applied once with a
    single `face_anchor` instead of twice.
    """
    if _cache_hit(output_dir, FACE_PREFIX):
        logger.info(f"Base face cache hit on disk: {output_dir}")
        return str(output_dir)

    if fru_path is None or not Path(fru_path).exists():
        logger.info("Base face compose: no FRU cutout available")
        return None

    face_bgra = load_alpha_bgra(Path(fru_path))
    if face_bgra is None:
        logger.warning(f"Base face compose: could not read cutout {fru_path}")
        return None

    kps = detect_eye_kps(detector, face_bgra[:, :, :3])
    if kps is None:
        logger.warning("Base face compose: could not detect eyes in the FRU cutout")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written = 0
    for scene_id, meta in scenes_config.items():
        if not _gender_allows(gender, meta):
            continue
        anchor = meta.get("face_anchor")
        if not anchor:
            continue
        scene_file = _find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue
        try:
            warp_and_blend_face(
                scene, face_bgra, kps,
                tuple(anchor["target_eye_midpoint"]),
                anchor["target_eye_distance"],
                anchor["target_tilt_angle"],
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Base face compose: bad face_anchor for scene {scene_id}: {e}")
            continue
        if _write_scene(scene, output_dir, f"{FACE_PREFIX}{scene_id}"):
            written += 1

    if not written:
        logger.warning("Base face compose: no scenes produced any output")
        return None
    logger.info(f"Base face compose: {written} scene(s) written to {output_dir}")
    return str(output_dir)


def compose_duo_body(
    sau_a: Path,
    sau_b: Path,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose two people's body cutouts onto every sau_duo scene.

    Two-person extension of compose_single_body: same placement math per
    person, using scale_1/anchor_1 and scale_2/anchor_2 onto one shared scene
    copy. No gender filter, for the same reason as the single-body path.
    """
    if _cache_hit(output_dir, BODY_PREFIX):
        logger.info(f"Base duo body cache hit on disk: {output_dir}")
        return str(output_dir)

    if sau_a is None or sau_b is None:
        logger.info("Base duo body compose: both SAU cutouts are required")
        return None

    cutout_a = load_alpha_bgra(Path(sau_a))
    cutout_b = load_alpha_bgra(Path(sau_b))
    if cutout_a is None or cutout_b is None:
        logger.warning("Base duo body compose: could not read one or both cutouts")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written = 0
    for scene_id, meta in scenes_config.items():
        scene_file = _find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue

        placed_a = _paste_body(scene, cutout_a, meta.get("scale_1", 1.0), meta.get("anchor_1", (0, 0)))
        placed_b = _paste_body(scene, cutout_b, meta.get("scale_2", 1.0), meta.get("anchor_2", (0, 0)))
        # A duo slide showing only one of the pair would misrepresent the
        # scene — skip it and let the caller fall back to single display.
        if not (placed_a and placed_b):
            logger.debug(f"Base duo body: scene {scene_id} could not place both people")
            continue

        if _write_scene(scene, output_dir, f"{BODY_PREFIX}{scene_id}"):
            written += 1

    if not written:
        logger.warning("Base duo body compose: no scenes produced any output")
        return None
    logger.info(f"Base duo body compose: {written} scene(s) written to {output_dir}")
    return str(output_dir)


def compose_duo_face(
    detector: cv2.FaceDetectorYN,
    fru_a: Path,
    fru_b: Path,
    gender_a: str | None,
    gender_b: str | None,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose two people's face cutouts onto every matching fru_duo scene.

    Geometrically identical to diagnostic-mode `compose_duo` — same
    warp_and_blend_face call per person, same face_anchor_1/face_anchor_2
    config shape — and deliberately kept as a separate function rather than a
    re-export of it, for two reasons specific to base mode: output must land
    in the shared display/ dir under the FACE_PREFIX name (so it neither
    collides with nor is mistaken for the body slides sitting beside it), and
    it must be written as JPEG like the rest of the base cache. compose_duo
    itself stays exactly as it is, still serving diagnostic mode.
    """
    if _cache_hit(output_dir, FACE_PREFIX):
        logger.info(f"Base duo face cache hit on disk: {output_dir}")
        return str(output_dir)

    if fru_a is None or fru_b is None:
        logger.info("Base duo face compose: both FRU cutouts are required")
        return None

    face_a = load_alpha_bgra(Path(fru_a))
    face_b = load_alpha_bgra(Path(fru_b))
    if face_a is None or face_b is None:
        logger.warning("Base duo face compose: could not read one or both cutouts")
        return None

    kps_a = detect_eye_kps(detector, face_a[:, :, :3])
    kps_b = detect_eye_kps(detector, face_b[:, :, :3])
    if kps_a is None or kps_b is None:
        logger.warning("Base duo face compose: could not detect eyes in one or both cutouts")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written = 0
    for scene_id, meta in scenes_config.items():
        # Same permissive rule as compose_duo: a scene is skipped only when
        # BOTH known genders are disallowed.
        allowed = meta.get("gender", ["male", "female", "unknown"])
        if gender_a and gender_a not in allowed and gender_b and gender_b not in allowed:
            continue

        anchor_a = meta.get("face_anchor_1")
        anchor_b = meta.get("face_anchor_2")
        if anchor_a is None or anchor_b is None:
            continue

        scene_file = _find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue

        try:
            for face, kps, anchor in (
                (face_a, kps_a, anchor_a),
                (face_b, kps_b, anchor_b),
            ):
                warp_and_blend_face(
                    scene, face, kps,
                    tuple(anchor["target_eye_midpoint"]),
                    anchor["target_eye_distance"],
                    anchor["target_tilt_angle"],
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Base duo face compose: bad anchor for scene {scene_id}: {e}")
            continue

        if _write_scene(scene, output_dir, f"{FACE_PREFIX}{scene_id}"):
            written += 1

    if not written:
        logger.warning("Base duo face compose: no scenes produced any output")
        return None
    logger.info(f"Base duo face compose: {written} scene(s) written to {output_dir}")
    return str(output_dir)
