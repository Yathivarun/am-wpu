"""Base-mode scene composition — compose a person's SAU (body) and FRU
(face) alpha cutouts onto local scene backgrounds, for one or two people.

This is the on-device replacement for the server pre-rendering every
person×scene combination: the server stores one cutout pair per person, and
the Pi composes on demand. The placement is a cv2/numpy port of the server
pipeline's own scene composition (am-ml-fullbody-sketch dev/pipeline/
{sau,fru}_postprocess.py `_compose_*_single_scene`), including the fixes made
there — so a slide composed here matches the one the server would render:

  SAU (body)  Height-aware scaling of the subject's CANVAS (the postprocessed
              sketch, padding included): the canvas is sized to
              ``scale x 3941 px x (height_cm / ref_height_cm)`` whatever the
              photo's resolution, and its BOTTOM edge sits on `anchor` (the
              feet line). Soft mask edge (dilate + blur), soft ground shadow,
              optional outline ring.
  FRU (face)  Eye-anchored affine warp (scale + rotate + translate), sized by
              head height when the scene provides one and eye distance
              otherwise, with a smooth outline ring drawn at on-scene scale.

The geometry both need — the canvas a SAU cutout came from, the face
landmarks and face box of a FRU cutout — is embedded in the cutout PNG by the
server (see png_meta.py). Cutouts without it (older server builds, hand-made
gallery cutouts) still compose: the SAU canvas is estimated from the
pipeline's 5% content padding, and FRU eyes fall back to YuNet detection.

SAU and FRU are different composition mechanics and are NEVER cross-composed
— each produces its own separate slides.

Scene assets (supplied separately — code must not assume they exist):
    data/base_scenes/
        sau_single/  scenes_config.json  {"<id>": {"scale": f, "anchor": [x, y],
                                                   "height_cm": f?}}
        sau_duo/     scenes_config.json  {"<id>": {"scale_1": f, "anchor_1": [x, y],
                                                   "scale_2": f, "anchor_2": [x, y],
                                                   "height_cm": f?}}
        fru_single/  scenes_config.json  {"<id>": {"gender": [...],
                                                   "face_anchor": {...}}}
        fru_duo/     scenes_config.json  {"<id>": {"gender": [...],
                                                   "face_anchor_1": {...},
                                                   "face_anchor_2": {...}}}
        <id>.png     scene backgrounds, filename stem == json key

`height_cm` on a body scene is the reference height its `scale` was
calibrated for (default 170); `face_anchor` may carry `target_head_height`.

Every entry point returns the output directory on success or None when
nothing was composable, and logs rather than raising — a missing asset must
never take down the recognition loop.
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wpu_client.services.face_recognition.png_meta import read_geometry

logger = logging.getLogger(__name__)

# Composed slides are photographic full-screen backgrounds, so JPEG is a far
# better fit than PNG here: ~6x smaller on a real 2240x1918 scene with no
# visible difference at projection distance, which matters on a kiosk that
# caches per-person AND per-pair output.
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

# Bump whenever composition output changes for the same inputs, so slides
# already cached on deployed units are recomposed instead of shown stale.
COMPOSE_VERSION = 2


# ─────────────────────────────────────────────────────────────────────────
# Tuning — mirrors the server pipeline's config.py (am-ml-fullbody-sketch
# dev/config.py). Keep in step with it: these are what make a Pi-composed
# slide match a server-composed one.
# ─────────────────────────────────────────────────────────────────────────

# The reference cutout the body `scale` values were calibrated against:
# a person of SAU_REFERENCE_HEIGHT_CM rendered at this canvas pixel height.
SAU_REFERENCE_HEIGHT_CM = 170.0
SAU_REFERENCE_CANVAS_HEIGHT_PX = 3941

# Preprocessing pads the subject's content box by this fraction per side to
# form the canvas. Only used to estimate the canvas for cutouts that carry no
# embedded geometry.
SAU_CANVAS_PADDING_PCT = 0.05

# Soft edge on the body mask: dilate (elliptical kernel) then Gaussian blur,
# applied at the cutout's native resolution before scaling.
SAU_MASK_DILATE_PX = 5
SAU_MASK_BLUR_PX = 11

SAU_SHADOW_PARAMS = {
    "enabled": True,
    "width_ratio": 0.6,       # ellipse width  = ratio x scaled canvas width
    "height_ratio": 0.6,      # ellipse height = ratio x scaled canvas width
    "opacity": 0.40,
    "blur_radius": 25,        # Gaussian sigma, px
    "vertical_offset": -30,   # px; negative lifts the ellipse centre above the feet line
}

SAU_OUTLINE_PARAMS = {
    "enabled": False,
    "thickness": 6,
    "color": (10, 10, 10),    # RGB
    "opacity": 1.0,
    "blur_radius": 1.5,
    "threshold": 240,
}

FRU_OUTLINE_PARAMS = {
    "enabled": True,
    "thickness": 4,
    "color": (10, 10, 10),    # RGB
    "opacity": 1.0,
    "blur_radius": 1.5,
    "threshold": 240,
}


# ─────────────────────────────────────────────────────────────────────────
# Cutouts
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Cutout:
    """A decoded cutout plus whatever geometry the server embedded in it."""

    bgra: np.ndarray
    geometry: dict | None = None


def load_alpha_bgra(path: Path) -> np.ndarray | None:
    """Read an image as 4-channel BGRA, whatever it was stored as."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def load_cutout(path: Path | None, modality: str) -> Cutout | None:
    if path is None or not Path(path).exists():
        return None
    bgra = load_alpha_bgra(Path(path))
    if bgra is None:
        return None
    geometry = read_geometry(path)
    if geometry is not None and geometry.get("modality") != modality:
        logger.warning(f"{path}: embedded geometry is for {geometry.get('modality')!r}, "
                       f"not {modality!r} — ignoring it")
        geometry = None
    return Cutout(bgra, geometry)


def detect_eye_kps(detector: cv2.FaceDetectorYN, face_bgr: np.ndarray) -> np.ndarray | None:
    """Row layout: [x,y,w,h, 5x(lm_x,lm_y), score]; first two landmarks = eyes."""
    h, w = face_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(face_bgr)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: float(row[2]) * float(row[3]))
    return np.asarray(best[4:8], dtype=np.float32).reshape(2, 2)


def face_kps(cutout: Cutout, detector: cv2.FaceDetectorYN | None) -> np.ndarray | None:
    """The two eye points (image-left first) in cutout pixels.

    Prefers the server's landmarks — mapped from the source photo, so always
    present when the server produced the cutout — and only runs YuNet on the
    sketch for cutouts that carry none.
    """
    kps = (cutout.geometry or {}).get("kps")
    if kps:
        try:
            arr = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
            if len(arr) >= 2 and np.isfinite(arr[:2]).all():
                return arr[:2]
        except (TypeError, ValueError):
            pass
    if detector is None:
        return None
    return detect_eye_kps(detector, np.ascontiguousarray(cutout.bgra[:, :, :3]))


def _face_bbox(cutout: Cutout) -> tuple[float, float, float, float] | None:
    box = (cutout.geometry or {}).get("face_bbox")
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if y2 > y1 and x2 > x1 else None


def _body_canvas(cutout: Cutout) -> tuple[int, int, int, int]:
    """(canvas_w, canvas_h, cutout_x, cutout_y): the canvas the server's
    compositor scales and anchors, and where the cutout sits inside it."""
    h, w = cutout.bgra.shape[:2]
    g = cutout.geometry or {}
    try:
        cw, ch = (int(v) for v in g["canvas_size"])
        x1, y1, x2, y2 = (int(v) for v in g["cutout_box"])
        if cw > 0 and ch > 0 and (x2 - x1, y2 - y1) == (w, h):
            return cw, ch, x1, y1
        logger.warning("Embedded SAU geometry doesn't match the cutout — estimating the canvas")
    except (KeyError, TypeError, ValueError):
        pass
    pad_w, pad_h = int(w * SAU_CANVAS_PADDING_PCT), int(h * SAU_CANVAS_PADDING_PCT)
    return w + 2 * pad_w, h + 2 * pad_h, pad_w, pad_h


# ─────────────────────────────────────────────────────────────────────────
# Pixel primitives
# ─────────────────────────────────────────────────────────────────────────


def _blend(scene_bgra: np.ndarray, bgr: np.ndarray, alpha: np.ndarray, x: int, y: int) -> bool:
    """Alpha-blend `bgr` (float/uint8 HxWx3) with `alpha` (float 0-1, HxW) onto
    the scene at (x, y), clipped to the scene. Returns False if nothing overlaps.

    cv2 has no PIL-style auto-clipping paste — an origin off the scene edge
    would raise a shape mismatch — so the rect is intersected by hand.
    """
    sh, sw = scene_bgra.shape[:2]
    h, w = alpha.shape[:2]
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(sw, x + w), min(sh, y + h)
    if dx1 >= dx2 or dy1 >= dy2:
        return False
    sx1, sy1 = dx1 - x, dy1 - y
    sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
    a = alpha[sy1:sy2, sx1:sx2, None]
    roi = scene_bgra[dy1:dy2, dx1:dx2, :3].astype(np.float32)
    src = bgr[sy1:sy2, sx1:sx2].astype(np.float32)
    scene_bgra[dy1:dy2, dx1:dx2, :3] = (roi * (1.0 - a) + src * a + 0.5).astype(np.uint8)
    return True


def _outline_ring(alpha_u8: np.ndarray, params: dict) -> np.ndarray | None:
    """Ring alpha (float 0-1) around a mask — port of the server's
    build_outline_layer: threshold, dilate, subtract, blur, opacity."""
    if not params.get("enabled", True) or alpha_u8.size == 0:
        return None
    opacity = float(params.get("opacity", 1.0))
    if opacity <= 0:
        return None
    thickness = max(1, int(params.get("thickness", 3)))
    binary = (alpha_u8 > int(params.get("threshold", 20))).astype(np.uint8) * 255
    if not binary.any():
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))
    ring = cv2.subtract(cv2.dilate(binary, kernel), binary)
    blur = float(params.get("blur_radius", 0))
    if blur > 0:
        k = int(round(blur))
        k = max(1, k + 1 if k % 2 == 0 else k)
        ring = cv2.GaussianBlur(ring, (k, k), 0)
    return np.clip(ring.astype(np.float32) * opacity / 255.0, 0.0, 1.0)


def _paint(scene_bgra: np.ndarray, ring: np.ndarray, color_rgb, x: int, y: int) -> None:
    """Blend a solid colour through `ring` alpha at (x, y)."""
    r, g, b = color_rgb
    solid = np.empty((*ring.shape, 3), np.float32)
    solid[...] = (b, g, r)
    _blend(scene_bgra, solid, ring, x, y)


def _draw_ground_shadow(scene_bgra: np.ndarray, center: tuple[float, float],
                        width: float, height: float, params: dict) -> None:
    """Soft elliptical contact shadow — port of the server's PIL ellipse +
    GaussianBlur(radius) + opacity, rendered only over its own neighbourhood."""
    if not params.get("enabled", True) or width < 1 or height < 1:
        return
    sigma = float(params.get("blur_radius", 20))
    margin = int(math.ceil(3 * sigma)) + 2
    cx, cy = center
    x1 = int(math.floor(cx - width / 2)) - margin
    y1 = int(math.floor(cy - height / 2)) - margin
    x2 = int(math.ceil(cx + width / 2)) + margin
    y2 = int(math.ceil(cy + height / 2)) + margin
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.ellipse(mask, (int(round(cx - x1)), int(round(cy - y1))),
                (max(1, int(round(width / 2))), max(1, int(round(height / 2)))),
                0, 0, 360, 255, -1, cv2.LINE_AA)
    if sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    shadow = mask.astype(np.float32) / 255.0 * float(params.get("opacity", 0.4))
    _blend(scene_bgra, np.zeros((*shadow.shape, 3), np.float32), shadow, x1, y1)


def _valid_height(height_cm) -> float | None:
    try:
        h = float(height_cm)
    except (TypeError, ValueError):
        return None
    return h if math.isfinite(h) and 50.0 <= h <= 272.0 else None


# ─────────────────────────────────────────────────────────────────────────
# Placement — one person onto one scene
# ─────────────────────────────────────────────────────────────────────────


def place_body(scene_bgra: np.ndarray, cutout: Cutout, scale, anchor,
               height_cm=None, ref_height_cm=None) -> bool:
    """Place one body cutout onto `scene_bgra` in place.

    Port of the server's _compose_sau_single_scene. The subject's CANVAS is
    sized so its height is ``scale x SAU_REFERENCE_CANVAS_HEIGHT_PX x
    (height_cm / ref_height_cm)`` — the photo's own resolution and crop cancel
    out — centred on anchor[0] with its bottom edge on anchor[1]. Unknown or
    implausible heights fall back to the reference height.

    Returns False (scene untouched) for a malformed scale/anchor or a subject
    landing entirely off-scene.
    """
    try:
        scale = float(scale)
        ax, ay = float(anchor[0]), float(anchor[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not (math.isfinite(scale) and scale > 0 and math.isfinite(ax) and math.isfinite(ay)):
        return False
    ref = _valid_height(ref_height_cm) or SAU_REFERENCE_HEIGHT_CM
    person = _valid_height(height_cm) or ref

    canvas_w, canvas_h, box_x, box_y = _body_canvas(cutout)
    eff = scale * SAU_REFERENCE_CANVAS_HEIGHT_PX * (person / ref) / canvas_h
    resized_w, resized_h = int(canvas_w * eff), int(canvas_h * eff)
    if resized_w < 1 or resized_h < 1:
        return False
    paste_x = int(ax - resized_w / 2)
    paste_y = int(ay - resized_h)

    sh, sw = scene_bgra.shape[:2]
    if paste_x >= sw or paste_y >= sh or paste_x + resized_w <= 0 or paste_y + resized_h <= 0:
        logger.debug("Body paste fully outside scene bounds — skipping")
        return False

    # Rebuild the full canvas and resize IT to (resized_w, resized_h), exactly
    # as the server does: resizing only the subject region lands its pixel
    # grid up to half a pixel off, which on fine pencil lines is visible.
    # Colour is replicated outward from the cutout so the soft mask fringe
    # never pulls in black.
    h, w = cutout.bgra.shape[:2]
    top, bottom = box_y, canvas_h - box_y - h
    left, right = box_x, canvas_w - box_x - w
    bgr = cv2.copyMakeBorder(np.ascontiguousarray(cutout.bgra[:, :, :3]),
                             top, bottom, left, right, cv2.BORDER_REPLICATE)
    alpha = cv2.copyMakeBorder(np.ascontiguousarray(cutout.bgra[:, :, 3]),
                               top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # Soft mask edge at native resolution, like the server.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SAU_MASK_DILATE_PX, SAU_MASK_DILATE_PX))
    alpha = cv2.GaussianBlur(cv2.dilate(alpha, kernel), (SAU_MASK_BLUR_PX, SAU_MASK_BLUR_PX), 0)

    interp = cv2.INTER_AREA if eff < 1 else cv2.INTER_LANCZOS4
    bgr = cv2.resize(bgr, (resized_w, resized_h), interpolation=interp)
    alpha = cv2.resize(alpha, (resized_w, resized_h), interpolation=interp)
    x, y = paste_x, paste_y

    _draw_ground_shadow(
        scene_bgra,
        (ax, ay + SAU_SHADOW_PARAMS.get("vertical_offset", 0)),
        resized_w * SAU_SHADOW_PARAMS.get("width_ratio", 0.5),
        resized_w * SAU_SHADOW_PARAMS.get("height_ratio", 0.12),
        SAU_SHADOW_PARAMS,
    )
    ring = _outline_ring(alpha, SAU_OUTLINE_PARAMS)
    if ring is not None:
        _paint(scene_bgra, ring, SAU_OUTLINE_PARAMS.get("color", (20, 20, 20)), x, y)
    return _blend(scene_bgra, bgr, alpha.astype(np.float32) / 255.0, x, y)


def place_face(scene_bgra: np.ndarray, cutout: Cutout, kps: np.ndarray, anchor: dict) -> bool:
    """Place one face cutout onto `scene_bgra` in place, eye-anchored.

    Port of the server's _compose_fru_single_scene as one affine warp: scale
    (by head height when the anchor has `target_head_height` and the cutout
    carries its face box, eye distance otherwise), rotate to the target tilt
    about the eye midpoint, translate that midpoint onto the target. The
    outline ring is built from the warped mask, i.e. at on-scene scale.

    Raises KeyError/TypeError/ValueError on a malformed anchor (callers log
    and skip the scene).
    """
    tgt_x, tgt_y = (float(v) for v in anchor["target_eye_midpoint"])
    target_eye_distance = float(anchor["target_eye_distance"])
    target_tilt = float(anchor["target_tilt_angle"])
    target_head = anchor.get("target_head_height")

    left, right = kps[0].astype(np.float64), kps[1].astype(np.float64)
    eye_dist = max(1.0, float(np.linalg.norm(right - left)))
    mid = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    face_angle = math.degrees(math.atan2(-(right[1] - left[1]), right[0] - left[0]))

    scale = None
    bbox = _face_bbox(cutout)
    if bbox is not None and target_head is not None and float(target_head) > 0:
        head_h = bbox[3] - bbox[1]
        if head_h > 1.0:
            scale = float(target_head) / head_h
    if scale is None:
        scale = target_eye_distance / eye_dist

    # Shrinking is done by an antialiased resize first, then the warp only
    # rotates + translates — as the server does (LANCZOS resize, then rotate).
    # A single warpAffine that also downscales samples without antialiasing,
    # which aliases fine pencil lines into moiré on small scene faces.
    src = cutout.bgra
    warp_scale = scale
    if scale < 1.0:
        h0, w0 = src.shape[:2]
        new_w, new_h = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
        src = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
        sx, sy = new_w / w0, new_h / h0
        mid = (mid[0] * sx, mid[1] * sy)
        warp_scale = 1.0

    M = cv2.getRotationMatrix2D(center=mid, angle=target_tilt - face_angle, scale=warp_scale)
    M[0, 2] += tgt_x - mid[0]
    M[1, 2] += tgt_y - mid[1]

    # Warp into just the face's on-scene footprint (plus room for the ring)
    # rather than a full-scene canvas — the scenes are ~2300x2700.
    h, w = src.shape[:2]
    corners = cv2.transform(np.float32([[[0, 0], [w, 0], [0, h], [w, h]]]), M)[0]
    pad = (int(FRU_OUTLINE_PARAMS.get("thickness", 4))
           + int(math.ceil(FRU_OUTLINE_PARAMS.get("blur_radius", 0))) + 2)
    sh, sw = scene_bgra.shape[:2]
    x1 = max(0, int(math.floor(corners[:, 0].min())) - pad)
    y1 = max(0, int(math.floor(corners[:, 1].min())) - pad)
    x2 = min(sw, int(math.ceil(corners[:, 0].max())) + pad)
    y2 = min(sh, int(math.ceil(corners[:, 1].max())) + pad)
    if x1 >= x2 or y1 >= y2:
        return False
    M[0, 2] -= x1
    M[1, 2] -= y1
    warped = cv2.warpAffine(src, M, (x2 - x1, y2 - y1), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    alpha_u8 = warped[:, :, 3]
    if not alpha_u8.any():
        return False

    if not (cutout.geometry or {}).get("outline_baked", False):
        ring = _outline_ring(alpha_u8, FRU_OUTLINE_PARAMS)
        if ring is not None:
            _paint(scene_bgra, ring, FRU_OUTLINE_PARAMS.get("color", (20, 20, 20)), x1, y1)
    return _blend(scene_bgra, warped[:, :, :3], alpha_u8.astype(np.float32) / 255.0, x1, y1)


# ─────────────────────────────────────────────────────────────────────────
# Scene configs, naming, caching
# ─────────────────────────────────────────────────────────────────────────


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


def find_scene_file(scenes_dir: Path, scene_id: str) -> Path | None:
    """Locate the background image whose stem matches a config key."""
    for ext in _SCENE_EXTENSIONS:
        candidate = scenes_dir / f"{scene_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def derive_scene_name(image_path: Path | str) -> str:
    """A readable scene name from its background's filename.

    The stem, with separators turned into spaces and words capitalised:
    `winter_market.png` -> "Winter Market". A numeric filename has nothing to
    humanise and comes back as-is ("4"), which is what the shipped scenes look
    like today; the value shows up once backgrounds get descriptive names.

    Kept as a pure filename transform so `name` can always be regenerated from
    what is on disk — see scripts/sync_scene_names.py.
    """
    stem = Path(image_path).stem
    words = [w for w in stem.replace("_", " ").replace("-", " ").split() if w]
    return " ".join(w.capitalize() if w.isalpha() else w for w in words) or stem


def scene_label(scene_id: str, meta: dict) -> str:
    """How a scene is named in logs: its `name`, falling back to its id."""
    name = meta.get("name")
    return f"{scene_id} ({name})" if name and name != scene_id else scene_id


def _stamp_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f".{prefix}stamp.json"


def _cache_hit(output_dir: Path, prefix: str, stamp: dict | None = None) -> bool:
    """Whether THIS composer already has current output in the shared display dir.

    Scoped to `prefix` so body and face composition cache independently even
    though they write side by side (see the *_PREFIX constants above).

    Output is only current if its stamp matches: the composition version plus
    any input that changes the pixels but not the cutout (a visitor's height).
    Stale output — from an older composer or another height — is deleted so
    it can never be shown, and the caller recomposes.
    """
    try:
        if not output_dir.is_dir():
            return False
        existing = [p for p in output_dir.iterdir() if p.name.startswith(prefix)]
        if not existing:
            return False
        expected = {"v": COMPOSE_VERSION, **(stamp or {})}
        try:
            current = json.loads(_stamp_path(output_dir, prefix).read_text())
        except (OSError, ValueError):
            current = None
        if current == expected:
            return True
        logger.info(f"Discarding stale {prefix}* slides in {output_dir} "
                    f"(stamp {current} != {expected})")
        for p in existing:
            p.unlink(missing_ok=True)
        _stamp_path(output_dir, prefix).unlink(missing_ok=True)
        return False
    except OSError:
        return False


def _write_stamp(output_dir: Path, prefix: str, stamp: dict | None = None) -> None:
    try:
        _stamp_path(output_dir, prefix).write_text(json.dumps({"v": COMPOSE_VERSION, **(stamp or {})}))
    except OSError as e:
        logger.warning(f"Could not write composition stamp in {output_dir}: {e}")


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
    """Whether a single-person scene accepts this person's gender.

    Permissive when gender is unknown: the server's gender field may not be
    rolled out yet, and a None gender excluding every scene would silently
    hide all slides.

    A scene listing only "mixed" is a duo scene and matches nobody here — one
    person is never a mixed pair — which is the intended outcome.
    """
    if not gender:
        return True
    return gender in meta.get("gender", ["male", "female", "unknown"])


def _duo_gender_allows(gender_a: str | None, gender_b: str | None, meta: dict) -> bool:
    """Whether a duo scene accepts this PAIR.

    A duo scene is art with two people posed in it, so what it constrains is
    the combination, not each person independently:

        ["male"]            two men
        ["female"]          two women
        ["male", "female"]  two men OR two women — same-gender either way
        ["mixed"]           one man and one woman
        ["male", "mixed"]   two men OR a mixed pair (lists combine)

    "mixed" is deliberately NOT the same as listing both genders. Art staged
    for two men rarely works for a man and a woman, so ["male", "female"]
    keeps its established meaning ("either same-gender pairing") and mixed
    pairs need their own opt-in.

    Same permissive fallback as the single-person path: an absent gender key,
    or a pair where either gender is unknown, matches everything rather than
    silently leaving a visitor with no slides.
    """
    allowed = meta.get("gender")
    if not allowed:
        return True
    if not gender_a or not gender_b:
        return True
    if gender_a != gender_b:
        return "mixed" in allowed
    return gender_a in allowed


def _height_stamp(*heights) -> dict:
    return {"height_cm": [_valid_height(h) for h in heights]}


# ─────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────


def compose_single_body(
    sau_path: Path,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
    height_cm: float | None = None,
) -> str | None:
    """Compose one person's body cutout onto every sau_single scene.

    `height_cm` is the visitor's real height; None (or implausible) renders
    them at each scene's reference height. No gender filtering — the server's
    own body crops config has none, so this ports what actually exists rather
    than inventing a rule.
    """
    stamp = _height_stamp(height_cm)
    if _cache_hit(output_dir, BODY_PREFIX, stamp):
        logger.info(f"Base body cache hit on disk: {output_dir}")
        return str(output_dir)

    cutout = load_cutout(sau_path, "sau")
    if cutout is None:
        logger.info("Base body compose: no readable SAU cutout available")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written: list[str] = []
    for scene_id, meta in scenes_config.items():
        scene_file = find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue
        if not place_body(scene, cutout, meta.get("scale", 1.0), meta.get("anchor", (0, 0)),
                          height_cm, meta.get("height_cm")):
            continue
        if _write_scene(scene, output_dir, f"{BODY_PREFIX}{scene_id}"):
            written.append(scene_label(scene_id, meta))

    if not written:
        logger.warning("Base body compose: no scenes produced any output")
        return None
    _write_stamp(output_dir, BODY_PREFIX, stamp)
    logger.info(
        f"Base body compose: {len(written)} scene(s) -> {output_dir} [{', '.join(written)}]"
    )
    return str(output_dir)


def compose_single_face(
    detector: cv2.FaceDetectorYN | None,
    fru_path: Path,
    gender: str | None,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose one person's face cutout onto every matching fru_single scene.

    `detector` is only used for cutouts that carry no embedded landmarks.
    """
    if _cache_hit(output_dir, FACE_PREFIX):
        logger.info(f"Base face cache hit on disk: {output_dir}")
        return str(output_dir)

    cutout = load_cutout(fru_path, "fru")
    if cutout is None:
        logger.info("Base face compose: no readable FRU cutout available")
        return None

    kps = face_kps(cutout, detector)
    if kps is None:
        logger.warning("Base face compose: no eye landmarks for the FRU cutout")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written: list[str] = []
    for scene_id, meta in scenes_config.items():
        if not _gender_allows(gender, meta):
            continue
        anchor = meta.get("face_anchor")
        if not anchor:
            continue
        scene_file = find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue
        try:
            if not place_face(scene, cutout, kps, anchor):
                continue
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Base face compose: bad face_anchor for scene {scene_id}: {e}")
            continue
        if _write_scene(scene, output_dir, f"{FACE_PREFIX}{scene_id}"):
            written.append(scene_label(scene_id, meta))

    if not written:
        logger.warning("Base face compose: no scenes produced any output")
        return None
    _write_stamp(output_dir, FACE_PREFIX)
    logger.info(
        f"Base face compose: {len(written)} scene(s) -> {output_dir} [{', '.join(written)}]"
    )
    return str(output_dir)


def compose_duo_body(
    sau_a: Path,
    sau_b: Path,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
    height_a: float | None = None,
    height_b: float | None = None,
) -> str | None:
    """Compose two people's body cutouts onto every sau_duo scene.

    Two-person form of compose_single_body: the same placement per person,
    using scale_1/anchor_1 and scale_2/anchor_2 onto one shared scene copy,
    each at their own height. No gender filter, for the same reason as the
    single-body path.
    """
    stamp = _height_stamp(height_a, height_b)
    if _cache_hit(output_dir, BODY_PREFIX, stamp):
        logger.info(f"Base duo body cache hit on disk: {output_dir}")
        return str(output_dir)

    cutout_a = load_cutout(sau_a, "sau")
    cutout_b = load_cutout(sau_b, "sau")
    if cutout_a is None or cutout_b is None:
        logger.info("Base duo body compose: both readable SAU cutouts are required")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written: list[str] = []
    for scene_id, meta in scenes_config.items():
        scene_file = find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue

        ref = meta.get("height_cm")
        placed_a = place_body(scene, cutout_a, meta.get("scale_1", 1.0),
                              meta.get("anchor_1", (0, 0)), height_a, ref)
        placed_b = place_body(scene, cutout_b, meta.get("scale_2", 1.0),
                              meta.get("anchor_2", (0, 0)), height_b, ref)
        # A duo slide showing only one of the pair would misrepresent the
        # scene — skip it and let the caller fall back to single display.
        if not (placed_a and placed_b):
            logger.debug(f"Base duo body: scene {scene_id} could not place both people")
            continue

        if _write_scene(scene, output_dir, f"{BODY_PREFIX}{scene_id}"):
            written.append(scene_label(scene_id, meta))

    if not written:
        logger.warning("Base duo body compose: no scenes produced any output")
        return None
    _write_stamp(output_dir, BODY_PREFIX, stamp)
    logger.info(
        f"Base duo body compose: {len(written)} scene(s) -> {output_dir} "
        f"[{', '.join(written)}]"
    )
    return str(output_dir)


def compose_duo_face(
    detector: cv2.FaceDetectorYN | None,
    fru_a: Path,
    fru_b: Path,
    gender_a: str | None,
    gender_b: str | None,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose two people's face cutouts onto every matching fru_duo scene.

    The two-person form of compose_single_face: the same warp per person,
    against face_anchor_1 and face_anchor_2. Unlike the single path it
    gender-filters on the PAIR — see _duo_gender_allows.
    """
    if _cache_hit(output_dir, FACE_PREFIX):
        logger.info(f"Base duo face cache hit on disk: {output_dir}")
        return str(output_dir)

    face_a = load_cutout(fru_a, "fru")
    face_b = load_cutout(fru_b, "fru")
    if face_a is None or face_b is None:
        logger.info("Base duo face compose: both readable FRU cutouts are required")
        return None

    kps_a = face_kps(face_a, detector)
    kps_b = face_kps(face_b, detector)
    if kps_a is None or kps_b is None:
        logger.warning("Base duo face compose: no eye landmarks for one or both cutouts")
        return None

    scenes_config = _load_scenes_config(scenes_config_path)
    if not scenes_config:
        return None

    written: list[str] = []
    for scene_id, meta in scenes_config.items():
        if not _duo_gender_allows(gender_a, gender_b, meta):
            continue

        anchor_a = meta.get("face_anchor_1")
        anchor_b = meta.get("face_anchor_2")
        if anchor_a is None or anchor_b is None:
            continue

        # The pair arrives in slug order, not by gender. In a mixed scene an
        # anchor may name the gender its figure is drawn as; swap the two
        # people when they arrive the wrong way round for it.
        pair = ((face_a, kps_a), (face_b, kps_b))
        want = str(anchor_a.get("gender") or "").lower()
        ga, gb = (gender_a or "").lower(), (gender_b or "").lower()
        if want and ga and gb and ga != gb and ga != want:
            pair = pair[::-1]

        scene_file = find_scene_file(scenes_dir, scene_id)
        if scene_file is None:
            continue
        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue

        try:
            placed = [place_face(scene, face, kps, anchor)
                      for (face, kps), anchor in zip(pair, (anchor_a, anchor_b), strict=True)]
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Base duo face compose: bad anchor for scene {scene_id}: {e}")
            continue
        if not all(placed):
            logger.debug(f"Base duo face: scene {scene_id} could not place both faces")
            continue

        if _write_scene(scene, output_dir, f"{FACE_PREFIX}{scene_id}"):
            written.append(scene_label(scene_id, meta))

    if not written:
        logger.warning("Base duo face compose: no scenes produced any output")
        return None
    _write_stamp(output_dir, FACE_PREFIX)
    logger.info(
        f"Base duo face compose: {len(written)} scene(s) -> {output_dir} "
        f"[{', '.join(written)}]"
    )
    return str(output_dir)
