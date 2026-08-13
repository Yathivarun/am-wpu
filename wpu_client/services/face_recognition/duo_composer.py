"""Duo scene composition — compose two people's alpha-cutout faces onto
matching scene backgrounds, side by side.

Used by face_service.py when exactly two known people are tracked
simultaneously in diagnostic mode. Reuses the SAME YuNet detector instance
the face service already has loaded (passed in) — no second model load.

Scene assets (edit paths in face_service.py if your layout differs):
    data/duo_scenes/
        scenes_config.json      ← face_anchor_1 / face_anchor_2 per scene
        1.png, 2.png, ...        ← scene backgrounds, filename stem == json key

Output:
    data/duo_output/<slug_a>__<slug_b>/scene_<id>.png

If the output directory already exists and is non-empty, composition is
skipped and the existing files are reused (on-disk cache across restarts,
in addition to the in-memory cache in face_service.py).

`compose_duo` itself stays the diagnostic-mode entry point. Its building
blocks (`load_alpha_bgra`, `detect_eye_kps`, `warp_and_blend_face`,
`pick_alpha_image`) are public because base_composer.py reuses them for
base-mode SAU/FRU composition — the geometry is identical, only the inputs
and output directories differ.
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def load_alpha_bgra(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def pick_alpha_image(alpha_dir: str) -> Path | None:
    """First image (sorted) in a person's alpha-images folder, or None."""
    d = Path(alpha_dir)
    if not d.is_dir():
        return None
    files = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()
    )
    return files[0] if files else None


def detect_eye_kps(detector: cv2.FaceDetectorYN, face_bgr: np.ndarray) -> np.ndarray | None:
    """Row layout: [x,y,w,h, 5x(lm_x,lm_y), score]; first two landmarks = eyes."""
    h, w = face_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(face_bgr)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: float(row[2]) * float(row[3]))
    return np.asarray(best[4:8], dtype=np.float32).reshape(2, 2)


def warp_and_blend_face(
    scene_bgra: np.ndarray,
    face_bgra: np.ndarray,
    face_kps: np.ndarray,
    target_eye_midpoint: tuple[float, float],
    target_eye_distance: float,
    target_tilt_angle: float,
) -> None:
    """One warpAffine (scale+rotate+translate combined) + bounded alpha blend."""
    left_eye = face_kps[0].astype(np.float64)
    right_eye = face_kps[1].astype(np.float64)

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1.0:
        eye_dist = 1.0
    mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    face_angle_deg = math.degrees(math.atan2(-dy, dx))

    scale = float(target_eye_distance) / eye_dist
    rotation_deg = target_tilt_angle - face_angle_deg

    M = cv2.getRotationMatrix2D(center=mid, angle=rotation_deg, scale=scale)
    M[0, 2] += target_eye_midpoint[0] - mid[0]
    M[1, 2] += target_eye_midpoint[1] - mid[1]

    scene_h, scene_w = scene_bgra.shape[:2]
    warped = cv2.warpAffine(
        face_bgra, M, (scene_w, scene_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    alpha_channel = warped[:, :, 3]
    ys, xs = np.nonzero(alpha_channel)
    if len(xs) == 0:
        return

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    alpha = (alpha_channel[y1:y2, x1:x2].astype(np.float32) / 255.0)[..., None]
    scene_roi = scene_bgra[y1:y2, x1:x2, :3].astype(np.float32)
    face_roi = warped[y1:y2, x1:x2, :3].astype(np.float32)
    blended = scene_roi * (1 - alpha) + face_roi * alpha
    scene_bgra[y1:y2, x1:x2, :3] = blended.astype(np.uint8)


def compose_duo(
    detector: cv2.FaceDetectorYN,
    face_a_path: Path,
    face_b_path: Path,
    gender_a: str | None,
    gender_b: str | None,
    scenes_dir: Path,
    scenes_config_path: Path,
    output_dir: Path,
) -> str | None:
    """Compose two alpha faces onto every matching scene. Returns the output
    directory path on success (existing or newly written), or None if
    composition wasn't possible (missing assets, no eyes detected, etc.) —
    the caller should fall back to single-face display in that case.
    """
    # On-disk cache: reuse a previously-composed output dir untouched.
    if output_dir.is_dir() and any(output_dir.iterdir()):
        logger.info(f"Duo cache hit on disk: {output_dir}")
        return str(output_dir)

    if not scenes_config_path.exists():
        logger.warning(f"Duo scenes config not found: {scenes_config_path}")
        return None

    face_a_bgra = load_alpha_bgra(face_a_path)
    face_b_bgra = load_alpha_bgra(face_b_path)
    if face_a_bgra is None or face_b_bgra is None:
        logger.warning(
            f"Duo compose: could not read alpha face(s) "
            f"({face_a_path if face_a_bgra is None else ''} "
            f"{face_b_path if face_b_bgra is None else ''})"
        )
        return None

    kps_a = detect_eye_kps(detector, face_a_bgra[:, :, :3])
    kps_b = detect_eye_kps(detector, face_b_bgra[:, :, :3])
    if kps_a is None or kps_b is None:
        logger.warning("Duo compose: could not detect eyes in one or both alpha faces")
        return None

    with open(scenes_config_path) as f:
        scenes_config: dict = json.load(f)

    written = []
    for scene_id, meta in scenes_config.items():
        allowed_genders = meta.get("gender", ["male", "female", "unknown"])
        if gender_a and gender_a not in allowed_genders and gender_b and gender_b not in allowed_genders:
            continue

        scene_file = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = scenes_dir / f"{scene_id}{ext}"
            if candidate.exists():
                scene_file = candidate
                break
        if scene_file is None:
            continue

        anchor_a = meta.get("face_anchor_1")
        anchor_b = meta.get("face_anchor_2")
        if anchor_a is None or anchor_b is None:
            continue

        scene = load_alpha_bgra(scene_file)
        if scene is None:
            continue

        warp_and_blend_face(
            scene, face_a_bgra, kps_a,
            tuple(anchor_a["target_eye_midpoint"]),
            anchor_a["target_eye_distance"],
            anchor_a["target_tilt_angle"],
        )
        warp_and_blend_face(
            scene, face_b_bgra, kps_b,
            tuple(anchor_b["target_eye_midpoint"]),
            anchor_b["target_eye_distance"],
            anchor_b["target_tilt_angle"],
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"scene_{scene_id}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(scene, cv2.COLOR_BGRA2BGR))
        written.append(out_path)

    if not written:
        logger.warning("Duo compose: no matching scenes produced any output")
        return None

    logger.info(f"Duo compose: {len(written)} scene(s) written to {output_dir}")
    return str(output_dir)
