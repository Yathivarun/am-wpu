"""Capture store: full camera frames plus each detected face, with metadata.

The recognition loop already keeps up to 50 frames per visit of *recognised*
people (data/dataset/). This keeps everything the camera saw that had a face
in it — recognised or not — as raw material for features that need the whole
frame (placement, crowd, body framing) and for the site test kit, which reads
its range, threshold and multi-face numbers straight out of these records.

Layout, one folder per saved frame:

    <dir>/<YYYYMMDD>/<HHMMSS>_<mmm>_<seq>/
        frame.jpg        the full `main` stream frame (1640x1232), as captured
        face_<i>.jpg     each detected face, cropped with a margin
        meta.json        the record below
    <dir>/<YYYYMMDD>/index.jsonl   one line per saved frame (same record)

A record holds the capture time, frame size, detection time, and per face:
bbox, landmarks, detector score, size, what recognition made of it (status,
identity, distance, nearest gallery entry) and the crop's sharpness and
brightness. Faces past the per-frame cap or below min_face_size are recorded
too, with a status saying why they were not identified.

Writing happens on its own thread behind a small queue: JPEG-encoding a full
frame costs tens of milliseconds on a Pi and must not stretch the recognition
loop. When the queue is full the frame is dropped and counted, never waited
for. A disk budget evicts the oldest captures first. Nothing here can raise
into the caller.
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

INDEX_NAME = "index.jsonl"
META_NAME = "meta.json"
FRAME_NAME = "frame.jpg"
QUEUE_SIZE = 8
# Re-scan the real disk usage every this many saves, to correct the running
# total for anything deleted or added behind our back.
RESCAN_EVERY = 500


def face_crop(frame: np.ndarray, bbox, margin: float) -> Optional[np.ndarray]:
    """The face's bbox grown by `margin` (fraction of its size) on every side,
    clipped to the frame. None for an empty box."""
    h, w = frame.shape[:2]
    x, y, fw, fh = (float(v) for v in bbox)
    mx, my = fw * margin, fh * margin
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(w, int(x + fw + mx)), min(h, int(y + fh + my))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def crop_quality(crop: np.ndarray) -> dict:
    """Sharpness (variance of the Laplacian) and mean brightness (0-255)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
    }


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


class CaptureStore:
    """Saves frames + face crops + metadata on a background thread."""

    def __init__(
        self,
        root,
        save_full_frame: bool = True,
        save_faces: bool = True,
        min_interval_s: float = 1.0,
        save_empty_every_s: float = 0.0,
        jpeg_quality: int = 90,
        face_margin: float = 0.25,
        max_bytes: int = 0,
    ):
        self.root = Path(root)
        self.save_full_frame = save_full_frame
        self.save_faces = save_faces
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.save_empty_every_s = max(0.0, float(save_empty_every_s))
        self.jpeg_quality = int(jpeg_quality)
        self.face_margin = float(face_margin)
        self.max_bytes = int(max_bytes or 0)

        self._queue: "queue.Queue" = queue.Queue(maxsize=QUEUE_SIZE)
        self._thread: Optional[threading.Thread] = None
        self._last_face_save = 0.0
        self._last_empty_save = 0.0
        self._seq = 0
        self._bytes = 0
        self._saves_since_scan = 0
        self.saved = 0
        self.dropped = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="capture-store", daemon=True)
        self._thread.start()
        logger.info(
            f"Capture store on: {self.root} (frames={self.save_full_frame}, "
            f"faces={self.save_faces}, every {self.min_interval_s:g}s, "
            f"cap {self.max_bytes / 1e9:.1f} GB)"
        )

    def stop(self, timeout: float = 5.0) -> None:
        if not self._thread:
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)
        self._thread = None
        logger.info(f"Capture store stopped: {self.saved} saved, {self.dropped} dropped")

    # ── producer side (recognition thread) ───────────────────────────────

    def wants(self, has_faces: bool, now: Optional[float] = None) -> bool:
        """Whether a frame captured now should be saved. Cheap; call before
        building a record."""
        now = time.time() if now is None else now
        if has_faces:
            return now - self._last_face_save >= self.min_interval_s
        return bool(self.save_empty_every_s) and now - self._last_empty_save >= self.save_empty_every_s

    def submit(self, frame: np.ndarray, record: dict) -> bool:
        """Queue one frame and its record. Never blocks; False if dropped.

        `frame` must not be written to afterwards (Picamera2 hands back a
        fresh array per capture, so the loop's frame is safe to pass as is).
        """
        now = record.get("t") or time.time()
        if record.get("faces"):
            self._last_face_save = now
        else:
            self._last_empty_save = now
        try:
            self._queue.put_nowait((frame, record))
            return True
        except queue.Full:
            self.dropped += 1
            if self.dropped in (1, 10) or self.dropped % 100 == 0:
                logger.warning(f"Capture store falling behind — {self.dropped} frame(s) dropped")
            return False

    # ── writer thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._bytes = _dir_size(self.root)
        except Exception:
            self._bytes = 0
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as e:
                logger.error(f"Capture store write failed: {e}", exc_info=True)

    def _write(self, frame: np.ndarray, record: dict) -> None:
        t = record.get("t") or time.time()
        self._seq += 1
        day = time.strftime("%Y%m%d", time.localtime(t))
        name = time.strftime("%H%M%S", time.localtime(t)) + f"_{int((t % 1) * 1000):03d}_{self._seq:06d}"
        day_dir = self.root / day
        cap_dir = day_dir / name
        cap_dir.mkdir(parents=True, exist_ok=True)
        params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        written = 0

        if self.save_full_frame:
            path = cap_dir / FRAME_NAME
            cv2.imwrite(str(path), frame, params)
            written += path.stat().st_size

        for face in record.get("faces", []):
            crop = face_crop(frame, face["bbox"], self.face_margin)
            if crop is None:
                continue
            face.update(crop_quality(crop))
            if self.save_faces:
                fname = f"face_{face['i']}.jpg"
                cv2.imwrite(str(cap_dir / fname), crop, params)
                face["crop"] = fname
                written += (cap_dir / fname).stat().st_size

        record["dir"] = f"{day}/{name}"
        if self.save_full_frame:
            record["frame"] = FRAME_NAME
        line = json.dumps(record, separators=(",", ":"))
        (cap_dir / META_NAME).write_text(line)
        with open(day_dir / INDEX_NAME, "a") as f:
            f.write(line + "\n")
        written += len(line) * 2

        self.saved += 1
        self._bytes += written
        self._saves_since_scan += 1
        if self._saves_since_scan >= RESCAN_EVERY:
            self._saves_since_scan = 0
            self._bytes = _dir_size(self.root)
        if self.max_bytes and self._bytes > self.max_bytes:
            self._evict()

    def _evict(self) -> None:
        """Delete the oldest captures until 90% of the budget is free again.

        Index lines of evicted captures stay behind (they are small, and the
        record of what was seen outlives the pixels); their `dir` no longer
        exists.
        """
        target = int(self.max_bytes * 0.9)
        removed = 0
        for day_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            for cap_dir in sorted(p for p in day_dir.iterdir() if p.is_dir()):
                if self._bytes <= target:
                    break
                size = _dir_size(cap_dir)
                shutil.rmtree(cap_dir, ignore_errors=True)
                self._bytes -= size
                removed += 1
            if self._bytes <= target:
                break
        if removed:
            logger.warning(f"Capture store over budget — evicted {removed} oldest capture(s)")


def build_from_config(cfg) -> Optional[CaptureStore]:
    """A CaptureStore from FaceRecognitionConfig.capture_store, or None when off."""
    cs = getattr(cfg, "capture_store", None)
    if cs is None or not cs.enabled:
        return None
    root = Path(cs.dir)
    if not root.is_absolute():
        from wpu_client.paths import PROJECT_ROOT

        root = PROJECT_ROOT / root
    return CaptureStore(
        root,
        save_full_frame=cs.save_full_frame,
        save_faces=cs.save_faces,
        min_interval_s=cs.min_interval_s,
        save_empty_every_s=cs.save_empty_every_s,
        jpeg_quality=cs.jpeg_quality,
        face_margin=cs.face_margin,
        max_bytes=cs.max_bytes,
    )
