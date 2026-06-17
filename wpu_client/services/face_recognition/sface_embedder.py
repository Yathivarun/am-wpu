"""Camera-free YuNet + SFace embedding.

Shared by the seed tool (`tools/seed_face.py`) so that gallery (seed) embeddings
are produced by the EXACT same steps as the live probe embeddings in
`face_service.FaceRecognitionService._capture_and_process`. Imports only
cv2/numpy (no picamera2), so it runs on the Pi or any workstation.

INVARIANT: the detect → largest-face → alignCrop(BGR) → feature → L2-normalise
sequence here MUST stay byte-for-byte identical to the live path in
face_service.py, otherwise seeded vectors and live vectors live in different
spaces and matching breaks. If you change one, change the other.
"""

import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

YUNET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx",
)
SFACE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_recognition_sface_2021dec.onnx",
)


class SFaceEmbedder:
    """YuNet (detect) + SFace (128-D embed) over a BGR image."""

    def __init__(self):
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None

    def load(self) -> None:
        # Same YuNet params as the live service: 320×320, score 0.6, nms 0.3, top_k 5000.
        self._detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320), 0.6, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        logger.info("SFaceEmbedder loaded (YuNet + SFace)")

    def detect_faces(self, bgr: np.ndarray) -> list:
        """Return the raw valid YuNet rows (bbox + 5 landmarks) for a BGR image."""
        h, w = bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        valid = []
        if faces is not None:
            coord_limit = 10 * max(h, w)
            for row in faces:
                arr = np.asarray(row, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    continue
                if float(np.max(np.abs(arr[:14]))) > coord_limit:
                    continue
                valid.append(row)
        return valid

    def embed_row(self, bgr: np.ndarray, row) -> np.ndarray:
        """Embed one detected face row → L2-normalised 128-D float32 vector."""
        aligned = self._recognizer.alignCrop(bgr, row)
        feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)

    def embed_largest(self, bgr: np.ndarray, min_face_size: int = 0):
        """Detect, pick the largest face by bbox area, return (vector, (x,y,w,h)).

        Returns (None, None) if no face; (None, bbox) if the largest face is
        below min_face_size.
        """
        valid = self.detect_faces(bgr)
        if not valid:
            return None, None
        row = max(valid, key=lambda r: float(r[2]) * float(r[3]))
        x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if fw < min_face_size or fh < min_face_size:
            return None, (x, y, fw, fh)
        return self.embed_row(bgr, row), (x, y, fw, fh)
