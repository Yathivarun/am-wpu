"""Camera-free YuNet embedders (SFace + MobileFaceNet).

Shared by the seed tool (`scripts/seed_face.py`) so that gallery (seed) embeddings
are produced by the EXACT same steps as the live probe embeddings in
`face_service.FaceRecognitionService`. Imports only cv2/numpy/onnxruntime (no
picamera2), so it runs on the Pi or any workstation.

INVARIANT: each embedder's detect → largest-face → align → embed → L2-normalise
sequence MUST stay byte-for-byte identical to the matching path in
face_service.py (`_embed_face`), otherwise seeded vectors and live vectors live
in different spaces and matching breaks. If you change one, change the other.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from wpu_client.paths import MODELS_DIR

logger = logging.getLogger(__name__)

YUNET_MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
SFACE_MODEL_PATH = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
MOBILEFACENET_MODEL_PATH = str(MODELS_DIR / "mobilefacenet.onnx")

# ArcFace 5-point reference template (112×112) — same as the server's auraface
# alignment and face_service._embed_face's mobilenet branch.
_ARCFACE_REF_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class _YuNetEmbedder:
    """Base: YuNet detection + largest-face selection shared by all embedders.

    Subclasses implement `load()` (set `self._detector`) and `embed_row()`.
    """

    def __init__(self):
        self._detector: Optional[cv2.FaceDetectorYN] = None

    def _load_detector(self) -> None:
        # Same YuNet params as the live service: 320×320, score 0.6, nms 0.3, top_k 5000.
        self._detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (320, 320), 0.6, 0.3, 5000
        )

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
        """Embed one detected face row → L2-normalised float32 vector."""
        raise NotImplementedError

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


class SFaceEmbedder(_YuNetEmbedder):
    """YuNet (detect) + SFace (128-D embed) over a BGR image."""

    def __init__(self):
        super().__init__()
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None

    def load(self) -> None:
        self._load_detector()
        self._recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        logger.info("SFaceEmbedder loaded (YuNet + SFace, 128-D)")

    def embed_row(self, bgr: np.ndarray, row) -> np.ndarray:
        """Embed one detected face row → L2-normalised 128-D float32 vector."""
        aligned = self._recognizer.alignCrop(bgr, row)
        feat = self._recognizer.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-8)


class MobileFaceNetEmbedder(_YuNetEmbedder):
    """YuNet (detect) + distilled MobileFaceNet (512-D embed) over a BGR image.

    Mirrors face_service._embed_face's mobilenet branch EXACTLY: align the YuNet
    5-pt landmarks to the ArcFace template → 112×112 → (x-127.5)/128 → ONNX → L2.
    """

    def __init__(self):
        super().__init__()
        self._session = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        import onnxruntime as ort

        self._load_detector()
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            MOBILEFACENET_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("MobileFaceNetEmbedder loaded (YuNet + MobileFaceNet, 512-D)")

    def embed_row(self, bgr: np.ndarray, row) -> np.ndarray:
        """Embed one detected face row → L2-normalised 512-D float32 vector."""
        lmk = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
        matrix, _ = cv2.estimateAffinePartial2D(
            lmk, _ARCFACE_REF_LANDMARKS, method=cv2.LMEDS
        )
        if matrix is None:
            raise RuntimeError("face alignment failed (no affine transform)")
        aligned = cv2.warpAffine(
            bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0
        )
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        blob = ((rgb.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1)[
            None, ...
        ]
        out = self._session.run(None, {self._input_name: blob})[0][0].astype(np.float32)
        return out / (np.linalg.norm(out) + 1e-8)


# Map the config `model` value → embedder class. Keys match config.yaml.
EMBEDDERS = {"sface": SFaceEmbedder, "mobilenet": MobileFaceNetEmbedder}


def build_embedder(model_name: str) -> _YuNetEmbedder:
    """Construct (unloaded) embedder for a config model name ("sface"|"mobilenet")."""
    try:
        return EMBEDDERS[model_name]()
    except KeyError:
        raise ValueError(
            f"Unknown model '{model_name}' (expected one of {sorted(EMBEDDERS)})"
        ) from None
