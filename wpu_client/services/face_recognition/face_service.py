"""Face recognition service - captures, recognizes, and identifies faces."""

import csv
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from picamera2 import Picamera2
from scipy.spatial.distance import cosine

from wpu_client.config.settings import FaceRecognitionConfig
from wpu_client.core.events import Event, EventBus
from wpu_client.core.service_base import ServiceBase
from wpu_client.models.api import IdentifyRequest, IdentifyResponse
from wpu_client.paths import DATA_DIR, MODELS_DIR
from wpu_client.services.face_recognition.asset_fetcher import (
    DISPLAY_DIR_NAME,
    enforce_disk_budget,
    fetch_person_assets,
    link_videos_into,
    touch,
)
from wpu_client.services.face_recognition.base_composer import (
    compose_duo_body,
    compose_duo_face,
    compose_single_body,
    compose_single_face,
)
from wpu_client.services.face_recognition.diagnostic_gallery import DiagnosticGallery
from wpu_client.services.face_recognition.duo_composer import compose_duo, pick_alpha_image
from wpu_client.utils.http import HTTPClient

logger = logging.getLogger(__name__)

DATASET_DIR = str(DATA_DIR / "dataset")
DATASET_MAX_BYTES = 30 * 1024 * 1024 * 1024  # 30 GB hard cap
DATASET_MAX_FRAMES_PER_VISIT = 50             # max frames saved per visit

# ── Duo scene composition (diagnostic mode only) ────────────────────────────
# When exactly two known people are tracked together, their alpha-cutout
# faces (data/embeddings/<slug>/sketches/images/) are composed onto every
# matching scene under DUO_SCENES_DIR, using DUO_SCENES_CONFIG for placement.
# Output is cached under DUO_OUTPUT_DIR/<slug_a>__<slug_b>/ and reused for as
# long as that exact pair stays in frame (and across restarts, on disk).
DUO_SCENES_DIR = DATA_DIR / "duo_scenes"
DUO_SCENES_CONFIG_PATH = DUO_SCENES_DIR / "scenes_config.json"
DUO_OUTPUT_DIR = DATA_DIR / "duo_output"

# ── Base-mode local composition (server-recognition mode) ───────────────────
# Base mode composes each visitor's slides on-device from the two raw alpha
# cutouts the server serves (SAU body / FRU face), instead of downloading
# pre-composed final images. Fetched cutouts and composed output are cached
# per person under BASE_ASSETS_DIR/<registration_id>/ and per pair under
# BASE_ASSETS_DIR/duo/<id_a>__<id_b>/.
#
# Deliberately separate from the DUO_* paths above: those stay
# diagnostic-only, share no assets and no code path with this, so base mode
# cannot regress the existing diagnostic duo feature.
BASE_ASSETS_DIR = DATA_DIR / "base_assets"
BASE_DUO_ASSETS_DIR = BASE_ASSETS_DIR / "duo"
BASE_SCENES_DIR = DATA_DIR / "base_scenes"

SAU_SINGLE_DIR = BASE_SCENES_DIR / "sau_single"
SAU_SINGLE_CONFIG_PATH = SAU_SINGLE_DIR / "scenes_config.json"
FRU_SINGLE_DIR = BASE_SCENES_DIR / "fru_single"
FRU_SINGLE_CONFIG_PATH = FRU_SINGLE_DIR / "scenes_config.json"
SAU_DUO_DIR = BASE_SCENES_DIR / "sau_duo"
SAU_DUO_CONFIG_PATH = SAU_DUO_DIR / "scenes_config.json"
FRU_DUO_DIR = BASE_SCENES_DIR / "fru_duo"
FRU_DUO_CONFIG_PATH = FRU_DUO_DIR / "scenes_config.json"

# Enable debug frame saving by setting environment variable: SAVE_DEBUG_FRAMES=1
SAVE_DEBUG_FRAMES = os.getenv("SAVE_DEBUG_FRAMES", "0") == "1"
DEBUG_FRAME_DIR = "/tmp/face_detection_debug"

# Model paths
YUNET_MODEL_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")

SFACE_MODEL_PATH = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")

# Distilled MobileFaceNet (512-D) — the alternative recognition model, selected
# with `model: mobilenet` in config.yaml. Matches the registration server's
# `auraface` gallery (YuNet 5-pt align → 112×112 → (x-127.5)/128 → L2-norm).
MOBILEFACENET_MODEL_PATH = str(MODELS_DIR / "mobilefacenet.onnx")

# ArcFace 5-point reference template (112×112) used to align faces for the
# MobileFaceNet embedder — identical to the server's auraface alignment.
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

# config `model` value → server gallery id sent in the identify request.
_SERVER_MODEL_ID = {"sface": "sface", "mobilenet": "auraface"}

# Maximum number of faces processed per captured frame (largest-first). Caps
# per-frame CPU cost when a crowd is in view; raise/lower freely.
MAX_FACES_PER_FRAME = 2

# ─── Camera configuration (imx219) ──────────────────────────────────────────
# `rpicam-hello --list-cameras` shows the 1640x1232 sensor mode is read out
# with a (0,0)/3280x2464 crop — i.e. it uses the FULL sensor array (full
# field of view), just binned 2x2, unlike 1920x1080 (which crops the sensor)
# or 3280x2464 (full FOV but far heavier to process).
#
# Picamera2 derives every configured stream from ONE selected sensor mode, so
# `main` and `lores` below share the same full-FOV sensor readout — `lores`
# is just a cheap, small, ISP-downscaled copy of it. Lowering CAMERA_LORES_SIZE
# only reduces preview resolution/cost, it never crops the field of view.
#
#   main  -> full-FOV, decent resolution -> used for detection/recognition
#            and saved dataset frames (needs enough detail to embed/match).
#   lores -> full-FOV, tiny/cheap        -> used only for the on-screen live
#            camera preview widget, where quality doesn't matter but keeping
#            the Pi responsive does.
CAMERA_MAIN_SIZE = (1640, 1232)      # detection / dataset capture — full FOV
CAMERA_MAIN_FORMAT = "RGB888"
CAMERA_LORES_SIZE = (320, 240)       # live preview — full FOV, cheap to pull
CAMERA_LORES_FORMAT = "YUV420"


def _has_content(path: Path) -> bool:
    """Whether `path` is a directory with at least one file in it.

    Used to confirm a cached sketch_dir still exists before handing it out:
    the base-asset disk cap can evict a directory that an in-memory cache
    still points at.
    """
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:
        return False


class FaceRecognitionService(ServiceBase):
    """
    Face recognition service that continuously captures images,
    detects faces, and identifies them via API.

    Detects and identifies EVERY face visible in a frame (capped by
    MAX_FACES_PER_FRAME, largest-first). Each visible face gets its own
    tracking entry, its own local dataset-saving counter, and its own line
    in the on-screen overlay ("Alice (92.3%)", "Not recognized", ...).

    Image fetching / the slideshow, however, is still driven by exactly ONE
    person at a time via the existing person.detected/person.left events —
    which of the currently-visible faces produces that pair is decided by
    the single, clearly-marked _select_person_for_display() method. Nothing
    else about image fetching changes when faces are added or removed from
    frame; only that one method needs editing if a real selection
    requirement (round-robin, random, closest-face, ...) is defined later.

    Embeddings are generated locally on the Pi using YuNet (detection) +
    SFace-128D (recognition, OpenCV). The pipeline mirrors the registration
    server's SFaceBackend exactly (alignCrop + feature + L2-normalise) so the
    128-D probe vectors are directly comparable to the gallery vectors stored
    in the `face_vectors_sface` Qdrant collection. The embedding is sent to the
    server API only for identity lookup; recognition itself happens on the Pi.

    Each visible face is first checked against the cached vectors of every
    currently-tracked person (cheap, local cosine match); only faces that
    don't match anyone already tracked fall back to the diagnostic gallery
    or a server /identify call.
    """

    def __init__(self, config: FaceRecognitionConfig, event_bus: EventBus):
        """
        Initialize the face recognition service.

        Args:
            config: Face recognition configuration
            event_bus: Event bus for inter-service communication
        """
        import os

        import psutil
        self._process = psutil.Process(os.getpid())

        super().__init__("face_recognition")
        self.config = config
        self.event_bus = event_bus
        self._camera: Optional[Picamera2] = None
        self._http_client: Optional[HTTPClient] = None

        # ── Multi-face tracking ─────────────────────────────────────────
        # registration_id (server visit id / gallery slug / a generated
        # "__unrecognized__<hex>" pseudo-id) -> per-person tracking state:
        #   person_name, face_vector, confidence, sketch_dir, unrecognized,
        #   last_seen, visit_start_time, visit_frame_count, bbox_area
        self._tracked_faces: dict[str, dict] = {}
        self._person_lock = threading.Lock()

        # Which currently-tracked face is driving the image-fetch/slideshow
        # person.detected/person.left pair. Chosen every frame by
        # _select_person_for_display() — that is the ONLY place to touch to
        # change which person's images get fetched when several are visible.
        self._displayed_registration_id: Optional[str] = None
        # Snapshot of whatever is currently displayed (person_name, confidence,
        # visit_start_time, visit_frame_count) — needed because a displayed
        # "duo" has a synthetic registration_id that isn't a key in
        # _tracked_faces, so _emit_person_left_event can't look it up there.
        self._displayed_meta: Optional[dict] = None

        # Duo composition cache: frozenset({slug_a, slug_b}) -> sketch_dir
        # (the folder of composed scene images). Built once per pair, reused
        # for as long as that exact pair stays tracked together. Used by both
        # modes now — diagnostic composes from the seeded gallery, base mode
        # from the server-fetched cutouts.
        self._duo_cache: dict[frozenset, str] = {}

        # Base-mode single-person composition cache:
        # registration_id -> display/ dir of composed slides. Mirrors
        # _duo_cache; the on-disk display/ dir is the second-level cache
        # that survives restarts.
        self._base_single_cache: dict[str, str] = {}

        # Similarity threshold for local matching (lower = more strict)
        # Cosine distance: 0 = identical, 1 = completely different
        self._similarity_threshold = 0.4

        # Dataset / visit-log setup
        os.makedirs(DATASET_DIR, exist_ok=True)
        self._visit_log_path = os.path.join(DATASET_DIR, "visit_log.csv")
        self._init_visit_log()

        # Debug frame saving
        self._frame_save_counter = 0
        self._frames_since_last_save = 0
        self._max_debug_frames = 50
        self._save_every_n_frames = 2
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
            logger.info(
                f"Debug frame saving enabled. Saving every {self._save_every_n_frames} "
                f"frames (max {self._max_debug_frames} total)"
            )
            logger.info(f"Debug frames will be saved to {DEBUG_FRAME_DIR}")

        # Recognition model selection (config.yaml: "sface" | "mobilenet").
        # "sface"     → OpenCV SFace 128-D (light, default).
        # "mobilenet" → distilled MobileFaceNet 512-D (matches server auraface).
        self._model_name: str = str(getattr(config, "model", "sface")).lower()
        if self._model_name not in _SERVER_MODEL_ID:
            logger.warning(
                f"Unknown recognition model '{self._model_name}' — falling back to 'sface'"
            )
            self._model_name = "sface"
        self._server_model_id: str = _SERVER_MODEL_ID[self._model_name]

        # Models (initialised in start())
        self._yunet_detector: Optional[cv2.FaceDetectorYN] = None
        self._sface_recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._mfn_session = None          # onnxruntime session (mobilenet only)
        self._mfn_input_name: Optional[str] = None

        # Diagnostic (offline) mode — local gallery instead of server /identify
        self._diagnostic_mode: bool = getattr(config, "diagnostic_mode", False)
        self._gallery: Optional[DiagnosticGallery] = None

    def start(self) -> None:
        """Start the face recognition service."""
        if self._running:
            logger.warning("Face recognition service is already running")
            return

        logger.info("Starting face recognition service")
        logger.info(
            f"Configuration: detection_interval={self.config.detection_interval}s, "
            f"min_face_size={self.config.min_face_size}px, n={self.config.n}"
        )
        logger.info(f"API endpoint: {self.config.api_endpoint}")
        self._running = True
        self._stop_event.clear()

        # Initialize HTTP client
        self._http_client = HTTPClient(timeout=10.0, max_retries=3)

        # Initialize camera
        # `main` (detection/dataset) and `lores` (live preview) are both
        # derived from the same full-FOV sensor mode — see the CAMERA_* /
        # module-level comment above for why. They can be resized/tuned
        # independently without affecting each other's field of view.
        try:
            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": CAMERA_MAIN_SIZE, "format": CAMERA_MAIN_FORMAT},
                lores={"size": CAMERA_LORES_SIZE, "format": CAMERA_LORES_FORMAT},
            )
            self._camera.configure(config)
            self._camera.start()
            logger.info(
                f"Picamera2 started successfully "
                f"(main={CAMERA_MAIN_SIZE[0]}x{CAMERA_MAIN_SIZE[1]} {CAMERA_MAIN_FORMAT}, "
                f"lores={CAMERA_LORES_SIZE[0]}x{CAMERA_LORES_SIZE[1]} {CAMERA_LORES_FORMAT})"
            )

            # +15% brighter than auto-exposure's own target, so low-light
            # scenes still detect. This is exposure COMPENSATION (EV bias on
            # top of AE), not a fixed/manual exposure — AE stays on and keeps
            # adapting as light changes, just biased brighter by log2(1.15)
            # ≈ 0.2 stops. Both `main` and `lores` come from one sensor
            # readout (see CAMERA_* comment above), so this affects both.
            try:
                self._camera.set_controls({"ExposureValue": 0.2})
                logger.info("Exposure compensation set: +0.2 EV (~+15% brightness)")
            except Exception as e:
                logger.warning(f"Failed to set exposure compensation: {e}")
        except Exception as e:
            logger.error(f"Failed to start Picamera2: {e}")
            self._running = False
            return

        # Initialize YuNet detector
        try:
            self._yunet_detector = cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config="",
                input_size=(320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            logger.info(f"YuNet detector initialised: {os.path.basename(YUNET_MODEL_PATH)}")
        except Exception as e:
            logger.error(f"Failed to initialise YuNet: {e}")
            self._running = False
            return

        # Initialize the configured recogniser. Both paths must match the
        # registration server's corresponding backend EXACTLY so the probe
        # embeddings are comparable to the gallery vectors:
        #   • sface     → cv2.FaceRecognizerSF.alignCrop(BGR, YuNet row) →
        #                 feature() → L2 → `face_vectors_sface` (128-D). Light on Pi CPU.
        #   • mobilenet → YuNet 5-pt align to the ArcFace template → 112×112 →
        #                 (x-127.5)/128 → MobileFaceNet ONNX → L2 →
        #                 `face_vectors_auraface` (512-D).
        try:
            if self._model_name == "mobilenet":
                import onnxruntime as ort
                # Cap the embedder at 2 cores. On the Pi 4 the unbounded session
                # grabs all 4 (~390% CPU) and starves YuNet detection + the
                # slideshow UI, queueing frames. 2 intra-op threads keep embed
                # latency ~76ms (vs ~65ms at 4) while leaving 2 cores for the
                # rest of the pipeline to run/overlap smoothly.
                so = ort.SessionOptions()
                so.intra_op_num_threads = 2
                so.inter_op_num_threads = 1
                self._mfn_session = ort.InferenceSession(
                    MOBILEFACENET_MODEL_PATH,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                self._mfn_input_name = self._mfn_session.get_inputs()[0].name
                out_dim = self._mfn_session.get_outputs()[0].shape[-1]
                logger.info(
                    f"MobileFaceNet recogniser initialised: "
                    f"{os.path.basename(MOBILEFACENET_MODEL_PATH)} "
                    f"({out_dim}-D, server gallery '{self._server_model_id}')"
                )
            else:
                self._sface_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
                logger.info(
                    f"SFace recogniser initialised: {os.path.basename(SFACE_MODEL_PATH)} "
                    f"(128-D, server gallery '{self._server_model_id}')"
                )
        except Exception as e:
            logger.error(f"Failed to initialise recogniser '{self._model_name}': {e}")
            self._running = False
            return

        # Diagnostic (offline) mode: load the local seeded gallery. Recognition
        # then happens entirely on-device — no server /identify calls.
        if self._diagnostic_mode:
            self._gallery = DiagnosticGallery(self.config.diagnostic_gallery_dir)
            count = self._gallery.load()
            logger.info(
                f"DIAGNOSTIC MODE ON — matching locally against {count} seeded "
                f"face(s) (threshold={self.config.diagnostic_match_threshold} cosine). "
                f"Server /identify is disabled."
            )
            if count == 0:
                logger.warning(
                    "Diagnostic gallery is empty — no one will be recognised. "
                    "Seed faces with: python scripts/seed_face.py --name ... --face ..."
                )

        # Run the recognition loop in a background thread
        self._run_in_thread(self._recognition_loop)

    def stop(self) -> None:
        """Stop the face recognition service."""
        if not self._running:
            return

        logger.info("Stopping face recognition service")
        self._running = False
        self._stop_event.set()

        self._wait_for_thread(timeout=5.0)

        if self._camera:
            self._camera.stop()
            self._camera.close()
            self._camera = None

        if self._http_client:
            self._http_client.close()
            self._http_client = None

        logger.info("Face recognition service stopped")

    def get_preview_frame(self, max_size: Optional[tuple[int, int]] = None) -> Optional[np.ndarray]:
        """
        Return a live RGB frame from the low-resolution preview stream.

        Reads the `lores` stream, which is derived from the same full-FOV
        sensor mode as `main` (see the CAMERA_* constants at the top of this
        file) — so the preview keeps the camera's full field of view, it's
        just downscaled by the ISP to CAMERA_LORES_SIZE, which is cheap and
        keeps the Pi responsive.

        `max_size` (w, h) optionally downscales the returned copy further to
        fit within that box, preserving aspect ratio, if the widget needs
        something even smaller than CAMERA_LORES_SIZE. Only this display copy
        is scaled — recognition captures its own frames from the `main` stream.

        Returns None if camera is not ready.
        """
        if not self._camera or not self._running:
            return None
        try:
            yuv = self._camera.capture_array("lores")
            # lores stream is YUV420 — convert straight to RGB for display
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            if max_size is not None:
                h, w = rgb.shape[:2]
                scale = min(max_size[0] / w, max_size[1] / h)
                if scale < 1.0:
                    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
                    rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
            return cv2.flip(rgb, 1)
        except Exception as e:
            logger.debug(f"Live preview capture failed: {e}")
            return None

    def _recognition_loop(self) -> None:
        """Main recognition loop - runs in background thread."""
        logger.info("Face recognition loop started")
        loop_count = 0

        while self._running and not self._stop_event.is_set():
            try:
                loop_count += 1
                logger.info(f"[Loop #{loop_count}] Starting face detection cycle...")
                self._capture_and_process()
                self._check_person_timeout()
                self._publish_overlay()
            except Exception as e:
                logger.error(f"Error in recognition loop: {e}", exc_info=True)

            self._stop_event.wait(self.config.detection_interval)

        # Emit person.left for whoever was being displayed when we stopped
        with self._person_lock:
            displayed = self._displayed_registration_id
        if displayed:
            self._emit_person_left_event(displayed)
            self._displayed_registration_id = None

        logger.info("Face recognition loop ended")

    # ─────────────────────────────────────────────────────────────────────
    # Capture + per-frame detection/identification
    # ─────────────────────────────────────────────────────────────────────

    def _capture_and_process(self) -> None:
        """Capture one frame, detect all faces in it, and identify each."""
        if not self._camera:
            logger.warning("Camera is not available")
            return

        try:
            frame = self._camera.capture_array("main")
        except Exception as e:
            logger.warning(f"Failed to capture frame: {e}")
            return

        logger.info(
            f"Frame captured: {frame.shape[1]}x{frame.shape[0]} px, "
            f"dtype={frame.dtype}, range=[{frame.min():.1f}, {frame.max():.1f}]"
        )

        # capture_array("main") gives BGR despite RGB888 config
        bgr_frame = frame
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # ── Face detection: YuNet ──────────────────────────────────────────
        logger.debug("Running YuNet face detection...")
        h, w = bgr_frame.shape[:2]
        self._yunet_detector.setInputSize((w, h))

        cpu_before = self._process.cpu_times()
        wall_start = time.time()

        start_time = time.time()
        _, faces = self._yunet_detector.detect(bgr_frame)

        wall_end = time.time()
        cpu_after = self._process.cpu_times()
        yunet_wall_ms = (wall_end - wall_start) * 1000
        yunet_user_ms = (cpu_after.user - cpu_before.user) * 1000
        yunet_system_ms = (cpu_after.system - cpu_before.system) * 1000

        logger.info(
            f"[YuNet] wall={yunet_wall_ms:.1f}ms | "
            f"cpu_user={yunet_user_ms:.1f}ms | "
            f"cpu_sys={yunet_system_ms:.1f}ms | "
            f"cpu_util={((yunet_user_ms + yunet_system_ms) / yunet_wall_ms * 100):.1f}%"
        )

        detection_time = time.time() - start_time

        # Keep the RAW YuNet rows (bbox + 5 landmarks). SFace.alignCrop needs
        # the landmarks, so we must NOT reduce detections to bbox-only tuples.
        # face_locations (CSS top/right/bottom/left) is built in parallel purely
        # for logging and the debug-frame overlay.
        valid_faces = []
        face_locations = []
        if faces is not None:
            coord_limit = 10 * max(h, w)
            for row in faces:
                arr = np.asarray(row, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    continue
                if float(np.max(np.abs(arr[:14]))) > coord_limit:
                    continue
                valid_faces.append(row)
                x, y, fw, fh = arr[:4]
                face_locations.append((int(y), int(x + fw), int(y + fh), int(x)))

        if not valid_faces:
            logger.info(f"No faces detected (took {detection_time:.2f}s)")
            if SAVE_DEBUG_FRAMES:
                self._frames_since_last_save += 1
                if self._frames_since_last_save >= self._save_every_n_frames:
                    if self._frame_save_counter < self._max_debug_frames:
                        self._save_debug_frame(rgb_frame, face_locations)
                    else:
                        logger.info(f"Reached max debug frames ({self._max_debug_frames}), not saving more")
                    self._frames_since_last_save = 0
            return

        logger.info(f"Detected {len(valid_faces)} face(s) (took {detection_time:.2f}s)")

        if SAVE_DEBUG_FRAMES and self._frame_save_counter < self._max_debug_frames:
            self._save_debug_frame(rgb_frame, face_locations)

        # Largest-first, capped at MAX_FACES_PER_FRAME — process every face
        # in frame (not just the biggest one) up to that cap.
        order = sorted(
            range(len(valid_faces)),
            key=lambda i: float(valid_faces[i][2]) * float(valid_faces[i][3]),
            reverse=True,
        )[:MAX_FACES_PER_FRAME]

        processed_count = 0
        for idx in order:
            row = valid_faces[idx]
            x, y, face_width, face_height = (
                int(row[0]), int(row[1]), int(row[2]), int(row[3]),
            )

            if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
                logger.debug(f"Skipping face too small: {face_width}x{face_height}")
                continue

            logger.info(f"Processing face at ({x},{y}), size: {face_width}x{face_height}")

            # ── Embedding: configured recogniser ──────────────────────────
            # Mirrors the matching server backend EXACTLY and feeds the raw
            # YuNet detection row (bbox + 5 landmarks). BGR is required:
            # registration decodes its gallery images with cv2.imdecode
            # (BGR), so feeding RGB would shift the embedding.
            try:
                face_array = self._embed_face(bgr_frame, row)
            except Exception as e:
                logger.error(f"{self._model_name} embedding failed: {e}", exc_info=True)
                continue

            # L2-normalise (EUCLID collection convention — matches registration)
            face_array = face_array / (np.linalg.norm(face_array) + 1e-8)
            face_vector = face_array.tolist()
            bbox_area = float(face_width) * float(face_height)

            if self._process_face_vector(face_vector, bgr_frame, bbox_area):
                processed_count += 1

        logger.info(f"Identified {processed_count}/{len(order)} processed face(s) this frame")
        self._update_display_selection()

    def _embed_face(self, bgr_frame: np.ndarray, yunet_row: np.ndarray) -> np.ndarray:
        """Embed the detected face with the configured model (RAW, un-normalised).

        The caller L2-normalises. `yunet_row` is the raw OpenCV YuNet detection
        row: [x, y, w, h, 5×(landmark_x, landmark_y), score].
        """
        if self._model_name == "mobilenet":
            # YuNet's 5 landmarks (indices 4:14) are already in the ArcFace
            # template order; align → 112×112 → RGB → (x-127.5)/128 → ONNX.
            lmk = np.asarray(yunet_row[4:14], dtype=np.float32).reshape(5, 2)
            matrix, _ = cv2.estimateAffinePartial2D(
                lmk, _ARCFACE_REF_LANDMARKS, method=cv2.LMEDS
            )
            if matrix is None:
                raise RuntimeError("face alignment failed (no affine transform)")
            aligned = cv2.warpAffine(
                bgr_frame, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0
            )
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            blob = ((rgb.astype(np.float32) - 127.5) * 0.0078125).transpose(2, 0, 1)[
                None, ...
            ]
            out = self._mfn_session.run(None, {self._mfn_input_name: blob})
            return out[0][0].astype(np.float32)

        # Default: OpenCV SFace (alignCrop handles alignment internally).
        aligned = self._sface_recognizer.alignCrop(bgr_frame, yunet_row)
        return self._sface_recognizer.feature(aligned).flatten().astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Per-face identification
    # ─────────────────────────────────────────────────────────────────────

    def _process_face_vector(
        self, face_vector: list[float], frame: np.ndarray, bbox_area: float
    ) -> bool:
        """
        Identify ONE detected face.

        Cheap path first: compare against the cached vector of every
        currently-tracked person (people already seen this session/visit).
        Only a face that matches nobody already tracked falls through to the
        diagnostic gallery or a server /identify call.

        On every match — cached or fresh — the tracked entry's confidence
        and distance are (re)written to this frame's values, so the overlay
        (built separately, continuously, by _publish_overlay) always shows
        a live match% and distance rather than a stale one-time value.

        Returns True if this face is now tracked (matched or newly
        identified, recognized or not), False only on a hard failure (e.g.
        HTTP error) where nothing could be tracked at all.
        """
        with self._person_lock:
            match = self._match_cached_faces(face_vector)
            if match is not None:
                match_id, distance = match
                entry = self._tracked_faces[match_id]
                entry["face_vector"] = face_vector
                entry["last_seen"] = time.time()
                entry["bbox_area"] = bbox_area
                if not entry["unrecognized"]:
                    entry["confidence"] = (1.0 - distance) * 100
                    entry["distance"] = distance
                    self._save_dataset_frame_for(entry, frame)
                logger.debug(
                    f"Matched cached face: {entry['person_name']} ({match_id}) dist={distance:.4f}"
                )
                return True

        logger.info("No cached match for this face — identifying...")
        if self._diagnostic_mode:
            return self._match_local_gallery(face_vector, frame, bbox_area)
        else:
            return self._send_identification_request(face_vector, frame, bbox_area)

    def _match_cached_faces(self, face_vector: list[float]) -> Optional[tuple[str, float]]:
        """
        Cheap local cosine match of `face_vector` against every currently
        tracked person's cached vector. Must be called with _person_lock
        already held. Returns (registration_id, distance) for the closest
        tracked person within threshold, or None if nobody is close enough.
        """
        best_id: Optional[str] = None
        best_dist = self._similarity_threshold
        for reg_id, entry in self._tracked_faces.items():
            dist = self._compute_face_similarity(face_vector, entry["face_vector"])
            if dist < best_dist:
                best_id, best_dist = reg_id, dist
        return (best_id, best_dist) if best_id is not None else None

    def _compute_face_similarity(self, face_vector1: list[float], face_vector2: list[float]) -> float:
        """
        Compute cosine distance between two face vectors.

        Returns:
            Cosine distance (0 = identical, 1 = completely different)
        """
        try:
            return cosine(face_vector1, face_vector2)
        except Exception as e:
            logger.error(f"Error computing face similarity: {e}")
            return 1.0  # Return max distance on error

    def _match_local_gallery(
        self, face_vector: list[float], frame: np.ndarray, bbox_area: float
    ) -> bool:
        """Diagnostic mode: identify one face against the local seeded gallery
        (no server). On a match, starts tracking that person under their own
        registration_id (the gallery slug), carrying their sketch_dir so the
        slideshow can show that person's own sketches."""
        if not self._gallery or len(self._gallery) == 0:
            logger.info("Diagnostic gallery empty — no local match possible")
            self._handle_unrecognized(face_vector, bbox_area)
            return True

        match = self._gallery.match(
            face_vector, self.config.diagnostic_match_threshold, self._model_name
        )
        if match is None:
            logger.info(
                f"No local match within threshold "
                f"({self.config.diagnostic_match_threshold} cosine) — not recognized"
            )
            self._handle_unrecognized(face_vector, bbox_area)
            return True

        confidence = (1.0 - match.distance) * 100
        logger.info(
            f"Local match: {match.entry.name} "
            f"dist={match.distance:.4f} conf={confidence:.1f}% sketches={match.entry.sketch_dir}"
        )
        entry = self._track_new_face(
            registration_id=match.entry.slug,
            face_vector=face_vector,
            person_name=match.entry.name,
            confidence=confidence,
            distance=match.distance,
            sketch_dir=match.entry.sketch_dir,
            bbox_area=bbox_area,
        )
        self._save_dataset_frame_for(entry, frame, distance=match.distance)
        return True

    def _send_identification_request(
        self, face_vector: list[float], frame: np.ndarray, bbox_area: float
    ) -> bool:
        """
        Send one locally-generated face embedding to the server API for
        identity lookup. No face recognition happens server-side — it only
        matches the vector against its known-faces database and returns a
        name + confidence score.
        """
        if not self._http_client:
            logger.warning("HTTP client not available")
            return False

        try:
            request = IdentifyRequest.from_vector_list(
                type="face",
                n=self.config.n,
                face_vector=face_vector,
                model=self._server_model_id,
            )
            request_data = request.model_dump()

            vector_preview = (
                f"{len(face_vector)}D vector, "
                f"first 3: [{face_vector[0]:.4f}, {face_vector[1]:.4f}, {face_vector[2]:.4f}]"
            )
            logger.info(f"Sending POST request to {self.config.api_endpoint}")
            logger.info(
                f"Request: type={request_data['type']}, n={request_data['n']}, "
                f"vector={vector_preview}"
            )

            response_data = self._http_client.post(
                self.config.api_endpoint,
                data=request_data,
            )
            logger.info(f"Raw response body: {response_data}")

            response = IdentifyResponse(**response_data)

            # API may return confidence as 0–1 or 0–100; normalise to 0–100
            confidence = response.confidence or 0.0
            if confidence <= 1.0:
                confidence = confidence * 100

            logger.info(
                f"Identification result: {response.name} ({confidence:.1f}%) | "
                f"match_type={response.match_type}, distance={response.distance}"
            )

            if not response.success or not response.registration_id:
                self._handle_unrecognized(face_vector, bbox_area)
                return True

            # Scene configs key gender in lowercase ("male"/"female"), same
            # normalisation the diagnostic gallery applies to its own
            # metadata. A server that doesn't return gender yet leaves this
            # None, which never excludes a scene.
            gender = response.gender.lower() if response.gender else None

            # Fetch this person's cutouts and compose their slides now, so
            # sketch_dir is already populated on the tracked entry — the duo
            # path and the single-person fallback both read it from there
            # rather than recomposing.
            sketch_dir = self._ensure_base_single_assets(response.registration_id, gender)

            entry = self._track_new_face(
                registration_id=response.registration_id,
                face_vector=face_vector,
                person_name=response.person_name,
                confidence=confidence,
                distance=response.distance or 0.0,
                sketch_dir=sketch_dir,
                bbox_area=bbox_area,
                gender=gender,
            )
            self._save_dataset_frame_for(entry, frame, distance=response.distance or 0.0)

            return True

        except Exception as e:
            logger.error(
                f"Failed to send identification request: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return False

    def _handle_unrecognized(self, face_vector: list[float], bbox_area: float) -> None:
        """
        A detected face matched nobody in the cache, gallery, or server.

        Tracked under a freshly generated pseudo registration-id so this
        stays "the same not-recognized face" across frames (re-matched via
        the cheap cache check) without merging it with any OTHER
        unrecognized face simultaneously in frame. No dataset frame is
        saved for unrecognized faces.
        """
        reg_id = f"__unrecognized__{uuid.uuid4().hex[:8]}"
        self._track_new_face(
            registration_id=reg_id,
            face_vector=face_vector,
            person_name="Not recognized",
            confidence=0.0,
            sketch_dir=None,
            bbox_area=bbox_area,
            unrecognized=True,
        )

    def _track_new_face(
        self,
        registration_id: str,
        face_vector: list[float],
        person_name: str,
        confidence: float,
        distance: float = 0.0,
        sketch_dir: Optional[str] = None,
        bbox_area: float = 0.0,
        unrecognized: bool = False,
        gender: Optional[str] = None,
    ) -> dict:
        """Start tracking a newly-identified face under its own registration_id."""
        with self._person_lock:
            now = time.time()
            entry = {
                "person_name": person_name,
                "face_vector": face_vector,
                "confidence": confidence,
                "distance": distance,
                "sketch_dir": sketch_dir,
                # Carried so duo composition can gender-filter scenes later
                # without re-querying the server.
                "gender": gender,
                "unrecognized": unrecognized,
                # Flips to True the first time the overlay shows this
                # person — that first line is a one-time "Welcome!", every
                # line after is the live name/match%/distance readout.
                "welcomed": False,
                "last_seen": now,
                "visit_start_time": now,
                "visit_frame_count": 0,
                "bbox_area": bbox_area,
            }
            self._tracked_faces[registration_id] = entry
            return entry

    # ─────────────────────────────────────────────────────────────────────
    # Base-mode local composition
    # ─────────────────────────────────────────────────────────────────────

    def _base_assets_dir(self, registration_id: str) -> Path:
        """Per-person asset root: data/base_assets/<registration_id>/."""
        return BASE_ASSETS_DIR / registration_id

    def _enforce_asset_budget(self, protect: Optional[set] = None) -> None:
        """Trim the base asset cache back under its configured size cap."""
        enforce_disk_budget(
            BASE_ASSETS_DIR,
            getattr(self.config, "base_assets_max_bytes", 0),
            protect=protect,
        )

    def _ensure_base_single_assets(
        self, registration_id: str, gender: Optional[str]
    ) -> Optional[str]:
        """Fetch + compose this person's own slides, returning their display dir.

        Two cache layers before any work happens: an in-memory dict (mirrors
        _duo_cache) and the on-disk display/ dir, which survives restarts. A
        repeat visitor therefore costs nothing beyond a cheap revalidation;
        only a brand-new visitor pays the fetch+compose latency, on this
        (recognition-loop) thread. That synchronous cost is the same trade-off
        the diagnostic duo feature already ships with — see the plan's
        first-recognition-latency note.

        Returns None when nothing composable came back, in which case the
        caller simply has no sketch_dir and the slideshow falls back.
        """
        if not registration_id:
            return None

        cached = self._base_single_cache.get(registration_id)
        if cached:
            # The cap can evict a dir that's still in this cache, so a hit is
            # only trusted while the files are actually there — otherwise fall
            # through and rebuild rather than hand out a dangling path.
            if _has_content(Path(cached)):
                touch(Path(cached).parent)
                return cached
            logger.info(f"Cached slides for {registration_id} were evicted — rebuilding")
            self._base_single_cache.pop(registration_id, None)

        assets_dir = self._base_assets_dir(registration_id)
        display_dir = assets_dir / DISPLAY_DIR_NAME

        try:
            assets = fetch_person_assets(
                self._http_client,
                self.config.wpu_endpoint,
                getattr(self.config, "sau_media_endpoint", ""),
                registration_id,
                getattr(self.config, "sau_cutout_filename", "sau_cutout.png"),
                getattr(self.config, "fru_cutout_filename", "fru_cutout.png"),
                int(getattr(self.config, "video_count", 1) or 0),
                assets_dir,
            )

            # A cutout that changed server-side (same object key, new photo)
            # invalidates everything composed from the old one — including
            # every duo pair this person appears in.
            if assets["changed"] and display_dir.is_dir():
                logger.info(
                    f"Cutouts changed for {registration_id} — discarding stale composed slides"
                )
                shutil.rmtree(display_dir, ignore_errors=True)
                self._invalidate_duo_caches_for(registration_id)

            compose_single_body(
                assets["sau"], SAU_SINGLE_DIR, SAU_SINGLE_CONFIG_PATH, display_dir,
            )
            compose_single_face(
                self._yunet_detector, assets["fru"], gender,
                FRU_SINGLE_DIR, FRU_SINGLE_CONFIG_PATH, display_dir,
            )
            link_videos_into(assets["videos"], display_dir)
        except Exception as e:
            logger.error(
                f"Base asset preparation failed for {registration_id}: {e}", exc_info=True
            )

        # Whatever ended up in display/ — composed slides, videos, or both —
        # is what the slideshow will show. Nothing there means nothing to show.
        if not _has_content(display_dir):
            logger.info(f"No composed slides available for {registration_id}")
            return None

        sketch_dir = str(display_dir)
        self._base_single_cache[registration_id] = sketch_dir
        touch(assets_dir)
        # Pin this person's whole asset dir, not just display/ — eviction
        # operates on the person dir, and deleting it would pull the slides
        # out from under the slideshow that is about to show them.
        self._enforce_asset_budget(protect={str(assets_dir)})
        return sketch_dir

    def _invalidate_duo_caches_for(self, registration_id: str) -> None:
        """Drop every cached duo pairing involving this person.

        Called when their cutouts change, since each composed pair scene
        embeds the old copy.
        """
        stale = [key for key in self._duo_cache if registration_id in key]
        for key in stale:
            self._duo_cache.pop(key, None)
        for pair_dir in (BASE_DUO_ASSETS_DIR.glob(f"*{registration_id}*")
                         if BASE_DUO_ASSETS_DIR.is_dir() else []):
            shutil.rmtree(pair_dir, ignore_errors=True)
        if stale:
            logger.info(f"Invalidated {len(stale)} cached duo pairing(s) for {registration_id}")

    def _build_base_duo_sketch_dir(
        self, entry_a: dict, entry_b: dict, id_a: str, id_b: str
    ) -> Optional[str]:
        """Compose a base-mode duo display dir for this pair.

        Reuses the raw cutouts already fetched by _ensure_base_single_assets
        when each person was first identified — no refetching here. Videos are
        single-person only, so none are linked in.
        """
        raw_a = self._base_assets_dir(id_a) / "raw"
        raw_b = self._base_assets_dir(id_b) / "raw"
        sau_a, sau_b = raw_a / "sau.png", raw_b / "sau.png"
        fru_a, fru_b = raw_a / "fru.png", raw_b / "fru.png"

        output_dir = BASE_DUO_ASSETS_DIR / f"{id_a}__{id_b}" / DISPLAY_DIR_NAME

        if sau_a.exists() and sau_b.exists():
            compose_duo_body(sau_a, sau_b, SAU_DUO_DIR, SAU_DUO_CONFIG_PATH, output_dir)
        if fru_a.exists() and fru_b.exists():
            compose_duo_face(
                self._yunet_detector, fru_a, fru_b,
                entry_a.get("gender"), entry_b.get("gender"),
                FRU_DUO_DIR, FRU_DUO_CONFIG_PATH, output_dir,
            )

        if not _has_content(output_dir):
            return None

        sketch_dir = str(output_dir)
        touch(output_dir.parent)
        # Pin the pair dir itself, plus both people's dirs — the pair was
        # composed from their cutouts and would have to refetch if evicted
        # while still on screen.
        self._enforce_asset_budget(protect={
            str(output_dir.parent),
            str(self._base_assets_dir(id_a)),
            str(self._base_assets_dir(id_b)),
        })
        return sketch_dir

    # ─────────────────────────────────────────────────────────────────────
    # Per-frame wrap-up: overlay (all faces) + display selection (one face)
    # ─────────────────────────────────────────────────────────────────────

    def _publish_overlay(self) -> None:
        """
        Refresh the on-screen name/details overlay from current tracking
        state. Called once per loop tick — every ~detection_interval
        seconds — independent of whether this exact frame detected any
        faces, so a name doesn't blink off just because a single frame
        missed a detection during someone's tracking cycle.

        Per tracked person:
          - First time they're shown: "{name} - Welcome!" (once).
          - Every tick after that, for as long as they're tracked (i.e.
            for as long their tracking cycle — and therefore their
            images/sketches on the slideshow — is active): a live
            "{name} ({confidence:.1f}%, dist={distance:.3f})" line, so
            it's always visible whose images are currently on screen.
          - An unrecognized face always reads "Not recognized".
        Multiple simultaneously-tracked people each get their own line.
        When nobody is tracked, an empty line list is published so the
        overlay clears back to normal.
        """
        if not self.config.display_result:
            return

        with self._person_lock:
            lines = []
            for entry in self._tracked_faces.values():
                if entry["unrecognized"]:
                    lines.append("Not recognized")
                elif not entry["welcomed"]:
                    lines.append(f"{entry['person_name']} - Welcome!")
                    entry["welcomed"] = True
                else:
                    lines.append(
                        f"{entry['person_name']} "
                        f"({entry['confidence']:.1f}%, dist={entry['distance']:.3f})"
                    )

        self.event_bus.publish(Event(
            event_type="faces.recognized",
            data={
                "lines": lines,
                "hide_delay": self.config.overlay_hide_delay,
            },
        ))

    def _select_person_for_display(self, tracked_faces: dict) -> Optional[str]:
        """
        ⚠️ EDIT HERE to change which visible person's images/sketches the
        slideshow fetches when multiple faces are in frame. This is the
        ONLY place that decides which single registration_id drives the
        person.detected/person.left pair — slideshow_service.py and the
        rest of the image-fetch pipeline are untouched by multi-face
        support and don't need to change when this logic changes.

        No specific selection requirement exists yet, so this defaults to
        "whoever currently has the biggest face in frame" — swap for
        round-robin, random.choice(...), closest-face-to-camera-center, a
        priority list of registration_ids, etc. as needed.

        Args:
            tracked_faces: registration_id -> tracking entry (dict) for
                every currently-tracked (non-timed-out) person.

        Returns:
            The chosen registration_id, or None if nobody is tracked.
        """
        if not tracked_faces:
            return None
        return max(tracked_faces.items(), key=lambda kv: kv[1].get("bbox_area", 0.0))[0]

    def _update_display_selection(self) -> None:
        """Re-run selection and emit person.left/person.detected if the
        selected display target changed.

        Diagnostic mode only: when exactly two currently-tracked faces are
        both recognized (neither is "Not recognized"), that pair drives the
        display via a composed duo scene instead of the usual single
        biggest-face rule — see _get_or_build_duo_sketch_dir(). Any other
        combination (0, 1, or >2 faces; or 2 faces where either is
        unrecognized) uses the existing _select_person_for_display() path
        unchanged.
        """
        with self._person_lock:
            recognized_ids = [
                rid for rid, e in self._tracked_faces.items() if not e["unrecognized"]
            ]
            # Duo display now works in BOTH modes: diagnostic composes from
            # the seeded gallery, base mode from the server-fetched cutouts.
            # The mode-specific part lives inside
            # _get_or_build_duo_sketch_dir, not in this gate.
            duo_mode = len(self._tracked_faces) == 2 and len(recognized_ids) == 2

            if duo_mode:
                id_a, id_b = sorted(recognized_ids)
                pair_key = frozenset((id_a, id_b))
                selected_id = f"duo::{id_a}::{id_b}"
                entry_a, entry_b = self._tracked_faces[id_a], self._tracked_faces[id_b]
            else:
                pair_key = None
                selected_id = self._select_person_for_display(self._tracked_faces)
                entry_a = entry_b = None

            if selected_id == self._displayed_registration_id:
                return

            previous_id = self._displayed_registration_id
            previous_meta = self._displayed_meta
            self._displayed_registration_id = selected_id

            if duo_mode:
                new_meta = {
                    "person_name": f"{entry_a['person_name']} & {entry_b['person_name']}",
                    "confidence": min(entry_a["confidence"], entry_b["confidence"]),
                    "visit_start_time": time.time(),
                    "visit_frame_count": 0,
                }
            elif selected_id is not None:
                single_entry = self._tracked_faces.get(selected_id)
                new_meta = dict(single_entry) if single_entry else None
            else:
                new_meta = None
            self._displayed_meta = new_meta

        if previous_id is not None:
            pm = previous_meta or {}
            self._emit_person_left_event(
                previous_id,
                person_name=pm.get("person_name"),
                confidence=pm.get("confidence", 0.0),
                visit_start_time=pm.get("visit_start_time"),
                visit_frame_count=pm.get("visit_frame_count", 0),
            )

        if duo_mode:
            sketch_dir = self._get_or_build_duo_sketch_dir(pair_key, entry_a, entry_b, id_a, id_b)
            if sketch_dir:
                self._emit_person_detected_event(
                    selected_id, new_meta["person_name"], new_meta["confidence"],
                    sketch_dir=sketch_dir, unrecognized=False,
                )
                return
            # Composition unavailable (missing cutouts/alpha images/scenes/
            # eyes) — fall back to showing ONE of the pair. Prefer whichever
            # of the two actually has usable slides already on their tracked
            # entry, so "one person has assets and the other doesn't" shows
            # the one who does, deterministically rather than by accident of
            # which face happens to be bigger. Both single-person composes
            # already ran at track time, so this costs nothing.
            logger.warning(
                "Duo composition unavailable for this pair — falling back to "
                "single-face display"
            )
            with self._person_lock:
                fallback_id = None
                for rid, entry in sorted(
                    [(id_a, entry_a), (id_b, entry_b)],
                    key=lambda t: t[1].get("bbox_area", 0),
                    reverse=True,
                ):
                    if entry.get("sketch_dir"):
                        fallback_id = rid
                        break
                if fallback_id is None:
                    fallback_id = self._select_person_for_display(self._tracked_faces)
                self._displayed_registration_id = fallback_id
                fb_entry = self._tracked_faces.get(fallback_id) if fallback_id else None
                self._displayed_meta = dict(fb_entry) if fb_entry else None
            if fallback_id is not None and fb_entry is not None:
                self._emit_person_detected_event(
                    fallback_id, fb_entry["person_name"], fb_entry["confidence"],
                    fb_entry.get("sketch_dir"), fb_entry.get("unrecognized", False),
                )
            return

        if selected_id is not None:
            entry = self._tracked_faces.get(selected_id)
            if entry is not None:
                self._emit_person_detected_event(
                    selected_id, entry["person_name"], entry["confidence"],
                    entry.get("sketch_dir"), entry.get("unrecognized", False),
                )

    def _get_or_build_duo_sketch_dir(
        self, pair_key: frozenset, entry_a: dict, entry_b: dict, id_a: str, id_b: str
    ) -> Optional[str]:
        """Return the composed-scenes folder for this pair, building it once
        and caching thereafter (in memory here; on disk inside the composer).
        Returns None if composition isn't possible — caller falls back.

        The in-memory cache wrapper is shared by both modes; only the way a
        pair is actually composed differs, hence the branch below.
        """
        cached = self._duo_cache.get(pair_key)
        if cached:
            if _has_content(Path(cached)):
                return cached
            # Evicted under the disk cap (pairs are evicted first) — drop the
            # stale entry and recompose below.
            self._duo_cache.pop(pair_key, None)

        if not self._diagnostic_mode:
            sketch_dir = self._build_base_duo_sketch_dir(entry_a, entry_b, id_a, id_b)
            if sketch_dir:
                self._duo_cache[pair_key] = sketch_dir
            return sketch_dir

        # ── Diagnostic mode: unchanged gallery-based composition ──────────
        if not self._gallery:
            return None

        gallery_entry_a = self._gallery.get_entry(id_a)
        gallery_entry_b = self._gallery.get_entry(id_b)
        if gallery_entry_a is None or gallery_entry_b is None:
            logger.warning(f"Duo compose: gallery entry missing for '{id_a}' or '{id_b}'")
            return None

        alpha_path_a = pick_alpha_image(gallery_entry_a.alpha_dir)
        alpha_path_b = pick_alpha_image(gallery_entry_b.alpha_dir)
        if alpha_path_a is None or alpha_path_b is None:
            logger.warning(
                f"Duo compose: no alpha image(s) found "
                f"(a={gallery_entry_a.alpha_dir!r}, b={gallery_entry_b.alpha_dir!r})"
            )
            return None

        output_dir = DUO_OUTPUT_DIR / f"{id_a}__{id_b}"
        sketch_dir = compose_duo(
            self._yunet_detector,
            alpha_path_a, alpha_path_b,
            gallery_entry_a.gender, gallery_entry_b.gender,
            DUO_SCENES_DIR, DUO_SCENES_CONFIG_PATH,
            output_dir,
        )
        if sketch_dir:
            self._duo_cache[pair_key] = sketch_dir
        return sketch_dir

    def _check_person_timeout(self) -> None:
        """Drop any tracked face not seen within the timeout window. If the
        currently-displayed person's entry is dropped, re-run selection so
        the slideshow moves on (to another visible face, or none)."""
        with self._person_lock:
            now = time.time()
            expired = [
                reg_id for reg_id, entry in self._tracked_faces.items()
                if now - entry["last_seen"] >= self.config.person_timeout
            ]
            for reg_id in expired:
                entry = self._tracked_faces.pop(reg_id)
                logger.info(
                    f"Person timeout: {entry['person_name']} "
                    f"not seen for {now - entry['last_seen']:.1f}s"
                )

        if expired:
            self._update_display_selection()

    # ─────────────────────────────────────────────────────────────────────
    # Event emission (image-fetch side — unchanged shape, per-entry data)
    # ─────────────────────────────────────────────────────────────────────

    def _emit_person_detected_event(
        self, registration_id: str, person_name: str, confidence: float,
        sketch_dir: Optional[str] = None, unrecognized: bool = False,
    ) -> None:
        """Emit person.detected for the single face chosen by
        _select_person_for_display() — this is what the slideshow's image
        fetch reacts to."""
        self.event_bus.publish(Event(
            event_type="person.detected",
            data={
                "registration_id": registration_id,
                "person_name": person_name,
                "confidence": confidence,
                "sketch_dir": sketch_dir,
                "unrecognized": unrecognized,
            },
        ))
        logger.info(
            f"Emitted person.detected: {person_name} "
            f"(registration_id: {registration_id}, sketches: {sketch_dir})"
        )

    def _emit_person_left_event(
        self,
        registration_id: str,
        person_name: Optional[str] = None,
        confidence: float = 0.0,
        visit_start_time: Optional[float] = None,
        visit_frame_count: int = 0,
    ) -> None:
        """Emit person.left for the previously-displayed face/pair and write
        its visit log row. Called when display-selection changes, or that
        face's tracking entry times out, or the service is stopping.

        `person_name`/etc. can be passed explicitly (used for a duo's
        synthetic registration_id, which is not a key in _tracked_faces);
        if omitted, they're looked up from _tracked_faces as before.
        """
        if person_name is None:
            with self._person_lock:
                entry = self._tracked_faces.get(registration_id)
            person_name = entry["person_name"] if entry else "unknown"
            confidence = entry["confidence"] if entry else 0.0
            visit_start_time = entry["visit_start_time"] if entry else time.time()
            visit_frame_count = entry["visit_frame_count"] if entry else 0

        self._log_visit(
            person_name=person_name,
            registration_id=registration_id,
            confidence=confidence,
            visit_start_time=visit_start_time if visit_start_time is not None else time.time(),
            visit_frame_count=visit_frame_count,
        )

        self.event_bus.publish(Event(
            event_type="person.left",
            data={"registration_id": registration_id, "person_name": person_name},
        ))
        logger.info(f"Emitted person.left: {person_name} (registration_id: {registration_id})")

    def _init_visit_log(self) -> None:
        """Create visit_log.csv with header row if it does not already exist."""
        if not os.path.exists(self._visit_log_path):
            with open(self._visit_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "person_name", "registration_id",
                    "timestamp_in", "timestamp_out",
                    "confidence", "frames_seen", "duration_seconds",
                ])
            logger.info(f"Created visit log: {self._visit_log_path}")

    def _log_visit(
        self, person_name: str, registration_id: str, confidence: float,
        visit_start_time: float, visit_frame_count: int,
    ) -> None:
        """Append one completed-visit row to visit_log.csv."""
        try:
            now = time.time()
            ts_in = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(visit_start_time or now))
            ts_out = time.strftime("%Y-%m-%d %H:%M:%S")
            duration = round(now - (visit_start_time or now), 1)
            with open(self._visit_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    person_name, registration_id,
                    ts_in, ts_out,
                    f"{confidence:.1f}", visit_frame_count, duration,
                ])
            logger.info(
                f"Visit logged: {person_name} | {visit_frame_count} frames | "
                f"{duration}s | conf={confidence:.1f}%"
            )
        except Exception as e:
            logger.error(f"Failed to write visit log: {e}")

    def _get_dataset_size(self) -> int:
        """Return total bytes used under DATASET_DIR."""
        total = 0
        for dirpath, _, filenames in os.walk(DATASET_DIR):
            for fname in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fname))
                except OSError:
                    pass
        return total

    def _save_dataset_frame_for(self, entry: dict, frame: np.ndarray, distance: float = 0.0) -> None:
        """
        Save one face-detected BGR frame to dataset/{person_name}/, using
        THIS tracked face's own per-visit frame counter — independent of
        every other currently-tracked face, so several people being
        recognised in the same frame each get their own dataset-saving
        cadence and 50-frame-per-visit cap.
        """
        try:
            person_name = entry["person_name"]
            confidence = entry["confidence"]

            if entry["visit_frame_count"] >= DATASET_MAX_FRAMES_PER_VISIT:
                logger.debug("Per-visit frame limit reached, skipping save")
                return

            if entry["visit_frame_count"] % 10 == 0:
                if self._get_dataset_size() >= DATASET_MAX_BYTES:
                    logger.warning("Dataset 30 GB cap reached — stopping frame saves")
                    return

            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in person_name)
            person_dir = os.path.join(DATASET_DIR, safe_name)
            os.makedirs(person_dir, exist_ok=True)

            ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}ms"
            filename = f"{ts}_conf{confidence:.0f}.jpg"
            filepath = os.path.join(person_dir, filename)

            # frame is BGR — write directly so colours are correct
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            entry["visit_frame_count"] += 1
            logger.info(f"Dataset frame saved: {filepath} (conf={confidence:.1f}%, dist={distance:.4f})")
        except Exception as e:
            logger.error(f"Failed to save dataset frame: {e}")

    def _save_debug_frame(self, frame: np.ndarray, face_locations: list) -> None:
        """
        Save a debug frame with face bounding boxes drawn on it.

        Args:
            frame: RGB frame from camera
            face_locations: List of (top, right, bottom, left) tuples
        """
        logger.info(f"_save_debug_frame: frame={frame.shape}, faces={len(face_locations)}")
        self._frame_save_counter += 1

        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"Face {right - left}x{bottom - top}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )

            cv2.putText(
                frame_bgr,
                f"Frame #{self._frame_save_counter} - Faces: {len(face_locations)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

            filepath = os.path.join(DEBUG_FRAME_DIR, f"frame_{self._frame_save_counter:04d}.jpg")
            cv2.imwrite(filepath, frame_bgr)
            logger.info(f"Saved debug frame: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save debug frame: {e}", exc_info=True)
