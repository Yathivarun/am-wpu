"""Face recognition service - captures, recognizes, and identifies faces."""

import csv
import logging
import os
import threading
import time
from typing import Optional

# Dataset directory (relative to project root — three levels up from this file)
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset")
DATASET_MAX_BYTES = 30 * 1024 * 1024 * 1024  # 30 GB hard cap
DATASET_MAX_FRAMES_PER_VISIT = 50             # max frames saved per visit

import cv2
import numpy as np
from picamera2 import Picamera2
from scipy.spatial.distance import cosine

from wpu_client.config.settings import FaceRecognitionConfig
from wpu_client.core.events import Event, EventBus
from wpu_client.core.service_base import ServiceBase
from wpu_client.models.api import IdentifyRequest, IdentifyResponse
from wpu_client.utils.http import HTTPClient

logger = logging.getLogger(__name__)

# Enable debug frame saving by setting environment variable: SAVE_DEBUG_FRAMES=1
SAVE_DEBUG_FRAMES = os.getenv("SAVE_DEBUG_FRAMES", "0") == "1"
DEBUG_FRAME_DIR = "/tmp/face_detection_debug"

# Model paths
YUNET_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_detection_yunet_2023mar.onnx",
)

GLINTR100_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "glintr100_int8_static_150.onnx",
)


class FaceRecognitionService(ServiceBase):
    """
    Face recognition service that continuously captures images,
    detects faces, and identifies them via API.

    Tracks person presence and emits events when person is detected or leaves.
    Embeddings are generated locally on the Pi using YuNet (detection) +
    GlintR100-INT8 (recognition). The embedding vector is sent to the server
    API only for identity lookup against the known-faces database.
    Uses cached face vector for local matching after initial recognition.
    """

    def __init__(self, config: FaceRecognitionConfig, event_bus: EventBus):
        """
        Initialize the face recognition service.

        Args:
            config: Face recognition configuration
            event_bus: Event bus for inter-service communication
        """
        super().__init__("face_recognition")
        self.config = config
        self.event_bus = event_bus
        self._camera: Optional[Picamera2] = None
        self._http_client: Optional[HTTPClient] = None

        # Person tracking
        self._current_visit_id: Optional[str] = None
        self._current_person_name: Optional[str] = None
        self._cached_face_vector: Optional[list[float]] = None
        self._last_face_seen: float = 0.0
        self._person_lock = threading.Lock()

        # Similarity threshold for local matching (lower = more strict)
        # Cosine distance: 0 = identical, 1 = completely different
        self._similarity_threshold = 0.4

        # Dataset / visit-log setup
        os.makedirs(DATASET_DIR, exist_ok=True)
        self._visit_log_path = os.path.join(DATASET_DIR, "visit_log.csv")
        self._init_visit_log()
        self._visit_start_time: Optional[float] = None
        self._visit_confidence: float = 0.0
        self._visit_frame_count: int = 0

        # Debug frame saving
        self._frame_save_counter = 0
        self._frames_since_last_save = 0
        self._max_debug_frames = 50
        self._save_every_n_frames = 2
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
            logger.info(f"Debug frame saving enabled. Saving every {self._save_every_n_frames} frames (max {self._max_debug_frames} total)")
            logger.info(f"Debug frames will be saved to {DEBUG_FRAME_DIR}")

        # Models (initialised in start())
        self._yunet_detector: Optional[cv2.FaceDetectorYN] = None
        self._glintr100_session = None

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
        try:
            self._camera = Picamera2()
            config = self._camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
            )
            self._camera.configure(config)
            self._camera.start()
            logger.info("Picamera2 started successfully (640×480 RGB888)")
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

        # Initialize GlintR100-INT8 recogniser
        # GlintR100 is an ArcFace-family model (Glint360K) — same preprocessing
        # as AuraFace: resize to 112×112, normalise with (x - 127.5) / 127.5,
        # NCHW layout, L2-normalise output embedding.
        try:
            import onnxruntime as ort
            self._glintr100_session = ort.InferenceSession(
                GLINTR100_MODEL_PATH,
                providers=["CPUExecutionProvider"],
            )
            inp = self._glintr100_session.get_inputs()[0]
            logger.info(
                f"GlintR100 recogniser initialised: {os.path.basename(GLINTR100_MODEL_PATH)} "
                f"| input: {inp.name} {inp.shape} {inp.type}"
            )
        except Exception as e:
            logger.error(f"Failed to initialise GlintR100: {e}")
            self._running = False
            return

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

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """
        Return a live RGB frame from the main camera stream.
        Picamera2 returns BGR byte order despite RGB888 config — converted here.
        Returns None if camera is not ready.
        """
        if not self._camera or not self._running:
            return None
        try:
            bgr = self._camera.capture_array("main")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
            except Exception as e:
                logger.error(f"Error in recognition loop: {e}", exc_info=True)

            self._stop_event.wait(self.config.detection_interval)

        # Emit person.left event if someone was being tracked when we stopped
        with self._person_lock:
            if self._current_visit_id:
                self._emit_person_left_event()

        logger.info("Face recognition loop ended")

    def _capture_and_process(self) -> None:
        """Capture frame, detect face, generate embedding locally, then identify."""
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
        start_time = time.time()
        _, faces = self._yunet_detector.detect(bgr_frame)
        detection_time = time.time() - start_time

        # Convert YuNet [x, y, w, h, ...] → dlib CSS (top, right, bottom, left)
        face_locations = []
        if faces is not None:
            coord_limit = 10 * max(h, w)
            for row in faces:
                arr = np.asarray(row, dtype=np.float64)
                if not np.all(np.isfinite(arr)):
                    continue
                if float(np.max(np.abs(arr[:14]))) > coord_limit:
                    continue
                x, y, fw, fh = arr[:4]
                face_locations.append((int(y), int(x + fw), int(y + fh), int(x)))

        if not face_locations:
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

        logger.info(f"Detected {len(face_locations)} face(s) (took {detection_time:.2f}s)")

        if SAVE_DEBUG_FRAMES and self._frame_save_counter < self._max_debug_frames:
            self._save_debug_frame(rgb_frame, face_locations)

        # Use the largest face
        largest_face = max(
            face_locations,
            key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]),
        )

        top, right, bottom, left = largest_face
        face_width = right - left
        face_height = bottom - top

        if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
            logger.info(f"Face too small: {face_width}x{face_height} (minimum: {self.config.min_face_size})")
            return

        logger.info(f"Using face at ({left},{top})→({right},{bottom}), size: {face_width}x{face_height}")

        # ── Embedding: GlintR100-INT8 (ArcFace, Glint360K) ────────────────
        # Preprocessing identical to AuraFace / standard ArcFace:
        #   • crop face region from RGB frame
        #   • resize to 112×112
        #   • normalise: (pixel - 127.5) / 127.5  →  range [-1, 1]
        #   • layout: NCHW (1, 3, 112, 112)
        #   • L2-normalise output vector before comparison / sending to API
        logger.info("Encoding face with GlintR100-INT8...")
        face_crop = rgb_frame[top:bottom, left:right]
        face_resized = cv2.resize(face_crop, (112, 112))
        face_input = (face_resized.astype(np.float32) - 127.5) / 127.5
        face_input = np.transpose(face_input, (2, 0, 1))[np.newaxis, :]  # (1, 3, 112, 112)

        input_name = self._glintr100_session.get_inputs()[0].name
        face_array = self._glintr100_session.run(None, {input_name: face_input})[0][0]

        # L2-normalise (standard for ArcFace-family models)
        face_array = face_array / (np.linalg.norm(face_array) + 1e-8)
        face_vector = face_array.tolist()

        logger.info(
            f"Face encoded ({len(face_vector)}D, "
            f"first 3: [{face_vector[0]:.4f}, {face_vector[1]:.4f}, {face_vector[2]:.4f}])"
        )

        # bgr_frame passed on so dataset frames are saved with correct colours
        self._process_face_vector(face_vector, bgr_frame)

    def _process_face_vector(self, face_vector: list[float], frame: np.ndarray) -> None:
        """
        Process face vector — local cosine match if we have a cached vector for
        the current visitor, otherwise forward to the server for identity lookup.

        Args:
            face_vector: 512D face embedding (L2-normalised)
            frame: BGR frame — saved to dataset once we have name + confidence
        """
        with self._person_lock:
            now = time.time()

            if self._cached_face_vector is not None:
                logger.info("Cached face vector exists — performing local similarity check...")
                similarity = self._compute_face_similarity(face_vector, self._cached_face_vector)
                logger.info(
                    f"Cosine distance: {similarity:.4f} (threshold: {self._similarity_threshold})"
                )

                if similarity < self._similarity_threshold:
                    # Same person — just update timestamp and save frame
                    self._last_face_seen = now
                    confidence = (1.0 - similarity) * 100
                    logger.info(f"Same person confirmed: {self._current_person_name} (dist={similarity:.4f})")
                    self._save_dataset_frame(frame, confidence=confidence, distance=similarity)
                    return
                else:
                    logger.info(
                        f"Different person detected (dist={similarity:.4f} > threshold) "
                        f"— forwarding to API for identification..."
                    )
            else:
                logger.info("No cached face vector — forwarding to API for identification...")

            # Send embedding to server for identity lookup
            self._send_identification_request(face_vector, frame)

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

    def _send_identification_request(self, face_vector: list[float], frame: np.ndarray) -> None:
        """
        Send locally-generated face embedding to the server API for identity lookup.
        No face recognition happens server-side — it only matches the vector against
        its known-faces database and returns a name + confidence score.

        Args:
            face_vector: 512D L2-normalised embedding vector
            frame: BGR frame — saved to dataset after we receive name + confidence
        """
        if not self._http_client:
            logger.warning("HTTP client not available")
            return

        try:
            request = IdentifyRequest.from_vector_list(
                type="face",
                n=self.config.n,
                face_vector=face_vector,
            )
            request_data = request.model_dump()

            vector_preview = (
                f"{len(face_vector)}D vector, "
                f"first 3: [{face_vector[0]:.4f}, {face_vector[1]:.4f}, {face_vector[2]:.4f}]"
            )
            logger.info(f"Sending POST request to {self.config.api_endpoint}")
            logger.info(f"Request: type={request_data['type']}, n={request_data['n']}, vector={vector_preview}")

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

            self._update_person_tracking(response.visit_id, face_vector, response.name, confidence)
            self._save_dataset_frame(frame, confidence=confidence, distance=response.distance or 0.0)

            if self.config.display_result and response.success:
                self.event_bus.publish(Event(
                    event_type="face.recognized",
                    data={
                        "person_name": response.person_name,
                        "confidence": confidence,
                        "hide_delay": self.config.overlay_hide_delay,
                    },
                ))

        except Exception as e:
            logger.error(
                f"Failed to send identification request: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def _update_person_tracking(
        self,
        visit_id: Optional[str],
        face_vector: list[float],
        person_name: Optional[str],
        confidence: float,
    ) -> None:
        """
        Update person tracking state and emit appropriate events.

        Args:
            visit_id: Visit ID from identification response
            face_vector: Face vector to cache for subsequent local matching
            person_name: Person name from identification response
            confidence: Confidence score (0–100)
        """
        now = time.time()

        if visit_id and visit_id == self._current_visit_id:
            # Same visit — refresh cache and timestamp
            self._cached_face_vector = face_vector
            self._last_face_seen = now
            logger.debug(f"Updated cached vector for {person_name} (visit_id: {visit_id})")
            return

        # New person
        if visit_id:
            if self._current_visit_id:
                self._emit_person_left_event()

            self._current_visit_id = visit_id
            self._current_person_name = person_name
            self._cached_face_vector = face_vector
            self._last_face_seen = now
            self._emit_person_detected_event(visit_id, person_name, confidence)

    def _check_person_timeout(self) -> None:
        """Emit person.left if no face has been seen within the timeout window."""
        with self._person_lock:
            if not self._current_visit_id:
                return

            time_since_last_seen = time.time() - self._last_face_seen
            if time_since_last_seen >= self.config.person_timeout:
                logger.info(
                    f"Person timeout: {self._current_person_name} "
                    f"not seen for {time_since_last_seen:.1f}s"
                )
                self._emit_person_left_event()

    def _emit_person_detected_event(
        self, visit_id: str, person_name: str, confidence: float
    ) -> None:
        """Emit person.detected event and start visit tracking."""
        self._visit_start_time = time.time()
        self._visit_confidence = confidence
        self._visit_frame_count = 0
        self.event_bus.publish(Event(
            event_type="person.detected",
            data={
                "visit_id": visit_id,
                "person_name": person_name,
                "confidence": confidence,
            },
        ))
        logger.info(f"Emitted person.detected: {person_name} (visit_id: {visit_id})")

    def _emit_person_left_event(self) -> None:
        """Emit person.left event, write visit log row, and clear tracking state."""
        if not self._current_visit_id:
            return

        self._log_visit(
            person_name=self._current_person_name or "unknown",
            visit_id=self._current_visit_id,
            confidence=self._visit_confidence,
        )

        self.event_bus.publish(Event(
            event_type="person.left",
            data={
                "visit_id": self._current_visit_id,
                "person_name": self._current_person_name,
            },
        ))
        logger.info(
            f"Emitted person.left: {self._current_person_name} "
            f"(visit_id: {self._current_visit_id})"
        )

        self._current_visit_id = None
        self._current_person_name = None
        self._cached_face_vector = None
        self._last_face_seen = 0.0
        self._visit_start_time = None
        self._visit_confidence = 0.0
        self._visit_frame_count = 0

    def _init_visit_log(self) -> None:
        """Create visit_log.csv with header row if it does not already exist."""
        if not os.path.exists(self._visit_log_path):
            with open(self._visit_log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "person_name", "visit_id",
                    "timestamp_in", "timestamp_out",
                    "confidence", "frames_seen", "duration_seconds",
                ])
            logger.info(f"Created visit log: {self._visit_log_path}")

    def _log_visit(self, person_name: str, visit_id: str, confidence: float) -> None:
        """Append one completed-visit row to visit_log.csv."""
        try:
            now = time.time()
            ts_in = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._visit_start_time or now))
            ts_out = time.strftime("%Y-%m-%d %H:%M:%S")
            duration = round(now - (self._visit_start_time or now), 1)
            with open(self._visit_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    person_name, visit_id,
                    ts_in, ts_out,
                    f"{confidence:.1f}", self._visit_frame_count, duration,
                ])
            logger.info(
                f"Visit logged: {person_name} | {self._visit_frame_count} frames | "
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

    def _save_dataset_frame(self, frame: np.ndarray, confidence: float, distance: float) -> None:
        """
        Save one face-detected BGR frame to dataset/{person_name}/.
        Skips if the 30 GB cap is reached or the per-visit frame limit is exceeded.
        """
        try:
            person_name = self._current_person_name or "unknown"
            visit_id    = self._current_visit_id   or "no_visit"

            if self._visit_frame_count >= DATASET_MAX_FRAMES_PER_VISIT:
                logger.debug("Per-visit frame limit reached, skipping save")
                return

            if self._visit_frame_count % 10 == 0:
                if self._get_dataset_size() >= DATASET_MAX_BYTES:
                    logger.warning("Dataset 30 GB cap reached — stopping frame saves")
                    return

            safe_name  = "".join(c if c.isalnum() or c in "-_" else "_" for c in person_name)
            person_dir = os.path.join(DATASET_DIR, safe_name)
            os.makedirs(person_dir, exist_ok=True)

            ts       = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}ms"
            filename = f"{ts}_conf{confidence:.0f}.jpg"
            filepath = os.path.join(person_dir, filename)

            # frame is BGR — write directly so colours are correct
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self._visit_frame_count += 1
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