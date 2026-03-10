"""Face recognition service - captures, recognizes, and identifies faces."""

import logging
import os
import threading
import time
from typing import Optional

import cv2
import face_recognition
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


class FaceRecognitionService(ServiceBase):
    """
    Face recognition service that continuously captures images,
    detects faces, and identifies them via API.

    Tracks person presence and emits events when person is detected or leaves.
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

        # Debug frame saving
        self._frame_save_counter = 0
        self._frames_since_last_save = 0
        self._max_debug_frames = 50  # Limit total frames to avoid filling disk
        self._save_every_n_frames = 2  # Save every Nth frame (change to 1 for every frame)
        if SAVE_DEBUG_FRAMES:
            os.makedirs(DEBUG_FRAME_DIR, exist_ok=True)
            logger.info(f"Debug frame saving enabled. Saving every {self._save_every_n_frames} frames (max {self._max_debug_frames} total)")
            logger.info(f"Debug frames will be saved to {DEBUG_FRAME_DIR}")

    def start(self) -> None:
        """Start the face recognition service."""
        if self._running:
            logger.warning("Face recognition service is already running")
            return

        logger.info("Starting face recognition service")
        logger.info(f"Configuration: detection_interval={self.config.detection_interval}s, "
                   f"min_face_size={self.config.min_face_size}px, n={self.config.n}")
        logger.info(f"API endpoint: {self.config.api_endpoint}")
        self._running = True
        self._stop_event.clear()

        # Initialize HTTP client
        self._http_client = HTTPClient(timeout=10.0, max_retries=3)

        # Initialize camera using Picamera2
        try:
            self._camera = Picamera2()
            # Configure camera for face recognition
            config = self._camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self._camera.configure(config)
            self._camera.start()
            logger.info("Picamera2 started successfully")
        except Exception as e:
            logger.error(f"Failed to start Picamera2: {e}")
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

        # Wait for thread to finish
        self._wait_for_thread(timeout=5.0)

        # Stop camera
        if self._camera:
            self._camera.stop()
            self._camera.close()
            self._camera = None

        # Close HTTP client
        if self._http_client:
            self._http_client.close()
            self._http_client = None

        logger.info("Face recognition service stopped")

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

            # Wait for the configured interval
            self._stop_event.wait(self.config.detection_interval)

        # Emit person.left event if someone was being tracked
        with self._person_lock:
            if self._current_visit_id:
                self._emit_person_left_event()

        logger.info("Face recognition loop ended")

    def _capture_and_process(self) -> None:
        """Capture frame, detect face, and process (identify or match locally)."""
        if not self._camera:
            logger.warning("Camera is not available")
            return

        # Capture frame from Picamera2 (already in RGB format)
        try:
            frame = self._camera.capture_array()
        except Exception as e:
            logger.warning(f"Failed to capture frame: {e}")
            return

        logger.info(f"Frame captured: {frame.shape[1]}x{frame.shape[0]} pixels, dtype={frame.dtype}, "
                   f"range=[{frame.min():.1f}, {frame.max():.1f}], detecting faces...")

        # Frame is already RGB from Picamera2 configuration
        rgb_frame = frame

        # Detect faces with timing
        logger.debug("Running HOG face detection model...")
        start_time = time.time()
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        detection_time = time.time() - start_time

        if not face_locations:
            logger.info(f"No faces detected in frame (detection took {detection_time:.2f}s)")

            # Save debug frames periodically to help diagnose issues
            if SAVE_DEBUG_FRAMES:
                self._frames_since_last_save += 1
                if self._frames_since_last_save >= self._save_every_n_frames:
                    if self._frame_save_counter < self._max_debug_frames:
                        self._save_debug_frame(rgb_frame, face_locations)
                    else:
                        logger.info(f"Reached max debug frames ({self._max_debug_frames}), not saving more")
                    self._frames_since_last_save = 0
            return

        logger.info(f"Detected {len(face_locations)} face(s) (detection took {detection_time:.2f}s)")

        # Save debug frame when face is detected
        if SAVE_DEBUG_FRAMES:
            if self._frame_save_counter < self._max_debug_frames:
                self._save_debug_frame(rgb_frame, face_locations)
            else:
                logger.info(f"Reached max debug frames ({self._max_debug_frames}), not saving more")

        # Find the largest face (by area)
        largest_face = max(
            face_locations,
            key=lambda loc: (loc[2] - loc[0]) * (loc[3] - loc[1]),
        )

        # Check if face is large enough
        top, right, bottom, left = largest_face
        face_width = right - left
        face_height = bottom - top

        if face_width < self.config.min_face_size or face_height < self.config.min_face_size:
            logger.info(f"Face too small: {face_width}x{face_height} (minimum: {self.config.min_face_size})")
            return

        logger.info(f"Using face at {left},{top} to {right},{bottom} (size: {face_width}x{face_height})")

        # Encode face to 128D vector
        logger.info("Encoding face to 128D vector...")
        face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face])

        if not face_encodings:
            logger.warning("Failed to encode face - face encoding returned empty result")
            return

        face_vector = face_encodings[0].tolist()
        logger.info(f"Face encoded successfully (128D vector, first 3 values: [{face_vector[0]:.4f}, {face_vector[1]:.4f}, {face_vector[2]:.4f}])")

        # Process the face vector
        self._process_face_vector(face_vector)

    def _process_face_vector(self, face_vector: list[float]) -> None:
        """
        Process face vector - either match locally or send to API for identification.

        Args:
            face_vector: 128D face embedding vector as list of floats
        """
        with self._person_lock:
            now = time.time()

            # Check if we have a cached face to match against
            if self._cached_face_vector is not None:
                logger.info("Cached face vector exists - performing local similarity check...")
                # Do local matching
                similarity = self._compute_face_similarity(face_vector, self._cached_face_vector)
                logger.info(f"Face similarity (cosine distance): {similarity:.4f} (threshold: {self._similarity_threshold})")

                if similarity < self._similarity_threshold:
                    # Same person - update last seen time
                    self._last_face_seen = now
                    logger.info(f"Same person: {self._current_person_name} (similarity: {similarity:.4f})")
                    return
                else:
                    # Different person - need to identify them
                    logger.info(f"Different person detected (similarity: {similarity:.4f} > threshold), sending to API...")
            else:
                logger.info("No cached face vector - sending to API for identification...")

            # No cached face or different person - send to API for identification
            self._send_identification_request(face_vector)

    def _compute_face_similarity(self, face_vector1: list[float], face_vector2: list[float]) -> float:
        """
        Compute cosine distance between two face vectors.

        Args:
            face_vector1: First face vector (128D)
            face_vector2: Second face vector (128D)

        Returns:
            Cosine distance (0 = identical, 1 = completely different)
        """
        try:
            return cosine(face_vector1, face_vector2)
        except Exception as e:
            logger.error(f"Error computing face similarity: {e}")
            return 1.0  # Return max distance if computation fails

    def _send_identification_request(self, face_vector: list[float]) -> None:
        """
        Send face identification request to API.

        Args:
            face_vector: 128D face embedding vector as list of floats
        """
        if not self._http_client:
            logger.warning("HTTP client not available")
            return

        try:
            # Create request with comma-separated vector string
            request = IdentifyRequest.from_vector_list(
                type="face",
                n=self.config.n,
                face_vector=face_vector,
            )

            request_data = request.model_dump()

            # Log request details (show vector preview, not full 128 values)
            vector_preview = f"{len(face_vector)}D vector, first 3: [{face_vector[0]:.4f}, {face_vector[1]:.4f}, {face_vector[2]:.4f}]"
            logger.info(f"Sending POST request to {self.config.api_endpoint}")
            logger.info(f"Request data: type={request_data['type']}, n={request_data['n']}, vector={vector_preview}")

            # Send as form-encoded data (application/x-www-form-urlencoded)
            response_data = self._http_client.post(
                self.config.api_endpoint,
                data=request_data,
            )

            # Log raw response body
            logger.info(f"Raw response body: {response_data}")

            # Parse response using the response model
            response = IdentifyResponse(**response_data)

            # Get confidence - API may return it as 0-1 or 0-100
            confidence = response.confidence or 0.0
            if confidence <= 1.0:
                confidence = confidence * 100

            logger.info(
                f"Identification result: {response.name} ({confidence:.1f}%) - "
                f"match_type: {response.match_type}, distance: {response.distance}"
            )

            # Update person tracking and emit events
            self._update_person_tracking(response.visit_id, face_vector, response.name, confidence)

            # Publish face.recognized event for overlay display
            if self.config.display_result and response.success:
                event = Event(
                    event_type="face.recognized",
                    data={
                        "person_name": response.person_name,
                        "confidence": confidence,
                        "hide_delay": self.config.overlay_hide_delay,
                    },
                )
                self.event_bus.publish(event)

        except Exception as e:
            logger.error(f"Failed to send identification request: {type(e).__name__}: {e}", exc_info=True)

    def _update_person_tracking(
        self,
        visit_id: Optional[str],
        face_vector: list[float],
        person_name: Optional[str],
        confidence: float
    ) -> None:
        """
        Update person tracking state and emit appropriate events.

        Args:
            visit_id: Visit ID from identification response
            face_vector: Face vector to cache
            person_name: Person name from identification response
            confidence: Confidence score
        """
        now = time.time()

        # Check if this is the same person (by visit_id)
        if visit_id and visit_id == self._current_visit_id:
            # Same person - update cache and last seen time
            self._cached_face_vector = face_vector
            self._last_face_seen = now
            logger.debug(f"Updated cached vector for {person_name} (visit_id: {visit_id})")
            return

        # New person detected
        if visit_id:
            # If we were tracking someone, emit person.left first
            if self._current_visit_id:
                self._emit_person_left_event()

            # Set new person
            self._current_visit_id = visit_id
            self._current_person_name = person_name
            self._cached_face_vector = face_vector
            self._last_face_seen = now

            # Emit person.detected event
            self._emit_person_detected_event(visit_id, person_name, confidence)

    def _check_person_timeout(self) -> None:
        """Check if the current person has left (no face detected for timeout period)."""
        with self._person_lock:
            if not self._current_visit_id:
                return

            now = time.time()
            time_since_last_seen = now - self._last_face_seen

            if time_since_last_seen >= self.config.person_timeout:
                logger.info(f"Person timeout: {self._current_person_name} not seen for {time_since_last_seen:.1f}s")
                self._emit_person_left_event()

    def _emit_person_detected_event(self, visit_id: str, person_name: str, confidence: float) -> None:
        """Emit person.detected event."""
        event = Event(
            event_type="person.detected",
            data={
                "visit_id": visit_id,
                "person_name": person_name,
                "confidence": confidence,
            },
        )
        self.event_bus.publish(event)
        logger.info(f"Emitted person.detected event: {person_name} (visit_id: {visit_id})")

    def _emit_person_left_event(self) -> None:
        """Emit person.left event and clear tracking state."""
        if not self._current_visit_id:
            return

        event = Event(
            event_type="person.left",
            data={
                "visit_id": self._current_visit_id,
                "person_name": self._current_person_name,
            },
        )
        self.event_bus.publish(event)
        logger.info(f"Emitted person.left event: {self._current_person_name} (visit_id: {self._current_visit_id})")

        # Clear tracking state
        self._current_visit_id = None
        self._current_person_name = None
        self._cached_face_vector = None
        self._last_face_seen = 0.0

    def _save_debug_frame(self, frame: np.ndarray, face_locations: list) -> None:
        """
        Save a debug frame with face boxes drawn on it.

        Args:
            frame: RGB frame from camera
            face_locations: List of face location tuples (top, right, bottom, left)
        """
        logger.info(f"_save_debug_frame called: frame shape={frame.shape}, faces={len(face_locations)}")
        self._frame_save_counter += 1

        try:
            # Convert to BGR for OpenCV (it uses BGR by default)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw face boxes
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Face {right-left}x{bottom-top}",
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add info text
            info_text = f"Frame #{self._frame_save_counter} - Faces: {len(face_locations)}"
            cv2.putText(frame_bgr, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save frame
            filepath = os.path.join(DEBUG_FRAME_DIR, f"frame_{self._frame_save_counter:04d}.jpg")
            cv2.imwrite(filepath, frame_bgr)
            logger.info(f"Saved debug frame to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save debug frame: {e}", exc_info=True)
