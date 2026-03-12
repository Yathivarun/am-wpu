"""Slideshow service - displays images in full-screen mode."""

import glob
import logging
import os
import threading
import time
from typing import Optional

import numpy as np
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GLib, GdkPixbuf, Gtk, Gio

from wpu_client.config.settings import SlideshowConfig
from wpu_client.core.events import Event, EventBus
from wpu_client.core.service_base import ServiceBase
from wpu_client.utils.http import HTTPClient

logger = logging.getLogger(__name__)


class SlideshowService(ServiceBase):
    """
    GTK4-based slideshow service.

    Supports two modes:
    - Stock images mode: Default slideshow from local directory
    - Visitor images mode: Slideshow of visitor-specific WPU images
    """

    def __init__(self, config: SlideshowConfig, event_bus: EventBus, wpu_endpoint: str, face_service=None):
        """
        Initialize the slideshow service.

        Args:
            config: Slideshow configuration
            event_bus: Event bus for inter-service communication
            wpu_endpoint: WPU API endpoint for fetching visitor images
            face_service: Optional FaceRecognitionService for camera preview
        """
        super().__init__("slideshow")
        self.config = config
        self.event_bus = event_bus
        self.wpu_endpoint = wpu_endpoint
        self.face_service = face_service  # used for bottom-left camera preview
        self._app: Optional[SlideshowApp] = None
        self._overlay_text: Optional[str] = None
        self._overlay_hide_time: float = 0
        self._overlay_lock = threading.Lock()

        # Mode tracking
        self._mode: str = "stock"  # "stock" or "visitor"
        self._mode_lock = threading.Lock()
        self._current_visit_id: Optional[str] = None

        # HTTP client for fetching WPU images
        self._http_client: Optional[HTTPClient] = None

        # Subscribe to events
        event_bus.subscribe("face.recognized", self._on_face_recognized)
        event_bus.subscribe("person.detected", self._on_person_detected)
        event_bus.subscribe("person.left", self._on_person_left)

    def start(self) -> None:
        """Start the slideshow service."""
        if self._running:
            logger.warning("Slideshow service is already running")
            return

        logger.info("Starting slideshow service")
        self._running = True

        # Initialize HTTP client for fetching WPU images
        self._http_client = HTTPClient(timeout=10.0, max_retries=3)

        # Run GTK app in main thread
        self._app = SlideshowApp(self.config, self, self.wpu_endpoint, self._http_client, self.face_service)
        self._app.run(None)

    def stop(self) -> None:
        """Stop the slideshow service."""
        if not self._running:
            return

        logger.info("Stopping slideshow service")
        self._running = False

        if self._app:
            self._app.quit()
            self._app = None

        if self._http_client:
            self._http_client.close()
            self._http_client = None

    def _on_face_recognized(self, event: Event) -> None:
        """Handle face recognition event - show overlay."""
        person_name = event.data.get("person_name", "Unknown")
        confidence = event.data.get("confidence", 0.0)
        hide_delay = event.data.get("hide_delay", 3)

        with self._overlay_lock:
            self._overlay_text = f"Identified: {person_name} ({confidence:.1f}%)"
            self._overlay_hide_time = time.time() + hide_delay

        logger.info(f"Displaying overlay: {self._overlay_text}")

    def _on_person_detected(self, event: Event) -> None:
        """Handle person detected event - fetch visitor images and switch mode."""
        visit_id = event.data.get("visit_id")
        person_name = event.data.get("person_name", "Unknown")

        if not visit_id:
            logger.warning("person.detected event missing visit_id")
            return

        logger.info(f"Person detected: {person_name} (visit_id: {visit_id})")

        # Fetch WPU images and switch mode
        if self._app:
            self._app.switch_to_visitor_mode(visit_id, person_name)

    def _on_person_left(self, event: Event) -> None:
        """Handle person left event - switch back to stock images."""
        person_name = event.data.get("person_name", "Unknown")
        logger.info(f"Person left: {person_name}")

        # Switch back to stock mode
        if self._app:
            self._app.switch_to_stock_mode()

    def get_overlay_text(self) -> Optional[str]:
        """Get current overlay text if it should be displayed."""
        with self._overlay_lock:
            if self._overlay_text and time.time() < self._overlay_hide_time:
                return self._overlay_text
            elif self._overlay_text and time.time() >= self._overlay_hide_time:
                self._overlay_text = None
        return None


class SlideshowWindow(Gtk.ApplicationWindow):
    """GTK window for slideshow display."""

    def __init__(self, app, image_files, config, service, mode="stock"):
        super().__init__(application=app)
        self.image_files = image_files
        self.config = config
        self.service = service
        self.current_index = 0
        self.mode = mode  # "stock" or "visitor"

        # Set up full screen or windowed
        if config.full_screen:
            self.fullscreen()
        else:
            self.set_default_size(800, 600)

        # Main container box
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        # Create picture widget for images
        self.picture = Gtk.Picture()
        self.picture.set_vexpand(True)
        self.picture.set_hexpand(True)

        # Set scaling based on config
        scale_mode = config.scale_mode
        if scale_mode == "fill":
            self.picture.set_content_fit(Gtk.ContentFit.FILL)
        elif scale_mode == "fit":
            self.picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        elif scale_mode == "crop":
            self.picture.set_content_fit(Gtk.ContentFit.COVER)
        else:
            self.picture.set_content_fit(Gtk.ContentFit.FILL)

        # Create overlay label for face recognition results
        self.overlay_label = Gtk.Label()
        self.overlay_label.set_halign(Gtk.Align.START)
        self.overlay_label.set_valign(Gtk.Align.START)
        self.overlay_label.set_margin_start(20)
        self.overlay_label.set_margin_top(20)

        # --- Camera preview (bottom-left) ---
        self.camera_preview = Gtk.Picture()
        self.camera_preview.set_halign(Gtk.Align.START)
        self.camera_preview.set_valign(Gtk.Align.END)
        self.camera_preview.set_margin_start(16)
        self.camera_preview.set_margin_bottom(16)
        self.camera_preview.set_size_request(320, 240)
        self.camera_preview.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.camera_preview.set_visible(False)  # hidden until first frame arrives

        # Apply CSS for styling
        provider = Gtk.CssProvider()
        css = f"""
        window {{
            background-color: {config.background_color};
        }}
        label {{
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            font-size: 18px;
        }}
        picture.camera-preview {{
            border: 2px solid rgba(255, 255, 255, 0.6);
            border-radius: 6px;
            background-color: rgba(0, 0, 0, 0.5);
        }}
        """.encode()
        provider.load_from_data(css)
        display = self.get_display()
        Gtk.StyleContext.add_provider_for_display(
            display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        self.camera_preview.add_css_class("camera-preview")

        # Use GtkOverlay for overlay positioning
        overlay = Gtk.Overlay()
        overlay.set_child(self.picture)
        overlay.add_overlay(self.overlay_label)
        overlay.add_overlay(self.camera_preview)

        self.main_box.append(overlay)
        self.set_child(self.main_box)

        # Load and display first image
        if self.image_files:
            self.load_image(0)

        # Set up key press controller
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

        # Start the slideshow timer
        self.timeout_id = GLib.timeout_add_seconds(config.advance_time, self.next_image)

        # Start overlay update timer
        self.overlay_timeout_id = GLib.timeout_add(100, self.update_overlay)

        # Start camera preview update timer — 100ms refresh (~10fps, looks live)
        self.preview_timeout_id = GLib.timeout_add(100, self.update_camera_preview)

    def set_images(self, new_images: list[str], mode: str):
        """
        Update the image list and mode.

        Args:
            new_images: New list of image paths/URLs
            mode: "stock" or "visitor"
        """
        self.image_files = new_images
        self.mode = mode
        self.current_index = 0

        if self.image_files:
            self.load_image(0)

        logger.info(f"Updated image list: {len(new_images)} images, mode={mode}")

    def load_image(self, index):
        """Load and display image at given index."""
        if not self.image_files or index >= len(self.image_files):
            return True

        try:
            image_path = self.image_files[index]

            # Check if this is a URL (visitor mode) or local file (stock mode)
            if image_path.startswith("http://") or image_path.startswith("https://"):
                # Load from URL using httpx and GdkPixbuf
                logger.debug(f"Loading image from URL")
                response = self.service._http_client._client.get(image_path)
                response.raise_for_status()

                # Load image from bytes using GdkPixbuf
                loader = GdkPixbuf.PixbufLoader()
                loader.write(response.content)
                loader.close()
                pixbuf = loader.get_pixbuf()
                texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            else:
                # Load from local file
                texture = Gdk.Texture.new_from_filename(image_path)

            self.picture.set_paintable(texture)
            self.current_index = index
            logger.debug(f"Displaying: {os.path.basename(image_path) if not image_path.startswith('http') else 'visitor image'}")
        except Exception as e:
            logger.error(f"Error loading image {index}: {e}")

        return True

    def next_image(self):
        """Advance to next image."""
        if not self.image_files:
            return True
        next_index = (self.current_index + 1) % len(self.image_files)
        self.load_image(next_index)
        return True  # Keep timer running

    def previous_image(self):
        """Go to previous image."""
        if not self.image_files:
            return
        prev_index = (self.current_index - 1) % len(self.image_files)
        self.load_image(prev_index)
        # Reset timer
        GLib.source_remove(self.timeout_id)
        self.timeout_id = GLib.timeout_add_seconds(self.config.advance_time, self.next_image)

    def update_overlay(self):
        """Update overlay label with current face recognition result."""
        overlay_text = self.service.get_overlay_text()
        if overlay_text:
            self.overlay_label.set_label(overlay_text)
            self.overlay_label.set_visible(True)
        else:
            self.overlay_label.set_visible(False)
        return True  # Keep timer running

    def update_camera_preview(self):
        """Refresh the bottom-left camera preview from the latest captured frame."""
        face_service = getattr(self.service, "face_service", None)
        if face_service is None:
            return True  # No face service wired up — skip silently

        frame = face_service.get_preview_frame()
        if frame is None:
            return True  # No frame yet

        try:
            # frame is RGB uint8 numpy array (H, W, 3)
            h, w = frame.shape[:2]
            rowstride = w * 3
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False,   # has_alpha
                8,       # bits_per_sample
                w, h,
                rowstride,
            )
            texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            self.camera_preview.set_paintable(texture)
            self.camera_preview.set_visible(True)
        except Exception as e:
            logger.debug(f"Camera preview update failed: {e}")

        return True  # Keep timer running

    def on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events."""
        if keyval == Gdk.KEY_Escape:
            self.close()
        elif keyval == Gdk.KEY_space or keyval == Gdk.KEY_Right:
            # Next image
            GLib.source_remove(self.timeout_id)
            self.next_image()
            self.timeout_id = GLib.timeout_add_seconds(self.config.advance_time, self.next_image)
        elif keyval == Gdk.KEY_Left:
            # Previous image
            self.previous_image()
        elif keyval == Gdk.KEY_f:
            # Toggle fullscreen
            if self.is_fullscreen():
                self.unfullscreen()
            else:
                self.fullscreen()
        return False


class SlideshowApp(Gtk.Application):
    """
    GTK application for slideshow.

    Manages switching between stock and visitor image modes.
    """

    def __init__(self, config, service, wpu_endpoint, http_client, face_service=None):
        super().__init__(application_id="com.wpu_client.slideshow")
        self.config = config
        self.service = service
        self.wpu_endpoint = wpu_endpoint
        self.http_client = http_client
        self.face_service = face_service
        self.stock_images = self._load_stock_images()
        self.visitor_images: list[str] = []
        self.current_window: Optional[SlideshowWindow] = None

    def _load_stock_images(self):
        """Load all stock image files from configured directory."""
        image_dir = self.config.image_directory
        image_extensions = self.config.image_extensions
        image_files = []

        for ext in image_extensions:
            pattern = os.path.join(image_dir, ext)
            image_files.extend(glob.glob(pattern))

        if not image_files:
            logger.error(f"No images found in {image_dir}")
            return []

        # Sort images based on configuration
        if self.config.sort_mode == "numeric":
            image_files.sort(
                key=lambda p: (
                    int(os.path.splitext(os.path.basename(p))[0])
                    if os.path.splitext(os.path.basename(p))[0].isdigit()
                    else os.path.basename(p)
                )
            )
        else:
            image_files.sort(key=lambda p: os.path.basename(p).lower())

        logger.info(f"Loaded {len(image_files)} stock images from {image_dir}")
        return image_files

    def _fetch_visitor_images(self, visit_id: str) -> list[str]:
        """Fetch visitor WPU images from the API."""
        try:
            logger.info(f"Fetching WPU images for visit_id: {visit_id}")
            response_data = self.http_client.get(
                self.wpu_endpoint,
                params={"visit_uuid": visit_id}
            )

            # The API returns a dict with "signed_urls" key containing list of URLs
            images = response_data.get("signed_urls", [])
            logger.info(f"Fetched {len(images)} WPU images for visit {visit_id}")
            return images

        except Exception as e:
            logger.error(f"Failed to fetch WPU images for visit {visit_id}: {e}")
            return []

    def switch_to_visitor_mode(self, visit_id: str, person_name: str):
        """Switch to visitor-specific images mode."""
        logger.info(f"Switching to visitor mode: {person_name} (visit_id: {visit_id})")

        # Fetch visitor images
        visitor_images = self._fetch_visitor_images(visit_id)

        if not visitor_images:
            logger.warning(f"No WPU images found for visit {visit_id}, staying in stock mode")
            return

        self.visitor_images = visitor_images

        # Update the existing window with new images
        if self.current_window:
            self.current_window.set_images(self.visitor_images, mode="visitor")

        # Update overlay to show mode change
        with self.service._overlay_lock:
            self.service._overlay_text = f"Welcome {person_name}!"
            self.service._overlay_hide_time = time.time() + 3

    def switch_to_stock_mode(self):
        """Switch back to stock images mode."""
        logger.info("Switching to stock mode")

        # Update the existing window with stock images
        if self.current_window:
            self.current_window.set_images(self.stock_images, mode="stock")

        # Clear visitor images
        self.visitor_images = []

    def do_activate(self):
        """Activate the application and show the window."""
        if not self.stock_images:
            logger.error("No stock images to display")
            return

        # Make face_service accessible to SlideshowWindow via the service reference
        self.service.face_service = self.face_service

        self.current_window = SlideshowWindow(
            self,
            self.stock_images,
            self.config,
            self.service,
            mode="stock"
        )
        self.current_window.present()
