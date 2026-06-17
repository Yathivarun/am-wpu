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

    def __init__(self, config: SlideshowConfig, event_bus: EventBus, wpu_endpoint: str,
                 face_service=None, diagnostic_mode: bool = False):
        """
        Initialize the slideshow service.

        Args:
            config: Slideshow configuration
            event_bus: Event bus for inter-service communication
            wpu_endpoint: WPU API endpoint for fetching visitor images
            face_service: Optional FaceRecognitionService for camera preview
            diagnostic_mode: If True, show the matched person's own local sketches
                (diagnostic_gallery/<slug>/sketches/) on recognition instead of
                fetching visitor images from the server.
        """
        super().__init__("slideshow")
        self.config = config
        self.event_bus = event_bus
        self.wpu_endpoint = wpu_endpoint
        self.face_service = face_service  # used for bottom-left camera preview
        self.diagnostic_mode = diagnostic_mode
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
        sketch_dir = event.data.get("sketch_dir")

        if not visit_id:
            logger.warning("person.detected event missing visit_id")
            return

        logger.info(f"Person detected: {person_name} (visit_id: {visit_id}, sketches: {sketch_dir})")

        # Fetch WPU images and switch mode
        if self._app:
            self._app.switch_to_visitor_mode(visit_id, person_name, sketch_dir)

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

        # Content fit is set dynamically per image in load_image() so that
        # portrait images use COVER (fills screen, clips edges, no stretch)
        # while landscape images respect the operator-configured scale_mode.
        # Set a safe default until the first image loads.
        self.picture.set_content_fit(Gtk.ContentFit.CONTAIN)

        # Create overlay label for face recognition results
        self.overlay_label = Gtk.Label()
        self.overlay_label.set_halign(Gtk.Align.START)
        self.overlay_label.set_visible(False)

        # --- Camera preview ---
        self.camera_preview = Gtk.Picture()
        self.camera_preview.set_size_request(320, 240)
        self.camera_preview.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.camera_preview.set_visible(False)  # hidden until first frame arrives

        # Stack preview + label in a vertical box — both anchored top-left
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.preview_box.set_halign(Gtk.Align.START)
        self.preview_box.set_valign(Gtk.Align.START)
        self.preview_box.set_margin_start(16)
        self.preview_box.set_margin_top(16)
        self.preview_box.append(self.camera_preview)
        self.preview_box.append(self.overlay_label)

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

        # Use GtkOverlay — single preview_box overlay contains both widgets
        overlay = Gtk.Overlay()
        overlay.set_child(self.picture)
        overlay.add_overlay(self.preview_box)

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

    def _get_content_fit_for_pixbuf(self, pixbuf: GdkPixbuf.Pixbuf) -> Gtk.ContentFit:
        """
        Determine the appropriate ContentFit mode based on image orientation.

        Portrait images (taller than wide) are displayed with COVER so they fill
        the full TV screen without any stretching — the narrow sides are cropped
        minimally.  Landscape images use the operator-configured scale_mode so
        their existing appearance is unchanged.

        Args:
            pixbuf: The loaded GdkPixbuf whose dimensions we inspect.

        Returns:
            Gtk.ContentFit value to apply to self.picture.
        """
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        is_portrait = height > width

        if is_portrait:
            # COVER fills the screen while preserving aspect ratio; any overflow
            # is clipped rather than stretched, which is the correct TV behaviour
            # for a portrait photo.
            return Gtk.ContentFit.CONTAIN

        # Landscape — honour the operator's configured preference
        scale_mode = self.config.scale_mode
        if scale_mode == "fill":
            return Gtk.ContentFit.FILL
        elif scale_mode == "fit":
            return Gtk.ContentFit.CONTAIN
        elif scale_mode == "crop":
            return Gtk.ContentFit.COVER
        else:
            return Gtk.ContentFit.FILL

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
            else:
                # Load from local file into a pixbuf so we can inspect dimensions
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)

            # Adjust content-fit dynamically based on whether the image is
            # portrait or landscape before converting to a texture.
            content_fit = self._get_content_fit_for_pixbuf(pixbuf)
            self.picture.set_content_fit(content_fit)

            texture = Gdk.Texture.new_for_pixbuf(pixbuf)
            self.picture.set_paintable(texture)
            self.current_index = index

            orientation = "portrait" if pixbuf.get_height() > pixbuf.get_width() else "landscape"
            logger.debug(
                f"Displaying ({orientation}, fit={content_fit.value_nick}): "
                f"{os.path.basename(image_path) if not image_path.startswith('http') else 'visitor image'}"
            )
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

    def _load_person_sketches(self, sketch_dir: str) -> list[str]:
        """Diagnostic mode: this person's own face-swap sketches.

        Reads <sketch_dir>/* (i.e. diagnostic_gallery/<slug>/sketches/).
        Returns [] if the folder is missing or empty — e.g. the person is seeded
        but their face-swap render hasn't been dropped in yet.
        """
        if not sketch_dir or not os.path.isdir(sketch_dir):
            logger.warning(f"Diagnostic: no sketches folder for this person ({sketch_dir!r})")
            return []
        files: list[str] = []
        for ext in self.config.image_extensions:
            files.extend(glob.glob(os.path.join(sketch_dir, ext)))
        files.sort(key=lambda p: os.path.basename(p).lower())
        logger.info(f"Diagnostic: loaded {len(files)} sketch(es) from {sketch_dir}")
        return files

    def switch_to_visitor_mode(self, visit_id: str, person_name: str, sketch_dir: str = None):
        """Switch to visitor-specific images mode."""
        logger.info(f"Switching to visitor mode: {person_name} (visit_id: {visit_id}, sketches: {sketch_dir})")

        # Diagnostic mode shows this person's own local sketches; normal mode
        # fetches this visitor's generated images from the server.
        if self.service.diagnostic_mode:
            visitor_images = self._load_person_sketches(sketch_dir)
        else:
            visitor_images = self._fetch_visitor_images(visit_id)

        if not visitor_images:
            logger.warning(
                f"No sketches to show for {person_name} (visit_id: {visit_id}); "
                f"staying on stock images"
            )
            return

        self.visitor_images = visitor_images

        # This runs on the face-recognition background thread (the event is
        # published synchronously), but set_images touches GTK widgets — which
        # is only safe on the GTK main loop. Marshal it across with idle_add.
        if self.current_window:
            GLib.idle_add(self.current_window.set_images, self.visitor_images, "visitor")

        # Update overlay to show mode change
        with self.service._overlay_lock:
            self.service._overlay_text = f"Welcome {person_name}!"
            self.service._overlay_hide_time = time.time() + 3

    def switch_to_stock_mode(self):
        """Switch back to stock images mode."""
        logger.info("Switching to stock mode")

        # Same as switch_to_visitor_mode: marshal the GTK update onto the main loop.
        if self.current_window:
            GLib.idle_add(self.current_window.set_images, self.stock_images, "stock")

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
