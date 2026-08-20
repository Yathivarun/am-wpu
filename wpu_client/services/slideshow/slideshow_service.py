"""Slideshow service - displays images in full-screen mode."""

import glob
import logging
import os
import tempfile
import threading
import time
from typing import Optional
from urllib.parse import urlparse

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Gst", "1.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gst, Gtk

from wpu_client.config.settings import SlideshowConfig
from wpu_client.core.events import Event, EventBus
from wpu_client.core.service_base import ServiceBase
from wpu_client.utils.http import HTTPClient

logger = logging.getLogger(__name__)

# Video slides advance on playback completion (Gtk.MediaStream's `ended`
# property), NOT after config.advance_time — see SlideshowWindow._load_video()
# / _on_video_ended(). This fixed duration is only a FALLBACK, used when a
# video's duration can't be determined (e.g. an unusual/partial file) and we
# can't rely on `ended` firing in a timely way — see _on_video_duration_known().
VIDEO_DISPLAY_SECONDS = 10

# --- Why video was rendering blank ---------------------------------------
# The previous implementation played video via Gtk.MediaFile, which (on the
# GStreamer backend) paints frames onto its Gdk.Paintable using the
# `gtk4paintablesink` GStreamer element. That element ships in a separate
# system package (typically `gstreamer1.0-gtk4`) that is NOT in this
# project's dependency list (see README's System Dependencies) and is not
# available at all on some Debian/Raspberry Pi OS releases (it needs a fairly
# recent GStreamer). Without it, GStreamer silently falls back to a sink that
# never touches the widget: the pipeline still reaches PLAYING, `duration`
# and `ended` still fire correctly (which is why timing/advance-on-completion
# "worked"), but nothing is ever painted — a persistent blank frame.
#
# Fix: decode frames ourselves with a plain GStreamer pipeline
# (filesrc ! decodebin ! videoconvert ! appsink) and push each decoded RGBA
# frame onto the existing Gtk.Picture as a Gdk.MemoryTexture — the exact
# technique proven working in test.py. This has zero dependency on
# gtk4paintablesink, only on the plugins already in the README's apt list
# (gstreamer1.0-plugins-good/bad/ugly, gstreamer1.0-libav).
# ---------------------------------------------------------------------------

_GST_INITIALIZED = False


def _ensure_gst_init() -> None:
    global _GST_INITIALIZED
    if not _GST_INITIALIZED:
        Gst.init(None)
        _GST_INITIALIZED = True


class _GstVideoPlayer:
    """Plays one video file by manually pulling decoded RGBA frames out of a
    GStreamer pipeline via appsink, and pushing them onto a Gtk.Picture as
    Gdk.MemoryTexture frames — see the module-level note above for why this
    replaces Gtk.MediaFile.

    Deliberately has no audio branch: `decodebin`'s audio pad (if any) is
    simply never linked to anything, so no audio output element exists and
    nothing plays — satisfying "muted playback" by construction rather than
    via a mute flag on a player that might still expose an audio sink.
    """

    def __init__(self, path: str, picture: Gtk.Picture, on_eos, on_error, on_duration=None):
        _ensure_gst_init()
        self.picture = picture
        self._on_eos = on_eos
        self._on_error = on_error
        self._on_duration = on_duration
        self._duration_reported = False
        self._stopped = False

        # decodebin has dynamic ("Sometimes") pads, but Gst.parse_launch
        # supports the `decodebin ! nextelement` idiom directly: it auto-links
        # decodebin's video pad to videoconvert once caps are known. Any
        # audio pad is left unlinked (see docstring) rather than silently
        # played.
        pipeline_str = (
            f'filesrc location="{path}" ! decodebin ! videoconvert ! '
            f'appsink name=sink caps="video/x-raw,format=RGBA" sync=true '
            f'max-buffers=2 drop=true'
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.set_property("emit-signals", True)
        self._new_sample_id = self.appsink.connect("new-sample", self._on_new_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        self._bus_handler_id = bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)

    def _on_new_sample(self, sink):
        """Runs on the GStreamer streaming thread, NOT the GTK main thread —
        every touch of self.picture must go through GLib.idle_add."""
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, info = buf.map(Gst.MapFlags.READ)
        if success:
            try:
                frame_bytes = GLib.Bytes.new(info.data)
                texture = Gdk.MemoryTexture.new(
                    width, height, Gdk.MemoryFormat.R8G8B8A8,
                    frame_bytes, width * 4,
                )
                GLib.idle_add(self.picture.set_paintable, texture)
            finally:
                buf.unmap(info)

        if not self._duration_reported and self._on_duration is not None:
            ok, duration_ns = self.pipeline.query_duration(Gst.Format.TIME)
            if ok and duration_ns > 0:
                self._duration_reported = True
                duration_s = duration_ns / Gst.SECOND
                GLib.idle_add(self._on_duration, duration_s)

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """Bus watch callbacks already run on the main loop's context, but we
        still marshal through idle_add for symmetry/safety."""
        if self._stopped:
            return True
        t = message.type
        if t == Gst.MessageType.EOS:
            GLib.idle_add(self._on_eos)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            GLib.idle_add(self._on_error, err, debug)
        return True

    def stop(self) -> None:
        """Tear down the pipeline and bus watch. Safe to call more than once
        (load_image() calls this on every slide transition)."""
        if self._stopped or self.pipeline is None:
            return
        self._stopped = True
        try:
            self.appsink.disconnect(self._new_sample_id)
        except Exception:
            pass
        bus = self.pipeline.get_bus()
        try:
            bus.disconnect(self._bus_handler_id)
            bus.remove_signal_watch()
        except Exception:
            pass
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None


def visitor_image_source(
    sketch_dir: Optional[str],
    diagnostic_mode: bool,
    use_legacy_final_images: bool,
    unrecognized: bool = False,
) -> str:
    """Where a visitor's slides should come from: "local", "server" or "none".

    Asset-driven rather than mode-driven: whenever locally-composed slides
    exist for this person, they win. That covers diagnostic mode (sketch_dir
    set on a gallery match) and base mode (sketch_dir populated by local
    SAU/FRU composition).

    Two rules override that:

    * `use_legacy_final_images` is the rollback valve — it must force the
      server path even when local slides exist, or flipping it would do
      nothing.
    * Diagnostic mode is OFFLINE by definition and must never reach for the
      server, not even as a fallback. The common way to end up with no local
      slides there is an unrecognized face, whose sketch_dir is always None;
      falling through to the server would fire on every unknown visitor and
      stall the calling (recognition) thread for timeout x max_retries with
      nothing to show for it.
    * An unrecognized face has no registration at all — it is tracked under a
      generated "__unrecognized__<hex>" pseudo-id, which no endpoint can ever
      resolve. Asking anyway is a guaranteed-failing round trip on the
      recognition thread, and the server answers it with a 500 (the pseudo-id
      isn't a UUID), so it also fills the server's logs with false alarms.
      This applies in base mode too, not just diagnostic.
    """
    has_local = bool(sketch_dir) and os.path.isdir(sketch_dir)
    if diagnostic_mode:
        return "local" if has_local and not use_legacy_final_images else "none"
    if unrecognized:
        # Never the server: there is nothing there under a pseudo-id. Local
        # slides are still honoured if something ever assigns them.
        return "local" if has_local else "none"
    if use_legacy_final_images:
        return "server"
    if has_local:
        return "local"
    return "server"


def _video_suffixes(config) -> tuple[str, ...]:
    """Lowercase file extensions (e.g. '.mov') from config.video_extensions'
    glob patterns (e.g. '*.mov'), for extension-based video/image routing."""
    return tuple(os.path.splitext(pattern)[1].lower() for pattern in config.video_extensions)


def _natural_key(path: str):
    """Sort key for `sort_mode: numeric` that never compares int to str.

    Digit-named files (e.g. "1.png") sort first, numerically; everything else
    sorts after, alphabetically. A mixed directory (numeric stock images next
    to a non-numeric name) previously crashed with `TypeError: '<' not
    supported between instances of 'int' and 'str'` because the old key
    returned an int for one file and a str for another.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    # (0, number) sorts numeric names first & numerically; (1, name) sorts the
    # rest — the differing first element means Python never compares the
    # second (int vs str) across groups.
    return (0, int(name)) if name.isdigit() else (1, name.lower())


class SlideshowService(ServiceBase):
    """
    GTK4-based slideshow service.

    Supports two modes:
    - Stock images mode: Default slideshow from local directory
    - Visitor images mode: Slideshow of visitor-specific WPU images
    """

    def __init__(self, config: SlideshowConfig, event_bus: EventBus, wpu_endpoint: str,
                 face_service=None, diagnostic_mode: bool = False,
                 use_legacy_final_images: bool = False):
        """
        Initialize the slideshow service.

        Args:
            config: Slideshow configuration
            event_bus: Event bus for inter-service communication
            wpu_endpoint: WPU API endpoint for fetching visitor images
            face_service: Optional FaceRecognitionService for camera preview
            diagnostic_mode: If True, show the matched person's own local sketches
                (data/embeddings/<slug>/sketches/) on recognition instead of
                fetching visitor images from the server.
            use_legacy_final_images: Rollback valve. If True, always fetch the
                server's pre-composed final images and ignore locally-composed
                slides. Lives on the face_recognition config (it describes what
                wpu_endpoint is expected to serve) and is passed in here, the
                same way diagnostic_mode is.
        """
        super().__init__("slideshow")
        self.config = config
        self.event_bus = event_bus
        self.wpu_endpoint = wpu_endpoint
        self.face_service = face_service  # used for bottom-left camera preview
        self.diagnostic_mode = diagnostic_mode
        self.use_legacy_final_images = use_legacy_final_images
        self._app: Optional[SlideshowApp] = None
        self._overlay_text: Optional[str] = None
        self._overlay_hide_time: float = 0
        self._overlay_lock = threading.Lock()

        # Mode tracking
        self._mode: str = "stock"  # "stock" or "visitor"
        self._mode_lock = threading.Lock()
        self._current_registration_id: Optional[str] = None

        # HTTP client for fetching WPU images
        self._http_client: Optional[HTTPClient] = None

        # Subscribe to events
        event_bus.subscribe("faces.recognized", self._on_faces_recognized)
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

    def _on_faces_recognized(self, event: Event) -> None:
        """Handle faces.recognized event - show one overlay line per currently
        tracked face (welcome / live match% / "Not recognized"), or clear the
        overlay when nobody is tracked."""
        lines = event.data.get("lines", [])
        hide_delay = event.data.get("hide_delay", 3)

        with self._overlay_lock:
            if lines:
                self._overlay_text = "\n".join(lines)
                self._overlay_hide_time = time.time() + hide_delay
            else:
                self._overlay_text = None

        logger.info(f"Displaying overlay: {lines}")

    def _on_person_detected(self, event: Event) -> None:
        """Handle person detected event - fetch visitor images and switch mode."""
        registration_id = event.data.get("registration_id")
        person_name = event.data.get("person_name", "Unknown")
        sketch_dir = event.data.get("sketch_dir")

        if not registration_id:
            logger.warning("person.detected event missing registration_id")
            return

        logger.info(
            f"Person detected: {person_name} "
            f"(registration_id: {registration_id}, sketches: {sketch_dir})"
        )

        # Fetch WPU images and switch mode
        if self._app:
            self._app.switch_to_visitor_mode(
                registration_id, person_name, sketch_dir,
                unrecognized=event.data.get("unrecognized", False),
            )

    def _on_person_left(self, event: Event) -> None:
        """Handle person left event - finish their sketch cycle, then revert."""
        person_name = event.data.get("person_name", "Unknown")
        logger.info(f"Person left: {person_name}")

        # Don't cut the slideshow short: keep showing the person's remaining
        # sketches and switch back to stock only once the full cycle is done.
        # If someone else is recognised meanwhile, person.detected switches
        # to their sketches immediately.
        if self._app:
            self._app.finish_visitor_cycle_then_stock()

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
        # Set after the person leaves (visitor mode only): keep advancing
        # through their sketches and revert to stock once the cycle wraps.
        self.finish_cycle_then_stock = False

        # Currently-playing _GstVideoPlayer (video slides only) and, if it was
        # downloaded from a URL, the temp file backing it — both None while
        # a static image is showing. Tracked so load_image() can stop/clean
        # up the outgoing video before showing the next slide.
        self._current_media: Optional[_GstVideoPlayer] = None
        self._current_media_temp_path: Optional[str] = None
        # True once the current video slide has triggered next_image() — a
        # guard so the "ended" signal and the duration-unknown fallback timer
        # (whichever fires first) can't both advance the same slide.
        self._video_advanced: bool = False

        # The slideshow advance timer's GLib source id. Declared here (not
        # just at the bottom of __init__) because load_image() now manages
        # it directly — each slide reschedules it for its own duration
        # (10s for video, config.advance_time for images) instead of one
        # fixed interval for everything.
        self.timeout_id: Optional[int] = None

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

        # Set default to FILL for anamorphic stretching
        self.picture.set_content_fit(Gtk.ContentFit.FILL)

        # Create overlay label for face recognition results
        self.overlay_label = Gtk.Label()
        self.overlay_label.set_halign(Gtk.Align.START)
        self.overlay_label.set_visible(False)

        # --- Camera preview ---
        # Size is enforced by downscaling the frame itself (see
        # update_camera_preview): Gtk.Picture's natural size is its paintable's
        # size, so a size request smaller than the texture has no effect.
        # getattr fallbacks keep this working if SlideshowConfig lacks the keys.
        self.preview_size = (
            getattr(config, "camera_preview_width", 160),
            getattr(config, "camera_preview_height", 120),
        )
        self.preview_enabled = getattr(config, "camera_preview_enabled", True)
        self.camera_preview = Gtk.Picture()
        self.camera_preview.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.camera_preview.set_visible(False)  # hidden until first frame arrives

        # Stack preview + label in a vertical box, anchored to the configured
        # corner: top-left, top-right, bottom-left, or bottom-right.
        position = getattr(config, "camera_preview_position", "bottom-left")
        vpos, _, hpos = position.partition("-")
        self.preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.preview_box.set_halign(Gtk.Align.END if hpos == "right" else Gtk.Align.START)
        self.preview_box.set_valign(Gtk.Align.END if vpos == "bottom" else Gtk.Align.START)
        if hpos == "right":
            self.preview_box.set_margin_end(16)
        else:
            self.preview_box.set_margin_start(16)
        if vpos == "bottom":
            self.preview_box.set_margin_bottom(16)
        else:
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

        # Load and display first image/video. load_image() schedules
        # self.timeout_id itself, using the right duration for whatever
        # this first slide turns out to be.
        if self.image_files:
            self.load_image(0)
        else:
            # Nothing to show yet (e.g. still fetching) — start a plain
            # timer so playback begins once set_images() populates the list.
            self.timeout_id = GLib.timeout_add_seconds(config.advance_time, self.next_image)

        # Set up key press controller
        key_controller = Gtk.EventControllerKey()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

        # Start overlay update timer
        self.overlay_timeout_id = GLib.timeout_add(100, self.update_overlay)

        # Start camera preview update timer — 100ms refresh (~10fps, looks live)
        self.preview_timeout_id = None
        if self.preview_enabled:
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
        self.finish_cycle_then_stock = False

        if self.image_files:
            self.load_image(0)

        logger.info(f"Updated image list: {len(new_images)} images, mode={mode}")

    def _get_content_fit_for_pixbuf(self, pixbuf: GdkPixbuf.Pixbuf) -> Gtk.ContentFit:
        """
        Original logic kept intact but bypassed in load_image to support anamorphic scaling.
        """
        width = pixbuf.get_width()
        height = pixbuf.get_height()
        is_portrait = height > width

        if is_portrait:
            return Gtk.ContentFit.CONTAIN

        scale_mode = self.config.scale_mode
        if scale_mode == "fill":
            return Gtk.ContentFit.FILL
        elif scale_mode == "fit":
            return Gtk.ContentFit.CONTAIN
        elif scale_mode == "crop":
            return Gtk.ContentFit.COVER
        else:
            return Gtk.ContentFit.FILL

    def _is_video_path(self, path: str) -> bool:
        """Whether `path` (local file or URL) is a video slide, per
        config.video_extensions. Detected by extension only — both for local
        files and for server-fetched `signed_urls`. Signed URLs are assumed to
        preserve the original filename/extension (typical for S3-style signed
        URLs); if that assumption ever breaks for some URLs, this is the spot
        to add a content-type fallback (e.g. HEAD request) instead of guessing."""
        p = urlparse(path).path if path.startswith(("http://", "https://")) else path
        return os.path.splitext(p)[1].lower() in _video_suffixes(self.config)

    def _teardown_current_media(self) -> None:
        """Stop and release whatever video is currently playing, and remove
        its temp file if it was downloaded from a URL. No-op when the
        current slide is a static image. Always called before showing the
        next slide (image or video) so an outgoing video never keeps
        decoding in the background or leaks its temp file."""
        if self._current_media is not None:
            try:
                self._current_media.stop()
            except Exception as e:
                logger.debug(f"Error stopping outgoing video: {e}")
            self._current_media = None
        self._video_advanced = False

        if self._current_media_temp_path is not None:
            try:
                os.remove(self._current_media_temp_path)
            except OSError as e:
                logger.debug(f"Error removing temp video file: {e}")
            self._current_media_temp_path = None

    def _download_video_to_temp(self, url: str) -> Optional[str]:
        """Download a remote .mov to a temp file.

        Gtk.MediaFile needs a real path on disk — unlike GdkPixbuf, it can't
        play from bytes already in memory — so a server-fetched video (normal
        mode's signed_urls) has to land on disk first. Diagnostic mode's
        sketch_dir videos are already local and skip this entirely.
        """
        content = self.service._http_client.download_binary(url)
        if content is None:
            logger.error(f"Failed to download video {url}")
            return None
        # Keep the source's real extension — GStreamer's decodebin sniffs the
        # container itself, but a truthful suffix keeps the temp file readable
        # in logs and lets any extension-based tooling behave sensibly.
        suffix = os.path.splitext(urlparse(url).path)[1] or ".mov"
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(content)
            return temp_path
        except OSError as e:
            logger.error(f"Failed to write temp video file for {url}: {e}")
            return None

    def _load_static_image(self, image_path: str) -> None:
        """Load and display a static image slide (unchanged from before)."""
        # Check if this is a URL (visitor mode) or local file (stock mode)
        if image_path.startswith("http://") or image_path.startswith("https://"):
            # Load from URL using httpx and GdkPixbuf
            logger.debug("Loading image from URL")
            content = self.service._http_client.download_binary(image_path)
            if content is None:
                raise RuntimeError(f"could not download image {image_path}")

            # Load image from bytes using GdkPixbuf
            loader = GdkPixbuf.PixbufLoader()
            loader.write(content)
            loader.close()
            pixbuf = loader.get_pixbuf()
        else:
            # Load from local file into a pixbuf so we can inspect dimensions
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(image_path)

        # =================================================================
        # ANAMORPHIC DISTORTION SCRIPT
        # 1. Forcefully stretch the loaded pixbuf to 1920x1080 natively in GTK.
        #    This ignores aspect ratio and pre-distorts the 6:7 image so the 
        #    hardware display squishing it cancels it out perfectly.
        # =================================================================
        pixbuf = pixbuf.scale_simple(1920, 1080, GdkPixbuf.InterpType.BILINEAR)

        # 2. Force FILL so GTK maps our newly stretched 1920x1080 image perfectly
        #    1:1 to the 1080p window without adding any internal letterboxing.
        self.picture.set_content_fit(Gtk.ContentFit.FILL)
        # =================================================================

        texture = Gdk.Texture.new_for_pixbuf(pixbuf)
        self.picture.set_paintable(texture)

        orientation = "portrait" if pixbuf.get_height() > pixbuf.get_width() else "landscape"
        logger.debug(
            f"Displaying ({orientation}, anamorphic stretch applied): "
            f"{os.path.basename(image_path) if not image_path.startswith('http') else 'visitor image'}"
        )

    def _load_video(self, video_path: str) -> None:
        """Load and play a video slide via a manual GStreamer appsink pipeline
        (_GstVideoPlayer), NOT Gtk.MediaFile — see the module-level note near
        VIDEO_DISPLAY_SECONDS for why Gtk.MediaFile rendered a blank frame.

        _GstVideoPlayer pushes Gdk.MemoryTexture frames onto the SAME
        self.picture widget used for images — no separate video widget.

        Anamorphic note: content_fit stays FILL, same as images. FILL always
        non-uniformly stretches whatever paintable it's given to exactly fill
        the widget's own box, ignoring the paintable's intrinsic aspect ratio
        — that's the same distortion the image path gets from its explicit
        pixbuf.scale_simple(1920, 1080, ...) pre-stretch (composing that
        pre-stretch with a further FILL-to-window-box stretch is mathematically
        identical to a single FILL-to-window-box stretch of the original
        image; the intermediate 1920x1080 step doesn't change the final
        on-screen shape). So video gets the same correction "for free" via
        FILL — each frame is a plain Gdk.MemoryTexture at the video's native
        resolution, and FILL stretches it into the window box exactly like it
        does for the pre-stretched image texture.

        Advance-on-completion: video plays to EOS and advances via
        _on_video_ended(), driven by the pipeline bus's EOS message — NOT a
        fixed timer. _on_video_duration_known() arms a fallback timer
        (VIDEO_DISPLAY_SECONDS) only if the pipeline's duration query hasn't
        succeeded yet, since EOS isn't guaranteed to fire in a timely way for
        a stream GStreamer never manages to fully parse.
        """
        if video_path.startswith("http://") or video_path.startswith("https://"):
            local_path = self._download_video_to_temp(video_path)
            if local_path is None:
                raise RuntimeError(f"could not download video {video_path}")
            temp_path = local_path
        else:
            local_path = video_path
            temp_path = None

        # Same FILL behaviour as images, for a consistent look (see docstring).
        self.picture.set_content_fit(Gtk.ContentFit.FILL)

        self._current_media_temp_path = temp_path
        self._video_advanced = False
        self._current_media = _GstVideoPlayer(
            local_path,
            self.picture,
            on_eos=self._on_video_ended,
            on_error=self._on_video_error,
            on_duration=self._on_video_duration_known,
        )

        # Safety-net fallback, armed immediately and unconditionally — not
        # only after the pipeline's duration becomes known. If GStreamer
        # can't decode the file at all (e.g. required plugins aren't
        # installed, see the README's System Dependencies), the pipeline may
        # never reach EOS or report an error — it just hangs on a black frame
        # forever. Without this, nothing would ever advance in that case.
        # _on_video_duration_known() reschedules this to the real duration
        # once known, so a legitimately long video isn't cut short by the
        # generic default.
        self.timeout_id = GLib.timeout_add_seconds(
            VIDEO_DISPLAY_SECONDS, self._on_video_fallback_timeout
        )

        logger.debug(f"Playing video: {os.path.basename(video_path)}")

    def _on_video_ended(self, *_args) -> None:
        """Advance-on-completion: fires on the pipeline bus's EOS message.
        Guarded against double-advancing alongside the duration-unknown
        fallback timer (both check/set self._video_advanced)."""
        if self._video_advanced:
            return
        self._video_advanced = True
        self.next_image()

    def _on_video_error(self, err, debug) -> None:
        """Playback failed (bad/corrupt file, unsupported codec, missing
        GStreamer plugin, etc.) — don't let a broken video stall the
        slideshow forever."""
        if self._video_advanced:
            return
        logger.error(f"Video playback error: {err} ({debug})")
        self._video_advanced = True
        self.next_image()

    def _on_video_duration_known(self, duration_s: float) -> None:
        """Once the pipeline reports a usable duration, advance-on-completion
        (via _on_video_ended) is the primary mechanism — but reschedule the
        safety-net fallback (armed unconditionally in _load_video) to the
        real duration + a small buffer, so a legitimately long video isn't
        cut short by the generic VIDEO_DISPLAY_SECONDS default."""
        if self._video_advanced:
            return
        if self.timeout_id is not None:
            GLib.source_remove(self.timeout_id)
        self.timeout_id = GLib.timeout_add_seconds(
            int(duration_s) + 2, self._on_video_fallback_timeout
        )

    def _on_video_fallback_timeout(self) -> bool:
        """Duration-unknown fallback: advance after a fixed duration if
        nothing else has advanced this slide yet."""
        self.timeout_id = None
        if self._current_media is not None and not self._video_advanced:
            self._video_advanced = True
            self.next_image()
        return False

    def load_image(self, index):
        """Load and display the slide (image or video) at given index.

        Images keep the fixed config.advance_time timer, scheduled here.
        Videos advance on playback completion instead — _load_video() wires
        up the `ended` signal (and a duration-unknown fallback timer) itself,
        so no timer is scheduled here for video; see _load_video()'s
        docstring for why a single fixed interval no longer fits a mixed
        image/video cycle.
        """
        if not self.image_files or index >= len(self.image_files):
            return True

        image_path = self.image_files[index]
        is_video = self._is_video_path(image_path)

        self._teardown_current_media()

        if self.timeout_id is not None:
            GLib.source_remove(self.timeout_id)
            self.timeout_id = None

        try:
            if is_video:
                self._load_video(image_path)  # schedules its own advance
            else:
                self._load_static_image(image_path)
                self.timeout_id = GLib.timeout_add_seconds(self.config.advance_time, self.next_image)
            self.current_index = index
        except Exception as e:
            logger.error(f"Error loading slide {index} ({image_path}): {e}")
            # Don't leave the slideshow stalled on a broken slide with no
            # timer scheduled — try the next one after a normal beat.
            self.timeout_id = GLib.timeout_add_seconds(self.config.advance_time, self.next_image)
            return True

        return True

    def next_image(self):
        """Advance to next slide.

        Returns False because load_image() now reschedules self.timeout_id
        itself (with the right duration for whatever slide comes next), so
        this particular GLib timer firing should not repeat on its own —
        doing so alongside load_image's own reschedule would leave two
        timers running at once. (Previously this returned True and was the
        sole recurring timer, back when every slide used the same fixed
        config.advance_time interval.)
        """
        if not self.image_files:
            return False
        next_index = (self.current_index + 1) % len(self.image_files)
        if self.mode == "visitor" and self.finish_cycle_then_stock and next_index == 0:
            # The departed person's sketch cycle just completed — revert now.
            self.finish_cycle_then_stock = False
            app = self.get_application()
            if app:
                app.switch_to_stock_mode()
            return False
        self.load_image(next_index)  # reschedules self.timeout_id itself
        return False

    def previous_image(self):
        """Go to previous slide. load_image() reschedules the advance timer
        itself, so no separate timer reset is needed here anymore."""
        if not self.image_files:
            return
        prev_index = (self.current_index - 1) % len(self.image_files)
        self.load_image(prev_index)

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

        frame = face_service.get_preview_frame(max_size=self.preview_size)
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
            # Next slide — next_image()/load_image() reschedule the advance
            # timer themselves with the correct duration for the new slide.
            self.next_image()
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
        """Load all stock image AND video files from configured directory,
        mixed into a single sorted cycle (same directory, same extensions
        config pattern as per-person sketches)."""
        image_dir = self.config.image_directory
        image_files = []

        for ext in self.config.image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        video_count_start = len(image_files)
        for ext in self.config.video_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        video_count = len(image_files) - video_count_start

        if not image_files:
            logger.error(f"No images or videos found in {image_dir}")
            return []

        # Sort images/videos together based on configuration
        if self.config.sort_mode == "numeric":
            image_files.sort(key=_natural_key)
        else:
            image_files.sort(key=lambda p: os.path.basename(p).lower())

        logger.info(
            f"Loaded {len(image_files)} stock slide(s) from {image_dir} "
            f"({video_count} video(s))"
        )
        return image_files

    def _fetch_visitor_images(self, registration_id: str) -> list[str]:
        """Fetch visitor WPU images from the API.

        Single attempt, never raises. This runs synchronously on the
        recognition thread (the person.detected handler), so retrying a
        failure would stall detection for timeout x max_retries while the
        visitor is still standing there — and the overwhelmingly common
        failure is a definitive one (no images for this registration) that
        no amount of retrying will change.
        """
        logger.info(f"Fetching WPU images for registration_id: {registration_id}")
        response_data = self.http_client.get_json_once(
            self.wpu_endpoint, params={"registration_id": registration_id}
        )
        if not response_data:
            logger.info(f"No WPU images available for visit {registration_id}")
            return []

        # The API returns a dict with "signed_urls" key containing list of URLs
        images = response_data.get("signed_urls", [])
        logger.info(f"Fetched {len(images)} WPU images for visit {registration_id}")
        return images

    def _load_person_sketches(self, sketch_dir: str) -> list[str]:
        """This person's locally-available slides, images and videos together.

        Reads a single flat <sketch_dir>/, whichever mode produced it:
        diagnostic mode's seeded sketches (data/embeddings/<slug>/sketches/),
        or base mode's locally-composed SAU/FRU scenes plus hardlinked videos
        (data/base_assets/<registration_id>/display/, or the duo equivalent).
        The body is mode-agnostic — it just globs a directory.

        Returns [] if the folder is missing or empty — e.g. the person is
        known but nothing has been composed or dropped in for them yet.
        """
        if not sketch_dir or not os.path.isdir(sketch_dir):
            logger.warning(f"No slides folder for this person ({sketch_dir!r})")
            return []
        files: list[str] = []
        for ext in self.config.image_extensions:
            files.extend(glob.glob(os.path.join(sketch_dir, ext)))
        # Videos live alongside the image sketches in the same folder and are
        # mixed into the same cycle.
        for ext in self.config.video_extensions:
            files.extend(glob.glob(os.path.join(sketch_dir, ext)))
        files.sort(key=lambda p: os.path.basename(p).lower())
        logger.info(f"Loaded {len(files)} local slide(s)/video(s) from {sketch_dir}")
        return files

    def switch_to_visitor_mode(self, registration_id: str, person_name: str,
                               sketch_dir: str = None, unrecognized: bool = False):
        """Switch to visitor-specific images mode.

        `unrecognized` marks the unknown-face preview: a random person's
        sketches are shown with a 'Not recognized' overlay instead of a welcome.
        """
        logger.info(
            f"Switching to visitor mode: {person_name} "
            f"(registration_id: {registration_id}, sketches: {sketch_dir})"
        )

        # See visitor_image_source() for the full policy, including why
        # diagnostic mode must never fall through to the server.
        source = visitor_image_source(
            sketch_dir,
            self.service.diagnostic_mode,
            self.service.use_legacy_final_images,
            unrecognized=unrecognized,
        )
        if source == "local":
            visitor_images = self._load_person_sketches(sketch_dir)
        elif source == "server":
            visitor_images = self._fetch_visitor_images(registration_id)
        else:
            logger.info(
                f"No local slides for {person_name} and the server is not an "
                f"option in this mode — staying on stock images"
            )
            visitor_images = []

        if not visitor_images:
            logger.warning(
                f"No sketches to show for {person_name} (registration_id: {registration_id}); "
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
            self.service._overlay_text = (
                "Not recognized" if unrecognized else f"Welcome {person_name}!"
            )
            self.service._overlay_hide_time = time.time() + 3

    def finish_visitor_cycle_then_stock(self):
        """Person left: let their remaining sketches display; the window
        reverts to stock when the visitor cycle wraps (see next_image)."""
        def _flag():
            win = self.current_window
            if win and win.mode == "visitor":
                win.finish_cycle_then_stock = True
                logger.info(
                    "Person left — finishing their sketch cycle before "
                    "reverting to stock images"
                )
            return False  # one-shot idle callback

        # Called from the face-recognition thread; the flag is read by GTK
        # main-loop timers, so marshal the write onto the main loop too.
        GLib.idle_add(_flag)

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