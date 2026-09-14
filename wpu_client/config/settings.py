"""Configuration management for WPU Client."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from wpu_client.paths import CONFIG_DIR


class SlideshowConfig(BaseModel):
    """Configuration for slideshow service."""

    enabled: bool = True
    full_screen: bool = True
    advance_time: int = 3
    image_directory: str = "data/stock_images"
    image_extensions: list[str] = Field(
        default_factory=lambda: ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]
    )
    # Video slides mixed in alongside images (stock and per-person sketches).
    # Played via Gtk.MediaFile; always muted, since this is a kiosk display.
    video_extensions: list[str] = Field(default_factory=lambda: ["*.mov", "*.mp4"])
    background_color: str = "black"
    scale_mode: str = "fill"  # fill, fit, crop
    sort_mode: str = "alphabetical"  # numeric, alphabetical


class FaceRecognitionConfig(BaseModel):
    """Configuration for face recognition service."""

    enabled: bool = True
    camera_id: int = 0
    n: int = 3
    # Recognition model — selects both the on-device embedder and the server
    # gallery the identify request queries. These are two disjoint vector
    # spaces, so a mismatch here doesn't degrade accuracy, it returns nothing.
    #   "mobilenet"  MobileFaceNet 512-D → server's `auraface` gallery. DEFAULT,
    #                and the only one registrations are actually enrolled into.
    #   "sface"      OpenCV SFace 128-D → server's `sface` gallery. Lighter on
    #                CPU, but that gallery is empty on the live cluster, so this
    #                matches nobody. Kept for the local diagnostic gallery and
    #                for re-testing if enrolment ever moves back to 128-D.
    model: str = "mobilenet"
    # Live cluster host. Every endpoint below points at the same server; only
    # the host needs changing for a different deployment.
    api_endpoint: str = "http://192.168.1.19:8000/api/v1/identify/"
    wpu_endpoint: str = "http://192.168.1.19:8000/api/v1/wpu/images"
    detection_interval: int = 5  # seconds between captures

    # ── Base-mode local composition ─────────────────────────────────────
    # Base (server-recognition) mode composes each visitor's slides ON THE PI
    # from two raw alpha cutouts the server serves via wpu_endpoint — a body
    # cutout ("SAU") and a face cutout ("FRU") — instead of downloading
    # pre-composed final images. Backgrounds/placement come from the local
    # data/base_scenes/ configs.
    #
    # The server labels the two cutouts explicitly in its response; these
    # filenames are only the fallback used to pick them out of the older
    # unlabelled `signed_urls` list, and match what it uploads today
    # (<registration_id>/wpu/<name>).
    sau_cutout_filename: str = "sau_cutout.png"
    fru_cutout_filename: str = "fru_cutout.png"

    # SAU videos. Base of the media route — the registration id is appended as
    # a path segment (/api/v1/sau/media/<registration_id>), unlike every other
    # endpoint here, which takes it as a query param.
    #
    # Videos are identified by POSITION, not by name: the server stores each
    # upload under a generated {uuid}{ext} key, so the capture station's
    # filename never reaches us. The station uploads the composited clip
    # first, so video_count=1 takes exactly that one and skips the raw
    # captures behind it. Raise it only to show those too.
    sau_media_endpoint: str = "http://192.168.1.19:8000/api/v1/sau/media"
    video_count: int = 1
    # Rollback valve: True restores the old behaviour of displaying the
    # server's pre-composed final images fetched from wpu_endpoint, skipping
    # local composition entirely. One config flip, no redeploy.
    use_legacy_final_images: bool = False
    # Disk budget for data/base_assets (fetched cutouts + composed slides +
    # videos). Composed output is regenerable, so once this is exceeded the
    # least-recently-used entries are evicted — duo pairs first, since they
    # grow quadratically with visitor count and rebuild from the retained
    # per-person cutouts.
    base_assets_max_bytes: int = 5 * 1024 * 1024 * 1024  # 5 GB

    min_face_size: int = 100  # minimum face size in pixels
    display_result: bool = True  # show result on screen overlay
    overlay_hide_delay: int = 3  # seconds to hide overlay
    person_timeout: int = 10  # seconds without face before person is considered left

    # Diagnostic (offline) mode — match faces against a LOCAL seeded gallery
    # instead of the server /identify API. See scripts/seed_face.py.
    diagnostic_mode: bool = False
    diagnostic_gallery_dir: str = "data/embeddings"
    # Cosine-distance gate for a local match. For unit vectors cosine_distance =
    # L2^2 / 2, so the server's L2 threshold of 1.08 corresponds to ~0.58 here;
    # 0.5 is a slightly stricter default. Tune per camera/lighting.
    diagnostic_match_threshold: float = 0.5


class ServicesConfig(BaseModel):
    """Configuration for all services."""

    slideshow: SlideshowConfig = Field(default_factory=SlideshowConfig)
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)


class Settings(BaseModel):
    """Global application settings."""

    log_level: str = "INFO"
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "Settings":
        """
        Load settings from YAML file.

        Args:
            config_path: Path to config file. Defaults to config.yaml in current directory.

        Returns:
            Settings instance with values from file or defaults.
        """
        if config_path is None:
            config_path = CONFIG_DIR / "config.yaml"
        else:
            config_path = Path(config_path)

        config_data = {}

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f) or {}
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}. Using defaults.")
        else:
            print(f"Config file not found at {config_path}. Using defaults.")
            print("Creating default config file...")
            cls.create_default_config(config_path)

        return cls(**config_data)

    @classmethod
    def create_default_config(cls, path: Path) -> None:
        """Write a config file containing this schema's defaults, verbatim.

        Dumped from the models rather than a hand-written literal. The literal
        that used to live here silently drifted: it still said `model: sface`
        and pointed both endpoints at localhost long after the defaults moved
        to mobilenet and the live cluster, so any device that auto-generated
        its config came up querying an empty gallery on the wrong host.

        This is the fallback for a device with no config at all. The documented
        starting point is config/config.yaml.example, which carries the same
        values plus the comments explaining them.
        """
        with open(path, "w") as f:
            f.write("# Auto-generated from the built-in defaults.\n")
            f.write("# See config/config.yaml.example for the commented template.\n")
            yaml.dump(cls().model_dump(), f, default_flow_style=False, sort_keys=False)

        print(f"Created default config at {path}")

    def get_service_config(self, service_name: str) -> Any:
        """
        Get configuration for a specific service.

        Args:
            service_name: Name of the service (e.g., 'slideshow', 'face_recognition')

        Returns:
            Service-specific configuration object.
        """
        return getattr(self.services, service_name, None)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, loading if necessary."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def reload_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Reload settings from file."""
    global _settings
    _settings = Settings.load(config_path)
    return _settings