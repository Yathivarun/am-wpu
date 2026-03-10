"""Configuration management for WPU Client."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class SlideshowConfig(BaseModel):
    """Configuration for slideshow service."""

    enabled: bool = True
    full_screen: bool = True
    advance_time: int = 3
    image_directory: str = "stock_images"
    image_extensions: list[str] = Field(
        default_factory=lambda: ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"]
    )
    background_color: str = "black"
    scale_mode: str = "fill"  # fill, fit, crop
    sort_mode: str = "numeric"  # numeric, alphabetical


class FaceRecognitionConfig(BaseModel):
    """Configuration for face recognition service."""

    enabled: bool = True
    camera_id: int = 0
    n: int = 4
    api_endpoint: str = "http://localhost:8000/api/v1/identify/"
    wpu_endpoint: str = "http://localhost:8000/api/v1/wpu/images"
    detection_interval: int = 5  # seconds between captures
    min_face_size: int = 100  # minimum face size in pixels
    display_result: bool = True  # show result on screen overlay
    overlay_hide_delay: int = 3  # seconds to hide overlay
    person_timeout: int = 10  # seconds without face before person is considered left


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
            config_path = Path.cwd() / "config.yaml"
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
        """Create a default configuration file."""
        default_config = {
            "log_level": "INFO",
            "services": {
                "slideshow": {
                    "enabled": True,
                    "full_screen": True,
                    "advance_time": 3,
                    "image_directory": "stock_images",
                    "background_color": "black",
                    "scale_mode": "fill",
                    "sort_mode": "numeric",
                },
                "face_recognition": {
                    "enabled": True,
                    "camera_id": 0,
                    "n": 4,
                    "api_endpoint": "http://localhost:8000/api/v1/identify/",
                    "wpu_endpoint": "http://localhost:8000/api/v1/wpu/images",
                    "detection_interval": 5,
                    "min_face_size": 100,
                    "display_result": True,
                    "overlay_hide_delay": 3,
                    "person_timeout": 10,
                },
            },
        }

        with open(path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

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
